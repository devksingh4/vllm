# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import os
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.policies.abstract import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.lora_aware import (
    LoRAHysteresisCouplingPolicy,
    LoRALooseCouplingPolicy,
    LoRASoftBoostCouplingPolicy,
    LoRATightCouplingPolicy,
)
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.policies.s3fifo import S3FIFOCachePolicy
from vllm.v1.kv_offload.cpu.policies.sieve import SIEVECachePolicy
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

logger = init_logger(__name__)

_BASE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
    "sieve": SIEVECachePolicy,
    "s3fifo": S3FIFOCachePolicy,
}

_LORA_COUPLING_MODES: dict[str, type] = {
    "lora_tight": LoRATightCouplingPolicy,
    "lora_loose": LoRALooseCouplingPolicy,
    "lora_hysteresis": LoRAHysteresisCouplingPolicy,
    "lora_soft": LoRASoftBoostCouplingPolicy,
}

# Flat registry: base policies + all lora coupling combinations
_CACHE_POLICIES: dict[str, type[CachePolicy]] = dict(_BASE_POLICIES)


def _build_policy(
    name: str, capacity: int
) -> tuple[CachePolicy, str]:
    """Build a CachePolicy from a policy name string.

    Supports plain names like ``"lru"`` as well as LoRA-coupled names
    like ``"lora_tight:sieve"`` (tight coupling wrapping SIEVE).
    """
    # Check for lora coupling syntax: "lora_tight:base" or "lora_loose:base"
    if ":" in name:
        coupling_name, base_name = name.split(":", 1)
        coupling_cls = _LORA_COUPLING_MODES.get(coupling_name)
        base_cls = _BASE_POLICIES.get(base_name)
        if coupling_cls is None:
            raise ValueError(
                f"Unknown LoRA coupling mode: {coupling_name!r}. "
                f"Supported: {list(_LORA_COUPLING_MODES)}"
            )
        if base_cls is None:
            raise ValueError(
                f"Unknown base cache policy: {base_name!r}. "
                f"Supported: {list(_BASE_POLICIES)}"
            )
        return coupling_cls(cache_capacity=capacity, inner_cls=base_cls), name

    # Plain base policy
    policy_cls = _CACHE_POLICIES.get(name)
    if policy_cls is None:
        all_names = list(_CACHE_POLICIES) + [
            f"{c}:{b}"
            for c in _LORA_COUPLING_MODES
            for b in _BASE_POLICIES
        ]
        raise ValueError(
            f"Unknown cache policy: {name!r}. Supported: {all_names}"
        )
    return policy_cls(cache_capacity=capacity), name


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        cache_policy: str | None = None,
        enable_events: bool = False,
    ):
        self.block_size: int = block_size
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        if (raw_val := os.getenv("VLLM_KV_OFFLOAD_POLICY")) and raw_val:
            cache_policy = raw_val.lower()
        self._policy, self._policy_name = _build_policy(
            cache_policy or "", num_blocks
        )

        atexit.register(self._log_policy_stats)

    def _log_policy_stats(self) -> None:
        s = self._policy.stats
        touch_evict_ratio = (
            f"{s.touch_blocks / s.evict_blocks:.1f}"
            if s.evict_blocks > 0
            else "inf"
        )
        avg_scan = (
            f"{s.evict_scan_steps / s.evict_calls:.1f}"
            if s.evict_calls > 0
            else "n/a"
        )
        hit_rate = (
            f"{s.lookup_hit_blocks / s.lookup_total_blocks:.2%}"
            if s.lookup_total_blocks > 0
            else "n/a"
        )
        logger.info(
            "Policy stats [%s]: "
            "touch=%d calls (%d blocks) | "
            "evict=%d calls (%d blocks, %d failed) | "
            "avg_scan_steps=%s | "
            "touch:evict block ratio=%s | "
            "inserts=%d removes=%d gets=%d | "
            "cache_size_at_last_evict=%d | "
            "cpu_offload_hit_rate=%s (%d/%d blocks)",
            self._policy_name,
            s.touch_calls,
            s.touch_blocks,
            s.evict_calls,
            s.evict_blocks,
            s.evict_failed,
            avg_scan,
            touch_evict_ratio,
            s.insert_calls,
            s.remove_calls,
            s.get_calls,
            s.cache_size_at_last_evict,
            hit_rate,
            s.lookup_hit_blocks,
            s.lookup_total_blocks,
        )

    # --- LoRA adapter awareness ---

    def register_block_adapters(
        self, mapping: dict[BlockHash, str | None]
    ) -> None:
        """Register adapter IDs for recently inserted block hashes.

        Only effective when the active policy is LoRA-aware (tight or loose
        coupling).  Silently ignored for plain policies.
        """
        if hasattr(self._policy, "register_block_adapter"):
            for block_hash, adapter_id in mapping.items():
                self._policy.register_block_adapter(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        """Update the set of LoRA adapters currently resident on GPU.

        Only effective when the active policy is LoRA-aware.
        """
        if hasattr(self._policy, "update_live_adapters"):
            self._policy.update_live_adapters(adapters)

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        num_fresh = min(
            len(block_hashes), self._num_blocks - self._num_allocated_blocks
        )
        num_reused = len(block_hashes) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        block_hashes: Iterable[BlockHash],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        block_hashes_list = list(block_hashes)
        hit_count = 0
        for block_hash in block_hashes_list:
            block = self._policy.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        self._policy.stats.lookup_hit_blocks += hit_count
        self._policy.stats.lookup_total_blocks += len(block_hashes_list)
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self._policy.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found in cache"
            assert block.is_ready, f"Block {block_hash!r} is not ready for reading"
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self._policy.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in block_hashes:
            block = self._policy.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found"
            assert block.ref_cnt > 0, f"Block {block_hash!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes_list = list(block_hashes)

        # filter out blocks that are already stored
        block_hashes_to_store = [
            bh for bh in block_hashes_list if self._policy.get(bh) is None
        ]

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_blocks_to_evict = len(block_hashes_to_store) - self._get_num_free_blocks()

        to_evict: list[BlockHash] = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(block_hashes_list)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            for block_hash, block in evicted:
                self._free_block(block)
                to_evict.append(block_hash)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.block_size,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(
            block_hashes_to_store
        ), "Block pool did not allocate the expected number of blocks"

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self._policy.insert(block_hash, block)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        stored_block_hashes: list[BlockHash] = []

        if success:
            for block_hash in block_hashes:
                block = self._policy.get(block_hash)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self._policy.get(block_hash)
                if block is not None and not block.is_ready:
                    self._policy.remove(block_hash)
                    self._free_block(block)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.block_size,
                    medium=self.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_policy_stats(self):
        return self._policy.stats
