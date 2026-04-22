# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA-aware KV cache eviction policies.

Four coupling strategies that coordinate LoRA adapter residency with
KV-cache block eviction decisions, ordered from hardest to softest:

- **Tight coupling**: Evicts entire (adapter, block-set) groups atomically.
  When an adapter is not in the live set, *all* its blocks are removed
  together.  Best when adapters retire permanently; suffers under GPU-slot
  churn because "non-live right now" is treated as "gone forever".

- **Loose coupling**: Hard-protects blocks of live adapters, forcing
  eviction to target orphaned blocks first.  Falls back to unrestricted
  eviction if orphans are exhausted.  Suffers for the dual reason: it
  over-protects stale live-adapter blocks and over-attacks briefly-orphaned
  sticky-adapter blocks.

- **Hysteresis coupling**: Like tight, but the non-live signal is delayed
  by a configurable number of ``update_live_adapters`` calls.  An adapter
  must be non-live continuously for the full window before its blocks are
  eligible for group eviction.  Filters GPU-slot churn.

- **Soft-boost coupling**: No protection, no forced attack.  When an
  adapter transitions into the live set, its blocks are ``touch()``-ed in
  the inner policy so they're treated as recently-used.  All eviction
  decisions then defer to the inner policy's natural ordering.  Lets
  scan-resistant inner policies (S3-FIFO, SIEVE) apply their full logic.

All four policies wrap an arbitrary base ``CachePolicy`` via the decorator
pattern and are themselves ``CachePolicy`` subclasses.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import (
    BlockStatus,
    CachePolicy,
    PolicyStats,
)


class _LoRABookkeeping:
    """Shared adapter-tracking state used by both coupling policies."""

    def __init__(self) -> None:
        # block_hash -> adapter_id (None = no adapter / base model)
        self.block_to_adapter: dict[BlockHash, str | None] = {}
        # adapter_id -> set of block hashes belonging to that adapter
        self.adapter_to_blocks: dict[str, set[BlockHash]] = defaultdict(set)
        # adapters currently loaded on GPU
        self.live_adapters: set[str] = set()

    def register(self, block_hash: BlockHash, adapter_id: str | None) -> None:
        self.block_to_adapter[block_hash] = adapter_id
        if adapter_id is not None:
            self.adapter_to_blocks[adapter_id].add(block_hash)

    def unregister(self, block_hash: BlockHash) -> None:
        adapter_id = self.block_to_adapter.pop(block_hash, None)
        if adapter_id is not None:
            blocks = self.adapter_to_blocks.get(adapter_id)
            if blocks is not None:
                blocks.discard(block_hash)
                if not blocks:
                    del self.adapter_to_blocks[adapter_id]

    def update_live(self, adapters: set[str]) -> None:
        self.live_adapters = set(adapters)

    def get_live_block_hashes(self) -> set[BlockHash]:
        """Return all block hashes belonging to live adapters."""
        result: set[BlockHash] = set()
        for adapter_id in self.live_adapters:
            blocks = self.adapter_to_blocks.get(adapter_id)
            if blocks:
                result.update(blocks)
        return result

    def get_non_live_adapters(self) -> list[str]:
        """Return adapter IDs not currently on GPU, sorted by block count
        (fewest first — to minimise over-eviction in tight coupling)."""
        non_live = [
            aid for aid in self.adapter_to_blocks
            if aid not in self.live_adapters and self.adapter_to_blocks[aid]
        ]
        non_live.sort(key=lambda aid: len(self.adapter_to_blocks[aid]))
        return non_live


class LoRATightCouplingPolicy(CachePolicy):
    """
    Tight coupling: (adapter, block-set) is a single eviction unit.

    When eviction is needed, this policy selects non-live adapters (those
    not currently on GPU) and evicts *all* their blocks atomically.
    Adapters with fewer blocks are preferred to minimise over-eviction.
    Falls back to the base policy for non-adapter blocks or when no full
    adapter group can satisfy the request.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self.stats = PolicyStats()

    # --- adapter metadata ---

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

    # --- CachePolicy interface ---

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self._inner.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self._inner.insert(block_hash, block)

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        self._inner.remove(block_hash)
        self._book.unregister(block_hash)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        bh_list = list(block_hashes)
        self._inner.touch(bh_list)
        self.stats.touch_blocks += len(bh_list)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1

        if n == 0:
            return []

        # Phase 1: Try to evict whole adapter groups (non-live adapters first).
        result: list[tuple[BlockHash, BlockStatus]] = []
        adapters_to_evict: list[str] = []

        for adapter_id in self._book.get_non_live_adapters():
            if len(result) >= n:
                break
            blocks = self._book.adapter_to_blocks.get(adapter_id, set())
            # Collect evictable blocks for this adapter
            group: list[tuple[BlockHash, BlockStatus]] = []
            for bh in blocks:
                if bh in protected:
                    continue
                block = self._inner.get(bh)
                if block is not None and block.ref_cnt == 0:
                    group.append((bh, block))

            if group:
                result.extend(group)
                adapters_to_evict.append(adapter_id)

        if len(result) >= n:
            # Trim excess — we may have collected more than n.
            # Keep full adapter groups up to the point where we have >= n.
            trimmed: list[tuple[BlockHash, BlockStatus]] = []
            for adapter_id in adapters_to_evict:
                group = [
                    (bh, bl) for bh, bl in result
                    if self._book.block_to_adapter.get(bh) == adapter_id
                ]
                trimmed.extend(group)
                if len(trimmed) >= n:
                    break
            result = trimmed

            # Commit: remove from inner policy and bookkeeping
            for bh, _ in result:
                self._inner.remove(bh)
                self._book.unregister(bh)

            self.stats.evict_blocks += len(result)
            self.stats.evict_scan_steps += len(result)
            return result

        # Phase 2: Not enough adapter-group blocks — fall back to inner policy.
        # Roll back (we haven't committed anything yet).
        result.clear()
        fallback = self._inner.evict(n, protected)
        if fallback is None:
            self.stats.evict_failed += 1
            return None

        # Clean up bookkeeping for whatever the inner policy evicted.
        for bh, _ in fallback:
            self._book.unregister(bh)

        self.stats.evict_blocks += len(fallback)
        return fallback


class LoRALooseCouplingPolicy(CachePolicy):
    """
    Loose coupling: block eviction priority adjusted by adapter residency.

    Blocks whose adapters are no longer on GPU ("orphaned blocks") are
    preferred for eviction but not immediately discarded.  This preserves
    their value if the adapter is reloaded later (only adapter weights need
    to be transferred, not the full prefill recomputed).

    Implementation: when evicting, first try to evict only from
    non-live-adapter blocks (by protecting live-adapter blocks).  If that
    fails, fall back to unrestricted eviction.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self.stats = PolicyStats()

    # --- adapter metadata ---

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

    # --- CachePolicy interface ---

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self._inner.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self._inner.insert(block_hash, block)

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        self._inner.remove(block_hash)
        self._book.unregister(block_hash)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        bh_list = list(block_hashes)
        self._inner.touch(bh_list)
        self.stats.touch_blocks += len(bh_list)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1

        if n == 0:
            return []

        # First pass: try evicting only orphaned blocks (protect live-adapter
        # blocks so the inner policy skips them).
        live_blocks = self._book.get_live_block_hashes()
        extended_protected = protected | live_blocks

        result = self._inner.evict(n, extended_protected)
        if result is not None:
            for bh, _ in result:
                self._book.unregister(bh)
            self.stats.evict_blocks += len(result)
            return result

        # Second pass: not enough orphaned blocks — fall back to unrestricted
        # eviction (the base policy decides based on its own ordering).
        result = self._inner.evict(n, protected)
        if result is None:
            self.stats.evict_failed += 1
            return None

        for bh, _ in result:
            self._book.unregister(bh)
        self.stats.evict_blocks += len(result)
        return result


class LoRAHysteresisCouplingPolicy(CachePolicy):
    """
    Tight coupling with hysteresis on the live -> non-live transition.

    An adapter becomes eligible for group eviction only after it has been
    continuously non-live for ``hysteresis_steps`` consecutive
    ``update_live_adapters`` calls.  This filters out GPU-slot churn where
    adapters briefly rotate in and out of the live set.

    Eligible-adapter group eviction uses the same logic as
    :class:`LoRATightCouplingPolicy`: adapters with fewer blocks are
    preferred first (to minimise over-eviction).  Falls back to the inner
    policy when no eligible group satisfies the request.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
        hysteresis_steps: int = 20,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self._hysteresis = hysteresis_steps
        self._step = 0
        # adapter_id -> step at which it last left the live set
        self._left_live_at: dict[str, int] = {}
        self.stats = PolicyStats()

    # --- adapter metadata ---

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        old = self._book.live_adapters
        new = set(adapters)
        for aid in old - new:
            self._left_live_at[aid] = self._step
        for aid in new - old:
            self._left_live_at.pop(aid, None)
        self._book.update_live(new)
        self._step += 1

    def _eligible_adapters(self) -> list[str]:
        """Non-live adapters whose hysteresis window has expired, sorted
        by block count (fewest first)."""
        eligible = [
            aid for aid, left_at in self._left_live_at.items()
            if self._step - left_at >= self._hysteresis
            and self._book.adapter_to_blocks.get(aid)
        ]
        eligible.sort(key=lambda aid: len(self._book.adapter_to_blocks[aid]))
        return eligible

    # --- CachePolicy interface ---

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self._inner.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self._inner.insert(block_hash, block)

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        self._inner.remove(block_hash)
        self._book.unregister(block_hash)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        bh_list = list(block_hashes)
        self._inner.touch(bh_list)
        self.stats.touch_blocks += len(bh_list)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        result: list[tuple[BlockHash, BlockStatus]] = []
        adapters_to_evict: list[str] = []

        for adapter_id in self._eligible_adapters():
            if len(result) >= n:
                break
            blocks = self._book.adapter_to_blocks.get(adapter_id, set())
            group: list[tuple[BlockHash, BlockStatus]] = []
            for bh in blocks:
                if bh in protected:
                    continue
                block = self._inner.get(bh)
                if block is not None and block.ref_cnt == 0:
                    group.append((bh, block))
            if group:
                result.extend(group)
                adapters_to_evict.append(adapter_id)

        if len(result) >= n:
            trimmed: list[tuple[BlockHash, BlockStatus]] = []
            for adapter_id in adapters_to_evict:
                group = [
                    (bh, bl) for bh, bl in result
                    if self._book.block_to_adapter.get(bh) == adapter_id
                ]
                trimmed.extend(group)
                if len(trimmed) >= n:
                    break
            for bh, _ in trimmed:
                self._inner.remove(bh)
                self._book.unregister(bh)
            self.stats.evict_blocks += len(trimmed)
            self.stats.evict_scan_steps += len(trimmed)
            return trimmed

        # Fallback: no eligible group satisfies the request.
        fallback = self._inner.evict(n, protected)
        if fallback is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in fallback:
            self._book.unregister(bh)
        self.stats.evict_blocks += len(fallback)
        return fallback


class LoRASoftBoostCouplingPolicy(CachePolicy):
    """
    Soft coupling: boost recency of blocks when their adapter goes live.

    No hard protection, no forced orphan-first eviction.  When an adapter
    transitions into the live set, its blocks are ``touch()``-ed in the
    inner policy so the inner policy treats them as recently-used.  All
    eviction decisions then defer to the inner policy's natural ordering.

    This preserves the full power of scan-resistant inner policies
    (S3-FIFO, SIEVE) while providing adapter-level recency signalling:
    blocks naturally drift toward eviction when their adapter stops being
    used, but within each adapter's blocks, the inner policy's per-block
    ordering still applies.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self.stats = PolicyStats()

    # --- adapter metadata ---

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        old = self._book.live_adapters
        new = set(adapters)
        # Adapters that just transitioned into the live set: touch their
        # blocks so the inner policy sees them as recently-used.
        for aid in new - old:
            blocks = self._book.adapter_to_blocks.get(aid)
            if blocks:
                self._inner.touch(list(blocks))
        self._book.update_live(new)

    # --- CachePolicy interface ---

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self._inner.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self._inner.insert(block_hash, block)

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        self._inner.remove(block_hash)
        self._book.unregister(block_hash)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        bh_list = list(block_hashes)
        self._inner.touch(bh_list)
        self.stats.touch_blocks += len(bh_list)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        result = self._inner.evict(n, protected)
        if result is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in result:
            self._book.unregister(bh)
        self.stats.evict_blocks += len(result)
        return result
