# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA-aware KV cache eviction policies.

Coupling strategies that coordinate LoRA adapter residency with KV-cache
block eviction decisions.  Grouped by design philosophy:

**Binary residency signal (early / baseline):**
- ``LoRATightCouplingPolicy`` — evicts entire non-live adapter groups.
- ``LoRALooseCouplingPolicy`` — protects all live-adapter blocks.
- ``LoRAHysteresisCouplingPolicy`` — tight, but with N-step debounce on
  the live→non-live transition.
- ``LoRASoftBoostCouplingPolicy`` — touches blocks on live transitions;
  defers eviction entirely to the inner policy.

**Continuous signals (refined):**
- ``LoRAFrequencyWeightedPolicy`` — decaying per-adapter frequency score;
  group-evicts the lowest-heat adapter first.
- ``LoRACorrelatedTouchPolicy`` — every block access lightly touches a
  sample of the adapter's other cached blocks.  A continuous version of
  soft-boost that does not require GPU-residency transitions.
- ``LoRABudgetPolicy`` — soft per-adapter cache-slot budgets.  When an
  adapter exceeds its share, its own oldest blocks are evicted first.
- ``LoRACostAwarePolicy`` — explicitly optimises for expected saved work,
  weighted by ``prefill_cost`` and ``reload_cost`` parameters.
- ``LoRATwoLevelLRUPolicy`` — standalone (non-decorator) policy: LRU of
  adapters, each containing LRU of blocks.  Evicts the LRU block of the
  LRU adapter.
- ``LoRAGhostListPolicy`` — tight coupling with ARC-style ghost lists.
  Adapters whose group was evicted get their blocks boosted on re-entry.

All policies except ``LoRATwoLevelLRUPolicy`` wrap an arbitrary base
``CachePolicy`` via the decorator pattern and are themselves
``CachePolicy`` subclasses.
"""

from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
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


# ---------------------------------------------------------------------------
# Refined strategies using continuous signals
# ---------------------------------------------------------------------------


def _commit_group_eviction(
    inner: CachePolicy,
    book: _LoRABookkeeping,
    result: list[tuple[BlockHash, BlockStatus]],
    adapters_to_evict: list[str],
    n: int,
) -> list[tuple[BlockHash, BlockStatus]]:
    """Trim collected blocks to smallest prefix of adapter-groups with >=n, commit."""
    trimmed: list[tuple[BlockHash, BlockStatus]] = []
    for adapter_id in adapters_to_evict:
        group = [
            (bh, bl) for bh, bl in result
            if book.block_to_adapter.get(bh) == adapter_id
        ]
        trimmed.extend(group)
        if len(trimmed) >= n:
            break
    for bh, _ in trimmed:
        inner.remove(bh)
        book.unregister(bh)
    return trimmed


class LoRAFrequencyWeightedPolicy(CachePolicy):
    """
    Group-eviction where the victim adapter is picked by decayed frequency.

    Maintains a per-adapter score that increases on every block touch and
    decays exponentially each ``update_live_adapters`` step.  The lowest-heat
    non-live adapter is evicted first, then the next lowest, until ``n``
    blocks are freed.  Continuous version of tight/hysteresis coupling.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
        decay: float = 0.9,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self._decay = decay
        self._heat: dict[str, float] = defaultdict(float)
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)
        if adapter_id is not None:
            self._heat[adapter_id] += 1.0

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)
        # Global decay each step
        for aid in list(self._heat):
            self._heat[aid] *= self._decay
            if self._heat[aid] < 1e-6:
                del self._heat[aid]

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
        # Bump per-adapter heat for each touched block.
        for bh in bh_list:
            aid = self._book.block_to_adapter.get(bh)
            if aid is not None:
                self._heat[aid] += 1.0
        self.stats.touch_blocks += len(bh_list)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        # Sort non-live adapters by ascending heat (coldest first).
        non_live = [
            aid for aid in self._book.adapter_to_blocks
            if aid not in self._book.live_adapters
            and self._book.adapter_to_blocks[aid]
        ]
        non_live.sort(key=lambda aid: self._heat.get(aid, 0.0))

        result: list[tuple[BlockHash, BlockStatus]] = []
        adapters_to_evict: list[str] = []
        for adapter_id in non_live:
            if len(result) >= n:
                break
            group: list[tuple[BlockHash, BlockStatus]] = []
            for bh in self._book.adapter_to_blocks.get(adapter_id, set()):
                if bh in protected:
                    continue
                block = self._inner.get(bh)
                if block is not None and block.ref_cnt == 0:
                    group.append((bh, block))
            if group:
                result.extend(group)
                adapters_to_evict.append(adapter_id)

        if len(result) >= n:
            trimmed = _commit_group_eviction(
                self._inner, self._book, result, adapters_to_evict, n
            )
            self.stats.evict_blocks += len(trimmed)
            self.stats.evict_scan_steps += len(trimmed)
            return trimmed

        fallback = self._inner.evict(n, protected)
        if fallback is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in fallback:
            self._book.unregister(bh)
        self.stats.evict_blocks += len(fallback)
        return fallback


class LoRACorrelatedTouchPolicy(CachePolicy):
    """
    On every block touch, also touch a random sample of the adapter's
    other cached blocks.

    Continuous alternative to ``LoRASoftBoostCouplingPolicy``: rather than
    waiting for an adapter to transition into the live set, every direct
    block access propagates a weaker "hotness" signal to that adapter's
    other cached blocks.  Keeps the adapter's full block pool fresh in the
    inner policy's ordering during active bursts.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
        propagation_k: int = 4,
        seed: int = 1234,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self._k = propagation_k
        self._rng = random.Random(seed)
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

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

        # Identify adapters of touched blocks, sample other cached blocks
        # of the same adapters, and touch them lightly.
        adapters_touched: set[str] = set()
        touched_set = set(bh_list)
        for bh in bh_list:
            aid = self._book.block_to_adapter.get(bh)
            if aid is not None:
                adapters_touched.add(aid)

        propagated: list[BlockHash] = []
        for aid in adapters_touched:
            pool = self._book.adapter_to_blocks.get(aid, set()) - touched_set
            if not pool:
                continue
            sample_size = min(self._k, len(pool))
            propagated.extend(self._rng.sample(list(pool), sample_size))
        if propagated:
            self._inner.touch(propagated)

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


class LoRABudgetPolicy(CachePolicy):
    """
    Soft per-adapter cache-slot budget.

    Each adapter seen gets an equal share of the cache (``capacity / N``
    where ``N`` is the number of adapters with cached blocks).  When an
    adapter's footprint exceeds its share, its own oldest blocks are
    preferred for eviction before any other adapter's blocks.  Prevents a
    single adapter's burst from evicting others' hot blocks.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._capacity = cache_capacity
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

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

    def _adapter_budget(self) -> int:
        num_adapters = max(1, len(self._book.adapter_to_blocks))
        return self._capacity // num_adapters

    def _over_budget_adapters(self) -> list[str]:
        """Return adapters whose footprint exceeds their fair share,
        sorted by overage descending."""
        budget = self._adapter_budget()
        overages = [
            (aid, len(blocks) - budget)
            for aid, blocks in self._book.adapter_to_blocks.items()
            if len(blocks) > budget
        ]
        overages.sort(key=lambda t: -t[1])
        return [aid for aid, _ in overages]

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        # First pass: protect all blocks EXCEPT those of over-budget adapters.
        # That is: keep protecting anything already protected, plus all blocks
        # of within-budget adapters.
        over = self._over_budget_adapters()
        if over:
            over_blocks: set[BlockHash] = set()
            for aid in over:
                over_blocks.update(self._book.adapter_to_blocks.get(aid, set()))
            all_cached = set(self._book.block_to_adapter.keys())
            within_budget = all_cached - over_blocks
            extended_protected = protected | within_budget
            result = self._inner.evict(n, extended_protected)
            if result is not None:
                for bh, _ in result:
                    self._book.unregister(bh)
                self.stats.evict_blocks += len(result)
                return result

        # Fallback: unrestricted eviction.
        result = self._inner.evict(n, protected)
        if result is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in result:
            self._book.unregister(bh)
        self.stats.evict_blocks += len(result)
        return result


class LoRACostAwarePolicy(CachePolicy):
    """
    Evict based on expected saved work.

    For each cached block ``b`` with adapter ``a``:
      value(b) = prefill_cost             if ``a`` is live
               = (prefill_cost - reload_cost)  otherwise (orphan)

    Orphan blocks are worth less because reusing them requires reloading
    the adapter first.  When ``reload_cost >= prefill_cost`` orphan blocks
    have zero or negative value and are always evicted first; when
    ``reload_cost << prefill_cost`` orphans are nearly as valuable as live.

    Implementation note: we can only express "prefer to evict X" by
    protecting the complement.  If total orphan value / live value differs,
    we try orphans first only when their total count is large enough to
    satisfy ``n``.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
        prefill_cost: float = 10.0,
        reload_cost: float = 3.0,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self._prefill_cost = prefill_cost
        self._reload_cost = reload_cost
        # Orphan value fraction relative to live value.
        self._orphan_value_ratio = max(
            0.0, (prefill_cost - reload_cost) / max(prefill_cost, 1e-6)
        )
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

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

        # When orphan value ratio is low, orphans are cheap to discard:
        # aggressively protect live blocks (like loose coupling).  When the
        # ratio is high (reload cheap relative to prefill), orphans are
        # almost as valuable as live blocks: fall back to inner ordering.
        if self._orphan_value_ratio < 0.5:
            live_blocks = self._book.get_live_block_hashes()
            result = self._inner.evict(n, protected | live_blocks)
            if result is not None:
                for bh, _ in result:
                    self._book.unregister(bh)
                self.stats.evict_blocks += len(result)
                return result

        result = self._inner.evict(n, protected)
        if result is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in result:
            self._book.unregister(bh)
        self.stats.evict_blocks += len(result)
        return result


class LoRATwoLevelLRUPolicy(CachePolicy):
    """
    Standalone two-level LRU: LRU-of-adapters, each with LRU-of-blocks.

    Unlike the other classes here, this is **not** a decorator over an
    inner policy.  It implements eviction directly: evict the LRU block
    of the LRU adapter.  A simpler, more principled alternative to tight
    coupling that supports single-block granularity.

    Non-adapter blocks (``adapter_id is None``) live in a shared bucket
    treated as its own "adapter" for ordering purposes.
    """

    _NO_ADAPTER = "__none__"

    def __init__(self, cache_capacity: int, **_kwargs: object) -> None:
        # adapter LRU order (most-recently used at end)
        self._adapter_lru: OrderedDict[str, None] = OrderedDict()
        # per-adapter block LRU (most-recently used at end)
        self._blocks: dict[str, OrderedDict[BlockHash, BlockStatus]] = {}
        # reverse index
        self._bh_to_adapter: dict[BlockHash, str] = {}
        # live-set (currently-live adapters are protected from eviction)
        self._live: set[str] = set()
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        # Two-level LRU requires adapter metadata at insert-time to place
        # the block in the correct bucket.  If register is called after
        # insert(), we relocate the block to the correct bucket.
        aid = adapter_id or self._NO_ADAPTER
        current = self._bh_to_adapter.get(block_hash)
        if current == aid:
            return
        if current is not None:
            block = self._blocks[current].pop(block_hash, None)
            if not self._blocks[current]:
                del self._blocks[current]
                self._adapter_lru.pop(current, None)
        else:
            block = None
        if block is None:
            # Block isn't in any bucket yet — skip (will be placed by insert).
            self._bh_to_adapter[block_hash] = aid
            return
        self._blocks.setdefault(aid, OrderedDict())[block_hash] = block
        self._adapter_lru[aid] = None
        self._adapter_lru.move_to_end(aid)
        self._bh_to_adapter[block_hash] = aid

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._live = set(adapters)

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        aid = self._bh_to_adapter.get(block_hash)
        if aid is None:
            return None
        bucket = self._blocks.get(aid)
        if bucket is None:
            return None
        return bucket.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        # If register_block_adapter was called first, use that bucket;
        # otherwise default to NO_ADAPTER and let register_block_adapter
        # relocate later.
        aid = self._bh_to_adapter.get(block_hash, self._NO_ADAPTER)
        self._blocks.setdefault(aid, OrderedDict())[block_hash] = block
        self._bh_to_adapter[block_hash] = aid
        self._adapter_lru[aid] = None
        self._adapter_lru.move_to_end(aid)

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        aid = self._bh_to_adapter.pop(block_hash, None)
        if aid is None:
            return
        bucket = self._blocks.get(aid)
        if bucket is None:
            return
        bucket.pop(block_hash, None)
        if not bucket:
            del self._blocks[aid]
            self._adapter_lru.pop(aid, None)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        count = 0
        for bh in block_hashes:
            aid = self._bh_to_adapter.get(bh)
            if aid is None:
                continue
            bucket = self._blocks.get(aid)
            if bucket is None or bh not in bucket:
                continue
            bucket.move_to_end(bh)
            self._adapter_lru.move_to_end(aid)
            count += 1
        self.stats.touch_blocks += count

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        evicted: list[tuple[BlockHash, BlockStatus]] = []
        scan_steps = 0

        # Iterate adapters in LRU order (least-recent first); among live
        # adapters, still iterate but prefer non-live first.
        def _adapter_order() -> list[str]:
            non_live = [a for a in self._adapter_lru if a not in self._live]
            live = [a for a in self._adapter_lru if a in self._live]
            return non_live + live

        # Pre-scan for atomicity: count evictable candidates.
        evictable = 0
        for aid in _adapter_order():
            bucket = self._blocks.get(aid)
            if bucket is None:
                continue
            for bh, block in bucket.items():
                scan_steps += 1
                if bh in protected or block.ref_cnt != 0:
                    continue
                evictable += 1
                if evictable >= n:
                    break
            if evictable >= n:
                break

        if evictable < n:
            self.stats.evict_scan_steps += scan_steps
            self.stats.evict_failed += 1
            return None

        # Commit: evict LRU block of LRU adapter until we have n.
        for aid in _adapter_order():
            if len(evicted) >= n:
                break
            bucket = self._blocks.get(aid)
            if bucket is None:
                continue
            to_remove: list[BlockHash] = []
            for bh, block in bucket.items():
                if len(evicted) >= n:
                    break
                if bh in protected or block.ref_cnt != 0:
                    continue
                to_remove.append(bh)
                evicted.append((bh, block))
            for bh in to_remove:
                del bucket[bh]
                self._bh_to_adapter.pop(bh, None)
            if not bucket:
                del self._blocks[aid]
                self._adapter_lru.pop(aid, None)

        self.stats.evict_scan_steps += scan_steps
        self.stats.evict_blocks += len(evicted)
        return evicted


class LoRAGhostListPolicy(CachePolicy):
    """
    Tight coupling with ARC-style ghost lists.

    When an adapter's block group is evicted, its block hashes go into a
    "ghost" set.  If the adapter's blocks re-enter the cache (insert), we
    immediately ``touch()`` them in the inner policy so they are placed at
    the MRU end.  This repairs the primary failure mode of tight coupling:
    adapters returning after group eviction would otherwise face cold LRU
    positions and be re-evicted quickly.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
        ghost_capacity_mult: float = 2.0,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self._ghost: OrderedDict[BlockHash, str] = OrderedDict()
        self._ghost_capacity = max(1, int(cache_capacity * ghost_capacity_mult))
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)
        # If this block was recently ghost-evicted, boost by touching.
        if block_hash in self._ghost:
            del self._ghost[block_hash]
            self._inner.touch([block_hash])

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

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

    def _add_to_ghost(self, block_hash: BlockHash, adapter_id: str) -> None:
        self._ghost[block_hash] = adapter_id
        # Trim oldest ghosts when over capacity.
        while len(self._ghost) > self._ghost_capacity:
            self._ghost.popitem(last=False)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        # Tight coupling style: evict whole non-live adapter groups.
        non_live = [
            aid for aid in self._book.adapter_to_blocks
            if aid not in self._book.live_adapters
            and self._book.adapter_to_blocks[aid]
        ]
        non_live.sort(key=lambda aid: len(self._book.adapter_to_blocks[aid]))

        result: list[tuple[BlockHash, BlockStatus]] = []
        adapters_to_evict: list[str] = []
        for aid in non_live:
            if len(result) >= n:
                break
            group: list[tuple[BlockHash, BlockStatus]] = []
            for bh in self._book.adapter_to_blocks.get(aid, set()):
                if bh in protected:
                    continue
                block = self._inner.get(bh)
                if block is not None and block.ref_cnt == 0:
                    group.append((bh, block))
            if group:
                result.extend(group)
                adapters_to_evict.append(aid)

        if len(result) >= n:
            trimmed: list[tuple[BlockHash, BlockStatus]] = []
            for aid in adapters_to_evict:
                group_bhs = [
                    (bh, bl) for bh, bl in result
                    if self._book.block_to_adapter.get(bh) == aid
                ]
                trimmed.extend(group_bhs)
                if len(trimmed) >= n:
                    break
            for bh, _ in trimmed:
                aid = self._book.block_to_adapter.get(bh)
                self._inner.remove(bh)
                self._book.unregister(bh)
                if aid is not None:
                    self._add_to_ghost(bh, aid)
            self.stats.evict_blocks += len(trimmed)
            self.stats.evict_scan_steps += len(trimmed)
            return trimmed

        fallback = self._inner.evict(n, protected)
        if fallback is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in fallback:
            aid = self._book.block_to_adapter.get(bh)
            self._book.unregister(bh)
            if aid is not None:
                self._add_to_ghost(bh, aid)
        self.stats.evict_blocks += len(fallback)
        return fallback


# ---------------------------------------------------------------------------
# Advanced policies: adaptive budget, position-aware, prefix-tree-aware
# ---------------------------------------------------------------------------


class LoRAAdaptiveBudgetPolicy(CachePolicy):
    """
    Per-adapter soft cache-slot budget, sized proportionally to recent
    access frequency.

    Extension of :class:`LoRABudgetPolicy`: instead of equal budgets
    (``capacity / N``), each adapter gets
    ``capacity * freq[adapter] / sum(freq)``.  Hot adapters get more
    slots; cold adapters get fewer.  Frequencies decay each
    ``update_live_adapters`` step.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
        decay: float = 0.95,
        min_budget: int = 1,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._capacity = cache_capacity
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self._freq: dict[str, float] = defaultdict(float)
        self._decay = decay
        self._min_budget = min_budget
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)
        if adapter_id is not None:
            self._freq[adapter_id] += 1.0

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)
        for aid in list(self._freq):
            self._freq[aid] *= self._decay
            if self._freq[aid] < 1e-6:
                del self._freq[aid]

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
        for bh in bh_list:
            aid = self._book.block_to_adapter.get(bh)
            if aid is not None:
                self._freq[aid] += 1.0
        self.stats.touch_blocks += len(bh_list)

    def _adaptive_budget(self) -> dict[str, int]:
        adapters = list(self._book.adapter_to_blocks.keys())
        if not adapters:
            return {}
        total_freq = sum(self._freq.get(aid, 0.0) for aid in adapters)
        if total_freq <= 0:
            # Fallback: equal split.
            share = max(self._min_budget, self._capacity // len(adapters))
            return {aid: share for aid in adapters}
        budgets: dict[str, int] = {}
        for aid in adapters:
            w = self._freq.get(aid, 0.0) / total_freq
            budgets[aid] = max(
                self._min_budget, int(self._capacity * w)
            )
        return budgets

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        budgets = self._adaptive_budget()
        over_blocks: set[BlockHash] = set()
        all_cached: set[BlockHash] = set()
        for aid, blocks in self._book.adapter_to_blocks.items():
            all_cached.update(blocks)
            if len(blocks) > budgets.get(aid, 0):
                over_blocks.update(blocks)

        if over_blocks:
            within_budget = all_cached - over_blocks
            extended_protected = protected | within_budget
            result = self._inner.evict(n, extended_protected)
            if result is not None:
                for bh, _ in result:
                    self._book.unregister(bh)
                self.stats.evict_blocks += len(result)
                return result

        result = self._inner.evict(n, protected)
        if result is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in result:
            self._book.unregister(bh)
        self.stats.evict_blocks += len(result)
        return result


class LoRAPositionAwarePolicy(CachePolicy):
    """
    Prefix-priority: blocks at low positions (early in the sequence) are
    likely shared across requests (system prompts, instructions) and
    deserve extra recency.

    When a block is touched, if its recorded position is below
    ``prefix_threshold``, it receives ``boost_factor - 1`` additional
    touches in the inner policy.  Requires callers to register positions
    via :meth:`register_block_position`; blocks without position
    registered are treated as non-prefix.
    """

    def __init__(
        self,
        cache_capacity: int,
        inner_cls: type[CachePolicy] | None = None,
        prefix_threshold: int = 10,
        boost_factor: int = 3,
    ) -> None:
        if inner_cls is None:
            from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
            inner_cls = LRUCachePolicy
        self._inner: CachePolicy = inner_cls(cache_capacity)
        self._book = _LoRABookkeeping()
        self._positions: dict[BlockHash, int] = {}
        self._prefix_threshold = prefix_threshold
        self._boost_factor = max(1, boost_factor)
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def register_block_position(self, block_hash: BlockHash, position: int) -> None:
        # Remember the lowest observed position for this block (a block
        # shared across requests should get the prefix treatment even if
        # one caller sees it at a late position).
        current = self._positions.get(block_hash)
        if current is None or position < current:
            self._positions[block_hash] = position

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

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
        self._positions.pop(block_hash, None)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        bh_list = list(block_hashes)
        self._inner.touch(bh_list)
        prefix = [
            bh for bh in bh_list
            if self._positions.get(bh, self._prefix_threshold) < self._prefix_threshold
        ]
        for _ in range(self._boost_factor - 1):
            if prefix:
                self._inner.touch(prefix)
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
            self._positions.pop(bh, None)
        self.stats.evict_blocks += len(result)
        return result


class LoRAPrefixTreePolicy(CachePolicy):
    """
    Prefix-tree-aware eviction: prefer evicting leaf blocks.

    A block is a *leaf* if it has no cached children.  Evicting a leaf
    detaches only itself; evicting an internal block orphans all its
    descendants (they can never be reached as a cache hit without
    re-prefilling the parent chain).  This policy first tries eviction
    among leaves only; it falls back to unrestricted eviction if leaves
    are insufficient.

    Callers register parent relationships via
    :meth:`register_block_parent`; blocks without a registered parent
    are treated as roots (always leaves with respect to parent side).
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
        # Forward and reverse tree edges among *currently cached* blocks.
        self._parent: dict[BlockHash, BlockHash] = {}
        self._children: dict[BlockHash, set[BlockHash]] = defaultdict(set)
        self.stats = PolicyStats()

    def register_block_adapter(
        self, block_hash: BlockHash, adapter_id: str | None
    ) -> None:
        self._book.register(block_hash, adapter_id)

    def register_block_parent(
        self, block_hash: BlockHash, parent_hash: BlockHash | None
    ) -> None:
        if parent_hash is None:
            return
        old = self._parent.get(block_hash)
        if old is not None:
            self._children[old].discard(block_hash)
            if not self._children[old]:
                del self._children[old]
        self._parent[block_hash] = parent_hash
        self._children[parent_hash].add(block_hash)

    def update_live_adapters(self, adapters: set[str]) -> None:
        self._book.update_live(adapters)

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self._inner.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self._inner.insert(block_hash, block)

    def _detach(self, bh: BlockHash) -> None:
        """Remove a block from tree bookkeeping; promote its children as roots."""
        parent = self._parent.pop(bh, None)
        if parent is not None:
            self._children[parent].discard(bh)
            if not self._children[parent]:
                del self._children[parent]
        children = self._children.pop(bh, set())
        for child in children:
            self._parent.pop(child, None)

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        self._inner.remove(block_hash)
        self._book.unregister(block_hash)
        self._detach(block_hash)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        bh_list = list(block_hashes)
        self._inner.touch(bh_list)
        self.stats.touch_blocks += len(bh_list)

    def _non_leaves(self) -> set[BlockHash]:
        return {bh for bh, kids in self._children.items() if kids}

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        non_leaves = self._non_leaves()
        if non_leaves:
            # First pass: protect all non-leaf blocks so only leaves evict.
            result = self._inner.evict(n, protected | non_leaves)
            if result is not None:
                for bh, _ in result:
                    self._book.unregister(bh)
                    self._detach(bh)
                self.stats.evict_blocks += len(result)
                return result

        # Fallback: unrestricted eviction.
        result = self._inner.evict(n, protected)
        if result is None:
            self.stats.evict_failed += 1
            return None
        for bh, _ in result:
            self._book.unregister(bh)
            self._detach(bh)
        self.stats.evict_blocks += len(result)
        return result
