# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
W-TinyLFU cache eviction policy.

Reference: Einziger, Friedman, Manes, "TinyLFU: A Highly Efficient Cache
Admission Policy" (EuroSys 2014) and its windowed variant used in Caffeine.

Structure:
- A small **window cache** (default 1% of capacity) holds newly-admitted
  blocks as an LRU.  Protects against scan-bursts that would otherwise
  flood the main cache.
- A larger **main cache** split into:
  - **Probation** (20% of main) — blocks on trial after promotion from
    window.  LRU-ordered.
  - **Protected** (80% of main) — proven hot blocks.  LRU-ordered.
- A **count-min sketch** approximates per-block access frequency.  When a
  block would be evicted from window and admitted to main, its estimated
  frequency is compared against probation's LRU victim; the winner stays.

Aging: the sketch is periodically halved once total increments reach a
threshold (default ``10 * capacity``), preventing frequency values from
growing unbounded and keeping the policy responsive to shifting access
patterns.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import (
    BlockStatus,
    CachePolicy,
    PolicyStats,
)


class TinyLFUCachePolicy(CachePolicy):
    """W-TinyLFU with SLRU main cache and count-min sketch."""

    def __init__(
        self,
        cache_capacity: int,
        window_fraction: float = 0.01,
        protected_fraction: float = 0.8,
        sketch_depth: int = 4,
        sketch_width_mult: int = 4,
        age_threshold_mult: int = 10,
    ) -> None:
        self._capacity = cache_capacity
        self._window_cap = max(1, int(cache_capacity * window_fraction))
        main_cap = max(1, cache_capacity - self._window_cap)
        self._protected_cap = max(1, int(main_cap * protected_fraction))
        self._probation_cap = max(1, main_cap - self._protected_cap)

        # Three LRU segments (OrderedDict: oldest first, newest last).
        self._window: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self._probation: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self._protected: OrderedDict[BlockHash, BlockStatus] = OrderedDict()

        # Count-min sketch: `depth` hash functions, each mapping to a row
        # of `width` counters.  Frequency estimate = min across rows.
        self._sketch_depth = sketch_depth
        self._sketch_width = max(64, cache_capacity * sketch_width_mult)
        self._sketch: list[list[int]] = [
            [0] * self._sketch_width for _ in range(sketch_depth)
        ]
        # Distinct seeds per row (mixed with bytes hash via bit rotation).
        self._sketch_seeds = [0x9E3779B1 * (i + 1) for i in range(sketch_depth)]

        self._sketch_sum = 0
        self._age_threshold = age_threshold_mult * cache_capacity

        self.stats = PolicyStats()

    # --- sketch helpers ---

    def _row_indices(self, bh: BlockHash) -> list[int]:
        base = hash(bh)
        return [
            (base ^ seed) % self._sketch_width
            for seed in self._sketch_seeds
        ]

    def _sketch_increment(self, bh: BlockHash) -> None:
        for row, col in enumerate(self._row_indices(bh)):
            self._sketch[row][col] += 1
        self._sketch_sum += 1
        if self._sketch_sum >= self._age_threshold:
            # Halve all counters — keeps policy responsive.
            for row in range(self._sketch_depth):
                self._sketch[row] = [c >> 1 for c in self._sketch[row]]
            self._sketch_sum >>= 1

    def _sketch_estimate(self, bh: BlockHash) -> int:
        return min(
            self._sketch[row][col]
            for row, col in enumerate(self._row_indices(bh))
        )

    # --- storage helpers ---

    def _find(self, bh: BlockHash) -> tuple[OrderedDict[BlockHash, BlockStatus], BlockStatus] | None:
        for storage in (self._window, self._probation, self._protected):
            if bh in storage:
                return storage, storage[bh]
        return None

    # --- CachePolicy interface ---

    def get(self, bh: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        found = self._find(bh)
        return found[1] if found else None

    def insert(self, bh: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self._sketch_increment(bh)
        # New blocks always enter the window.  We let window exceed its
        # nominal size temporarily; the next evict() call trims.  When a
        # window block gets *touched* and is promoted to probation, that's
        # where the TinyLFU admission contest against probation's LRU
        # happens (see touch()).
        self._window[bh] = block

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        count = 0
        for bh in block_hashes:
            self._sketch_increment(bh)
            if bh in self._protected:
                self._protected.move_to_end(bh)
                count += 1
            elif bh in self._probation:
                block = self._probation.pop(bh)
                self._protected[bh] = block
                # If protected overflows, demote its LRU to probation tail.
                # One demote per touch — no unbounded loop.
                if len(self._protected) > self._protected_cap:
                    d_bh, d_block = self._protected.popitem(last=False)
                    self._probation[d_bh] = d_block
                count += 1
            elif bh in self._window:
                # Window block got a hit: promote to probation via TinyLFU
                # admission contest against probation's current LRU.
                block = self._window.pop(bh)
                self._maybe_admit_to_probation(bh, block)
                count += 1
        self.stats.touch_blocks += count

    def _maybe_admit_to_probation(self, bh: BlockHash, block: BlockStatus) -> None:
        """Admit a block to probation; if full, contest against probation's LRU."""
        if len(self._probation) < self._probation_cap:
            self._probation[bh] = block
            return
        # Probation full: TinyLFU contest against probation's LRU.
        victim_bh = next(iter(self._probation))
        if self._sketch_estimate(bh) > self._sketch_estimate(victim_bh):
            # Winner: swap.  Victim goes back to window (where it'll age
            # and eventually be picked by evict()).
            victim_block = self._probation.pop(victim_bh)
            self._probation[bh] = block
            self._window[victim_bh] = victim_block
        else:
            # Loser: stays in window (as MRU since we just touched it).
            self._window[bh] = block

    def remove(self, bh: BlockHash) -> None:
        self.stats.remove_calls += 1
        for storage in (self._window, self._probation, self._protected):
            if bh in storage:
                del storage[bh]
                return

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        # Pre-scan for atomicity.
        scan_steps = 0
        evictable = 0
        for storage in (self._probation, self._window, self._protected):
            for bh, block in storage.items():
                scan_steps += 1
                if evictable >= n:
                    break
                if bh in protected or block.ref_cnt != 0:
                    continue
                evictable += 1
            if evictable >= n:
                break

        if evictable < n:
            self.stats.evict_scan_steps += scan_steps
            self.stats.evict_failed += 1
            return None

        evicted: list[tuple[BlockHash, BlockStatus]] = []
        # Commit: evict probation → window → protected (coldest segment first).
        for storage in (self._probation, self._window, self._protected):
            if len(evicted) >= n:
                break
            to_remove: list[BlockHash] = []
            for bh, block in storage.items():
                if len(evicted) >= n:
                    break
                if bh in protected or block.ref_cnt != 0:
                    continue
                to_remove.append(bh)
                evicted.append((bh, block))
            for bh in to_remove:
                del storage[bh]

        self.stats.evict_scan_steps += scan_steps
        self.stats.evict_blocks += len(evicted)
        return evicted
