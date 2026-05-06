# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU-K (O'Neil et al., SIGMOD 1993): evict by oldest Kth-most-recent
access. Blocks accessed fewer than K times are evicted first."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import (
    BlockStatus,
    CachePolicy,
    PolicyStats,
)


class LRUKCachePolicy(CachePolicy):
    """LRU-K: evict by oldest Kth-most-recent access timestamp."""

    def __init__(self, cache_capacity: int, k: int = 2) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._capacity = cache_capacity
        self._k = k
        self._blocks: dict[BlockHash, BlockStatus] = {}
        # Ring buffer of last K access timestamps per block (oldest at left).
        self._history: dict[BlockHash, deque[int]] = {}
        self._counter = 0
        self.stats = PolicyStats()

    def _record_access(self, bh: BlockHash) -> None:
        self._counter += 1
        hist = self._history.get(bh)
        if hist is None:
            hist = deque(maxlen=self._k)
            self._history[bh] = hist
        hist.append(self._counter)

    def get(self, bh: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self._blocks.get(bh)

    def insert(self, bh: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self._blocks[bh] = block
        self._record_access(bh)

    def remove(self, bh: BlockHash) -> None:
        self.stats.remove_calls += 1
        self._blocks.pop(bh, None)
        self._history.pop(bh, None)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        count = 0
        for bh in block_hashes:
            if bh in self._blocks:
                self._record_access(bh)
                count += 1
        self.stats.touch_blocks += count

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        if n == 0:
            return []

        # Tier 0 (<K accesses, evicted first) vs tier 1 (Kth-oldest first).
        def sort_key(bh: BlockHash) -> tuple[int, int]:
            hist = self._history.get(bh)
            if hist is None or len(hist) < self._k:
                return (0, hist[-1] if hist else 0)
            return (1, hist[0])

        candidates = [
            bh for bh, block in self._blocks.items()
            if bh not in protected and block.ref_cnt == 0
        ]
        candidates.sort(key=sort_key)

        if len(candidates) < n:
            self.stats.evict_scan_steps += len(self._blocks)
            self.stats.evict_failed += 1
            return None

        evicted: list[tuple[BlockHash, BlockStatus]] = []
        for bh in candidates[:n]:
            block = self._blocks.pop(bh)
            self._history.pop(bh, None)
            evicted.append((bh, block))

        self.stats.evict_scan_steps += len(self._blocks) + len(evicted)
        self.stats.evict_blocks += len(evicted)
        return evicted
