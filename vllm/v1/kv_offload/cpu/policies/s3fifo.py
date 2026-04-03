# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
S3-FIFO cache eviction policy.

Reference: Yang et al., "FIFO Queues are All You Need for Cache Eviction"
(SOSP 2023).
"""

from __future__ import annotations

from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import (
    BlockStatus,
    CachePolicy,
    PolicyStats,
)

_MAX_FREQ: int = 3


class S3FIFOCachePolicy(CachePolicy):
    """
    S3-FIFO cache eviction policy using Python dicts for native C-level FIFO ordering,
    eliminating DLL overhead and double-scans.
    """

    def __init__(self, cache_capacity: int) -> None:
        self._capacity = cache_capacity
        self._s_capacity = max(1, cache_capacity // 10)
        self._m_capacity = cache_capacity - self._s_capacity
        self._ghost_capacity = cache_capacity

        # S queue (insertion order = oldest to newest). Freq is implicitly 0.
        self._s_map: dict[BlockHash, BlockStatus] = {}

        # M queue (insertion order = oldest to newest).
        self._m_map: dict[BlockHash, BlockStatus] = {}
        self._m_freq: dict[BlockHash, int] = {}

        # Ghost set (tracks recently evicted S-queue items).
        self._ghost: dict[BlockHash, None] = {}

        self.stats = PolicyStats()

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self._s_map.get(block_hash) or self._m_map.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        if block_hash in self._ghost:
            self._ghost.pop(block_hash)
            self._m_map[block_hash] = block
            self._m_freq[block_hash] = 0
        else:
            self._s_map[block_hash] = block

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        self._s_map.pop(block_hash, None)
        if block_hash in self._m_map:
            self._m_map.pop(block_hash)
            self._m_freq.pop(block_hash)
        self._ghost.pop(block_hash, None)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        count = 0
        for bh in block_hashes:
            if bh in self._m_map:
                if self._m_freq[bh] < _MAX_FREQ:
                    self._m_freq[bh] += 1
                    count += 1
            elif bh in self._s_map:
                block = self._s_map.pop(bh)
                self._m_map[bh] = block
                self._m_freq[bh] = 1
                count += 1
        self.stats.touch_blocks += count

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        self.stats.cache_size_at_last_evict = len(self._s_map) + len(self._m_map)

        if n == 0:
            return []

        # 1. Pre-scan to guarantee atomicity (must find at least 'n' evictable blocks globally)
        scan_steps = 0
        evictable_count = 0

        for bh, block in self._s_map.items():
            if evictable_count == n:
                break
            scan_steps += 1
            if block.ref_cnt == 0 and bh not in protected:
                evictable_count += 1

        if evictable_count < n:
            for bh, block in self._m_map.items():
                if evictable_count == n:
                    break
                scan_steps += 1
                if block.ref_cnt == 0 and bh not in protected:
                    evictable_count += 1

        if evictable_count < n:
            self.stats.evict_scan_steps += scan_steps
            self.stats.evict_failed += 1
            return None

        # 2. S3-FIFO decay loop
        result: list[tuple[BlockHash, BlockStatus]] = []

        # Phase A: Evict from S queue
        s_keys = list(self._s_map.keys())
        for bh in s_keys:
            if len(result) == n:
                break
            scan_steps += 1
            block = self._s_map[bh]
            if block.ref_cnt == 0 and bh not in protected:
                del self._s_map[bh]
                self._ghost[bh] = None
                result.append((bh, block))

        # Phase B: Evict from M queue with multi-pass decay
        while len(result) < n:
            m_keys = list(self._m_map.keys())
            for bh in m_keys:
                if len(result) == n:
                    break
                scan_steps += 1

                block = self._m_map.get(bh)
                if block is None:
                    continue  # Safely skip if already evicted in a previous pass

                if block.ref_cnt == 0 and bh not in protected:
                    if self._m_freq[bh] == 0:
                        del self._m_map[bh]
                        del self._m_freq[bh]
                        result.append((bh, block))
                    else:
                        # Decay and recycle (move to tail of dict)
                        self._m_freq[bh] -= 1
                        self._m_map[bh] = self._m_map.pop(bh)

        # 3. Trim ghost to capacity
        overflow = len(self._ghost) - self._ghost_capacity
        if overflow > 0:
            ghost_iter = iter(self._ghost)
            to_remove = [next(ghost_iter) for _ in range(overflow)]
            for bh in to_remove:
                del self._ghost[bh]

        self.stats.evict_scan_steps += scan_steps
        self.stats.evict_blocks += len(result)
        return result
