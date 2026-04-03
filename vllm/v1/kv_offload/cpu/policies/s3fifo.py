# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
S3-FIFO cache eviction policy.

Reference: Yang et al., "FIFO Queues are All You Need for Cache Eviction"
(SOSP 2023).
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

_MAX_FREQ: int = 3


class _FIFONode:
    """Doubly-linked list node shared by the S and M queues."""

    __slots__ = ("block_hash", "block", "freq", "prev", "next")

    def __init__(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.block_hash = block_hash
        self.block = block
        self.freq: int = 0
        self.prev: _FIFONode | None = None
        self.next: _FIFONode | None = None


class S3FIFOCachePolicy(CachePolicy):
    """
    S3-FIFO cache eviction policy using two FIFO queues and a ghost set.
    """

    def __init__(self, cache_capacity: int) -> None:
        self._capacity = cache_capacity
        # 10 % small / 90 % main split (paper recommendation).
        self._s_capacity = max(1, cache_capacity // 10)
        self._m_capacity = cache_capacity - self._s_capacity
        self._ghost_capacity = cache_capacity

        # S queue (head = oldest, tail = newest)
        self._s_map: dict[BlockHash, _FIFONode] = {}
        self._s_head: _FIFONode | None = None
        self._s_tail: _FIFONode | None = None

        # M queue
        self._m_map: dict[BlockHash, _FIFONode] = {}
        self._m_head: _FIFONode | None = None
        self._m_tail: _FIFONode | None = None

        # Ghost set (oldest entries evicted first when capacity exceeded)
        self._ghost: OrderedDict[BlockHash, None] = OrderedDict()

        self.stats = PolicyStats()

    # ------------------------------------------------------------------
    # CachePolicy interface
    # ------------------------------------------------------------------

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        node = self._s_map.get(block_hash) or self._m_map.get(block_hash)
        return node.block if node is not None else None

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        node = _FIFONode(block_hash, block)
        if block_hash in self._ghost:
            # Re-insertion after an S-eviction: skip S and go straight to M.
            self._ghost.pop(block_hash)
            self._m_map[block_hash] = node
            self._m_append(node)
        else:
            self._s_map[block_hash] = node
            self._s_append(node)

    def remove(self, block_hash: BlockHash) -> None:
        """Remove a block (used to clean up after a failed store)."""
        self.stats.remove_calls += 1
        if block_hash in self._s_map:
            node = self._s_map.pop(block_hash)
            self._s_unlink(node)
        elif block_hash in self._m_map:
            node = self._m_map.pop(block_hash)
            self._m_unlink(node)
        # Drop from ghost too so a future insert goes to S (not M).
        self._ghost.pop(block_hash, None)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        count = 0
        for block_hash in block_hashes:
            node = self._s_map.get(block_hash)
            if node is not None:
                if node.freq < _MAX_FREQ:
                    node.freq += 1
                    count += 1
                # Eager promotion: an S block that has been accessed at least
                # once is proven useful — move it to M immediately
                if node.freq > 0:
                    del self._s_map[block_hash]
                    self._s_unlink(node)
                    self._m_map[block_hash] = node
                    self._m_append(node)
            else:
                node = self._m_map.get(block_hash)
                if node is not None and node.freq < _MAX_FREQ:
                    node.freq += 1
                    count += 1
        self.stats.touch_blocks += count

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        self.stats.cache_size_at_last_evict = len(self._s_map) + len(self._m_map)

        if n == 0:
            return []

        # Pre-scan to guarantee atomicity
        scan_steps = 0
        evictable_count = 0

        node = self._s_head
        while node is not None and evictable_count < n:
            scan_steps += 1
            if node.block.ref_cnt == 0 and node.block_hash not in protected:
                evictable_count += 1
            node = node.next

        if evictable_count < n:
            node = self._m_head
            while node is not None and evictable_count < n:
                scan_steps += 1
                if node.block.ref_cnt == 0 and node.block_hash not in protected:
                    evictable_count += 1
                node = node.next

        if evictable_count < n:
            self.stats.evict_scan_steps += scan_steps
            self.stats.evict_failed += 1
            return None

        # S3-FIFO decay loop
        result: list[tuple[BlockHash, BlockStatus]] = []

        while len(result) < n:
            # Traverse S queue
            node = self._s_head
            while node is not None and len(result) < n:
                scan_steps += 1
                next_node = node.next

                if node.block.ref_cnt == 0 and node.block_hash not in protected:
                    if node.freq == 0:
                        # Evict from S
                        bh = node.block_hash
                        del self._s_map[bh]
                        self._s_unlink(node)
                        self._ghost[bh] = None
                        result.append((bh, node.block))
                    else:
                        # Promote to M
                        node.freq -= 1
                        bh = node.block_hash
                        del self._s_map[bh]
                        self._s_unlink(node)
                        self._m_map[bh] = node
                        self._m_append(node)
                node = next_node

            # Traverse M queue
            if len(result) < n:
                node = self._m_head
                while node is not None and len(result) < n:
                    scan_steps += 1
                    next_node = node.next

                    if node.block.ref_cnt == 0 and node.block_hash not in protected:
                        if node.freq == 0:
                            # Evict from M
                            bh = node.block_hash
                            del self._m_map[bh]
                            self._m_unlink(node)
                            result.append((bh, node.block))
                        else:
                            # Recycle in M
                            node.freq -= 1
                            self._m_unlink(node)
                            self._m_append(node)
                    node = next_node

        # Trim ghost to capacity (evict oldest entries first).
        while len(self._ghost) > self._ghost_capacity:
            self._ghost.popitem(last=False)

        self.stats.evict_scan_steps += scan_steps
        self.stats.evict_blocks += len(result)
        return result

    def _s_append(self, node: _FIFONode) -> None:
        """Append node to the tail (newest end) of S."""
        node.prev = node.next = None
        if self._s_tail is None:
            self._s_head = self._s_tail = node
        else:
            self._s_tail.next = node
            node.prev = self._s_tail
            self._s_tail = node

    def _s_unlink(self, node: _FIFONode) -> None:
        """Remove node from S, maintaining head/tail."""
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._s_head = node.next

        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._s_tail = node.prev

        node.prev = node.next = None

    def _m_append(self, node: _FIFONode) -> None:
        """Append node to the tail (newest end) of M."""
        node.prev = node.next = None
        if self._m_tail is None:
            self._m_head = self._m_tail = node
        else:
            self._m_tail.next = node
            node.prev = self._m_tail
            self._m_tail = node

    def _m_unlink(self, node: _FIFONode) -> None:
        """Remove node from M, maintaining head/tail."""
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._m_head = node.next

        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._m_tail = node.prev

        node.prev = node.next = None
