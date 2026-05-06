# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SIEVE (Zhang et al., NSDI 2024): DLL with a "hand" pointer; one-bit
visited flag is cleared on scan, unvisited blocks are evicted."""

from __future__ import annotations

from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import (
    BlockStatus,
    CachePolicy,
    PolicyStats,
)


class _SieveNode:
    __slots__ = ("block_hash", "block", "visited", "prev", "next")

    def __init__(self, block_hash: BlockHash, block: BlockStatus):
        self.block_hash = block_hash
        self.block = block
        self.visited: bool = False
        self.prev: _SieveNode | None = None
        self.next: _SieveNode | None = None


class SIEVECachePolicy(CachePolicy):
    def __init__(self, cache_capacity: int) -> None:
        self._map: dict[BlockHash, _SieveNode] = {}
        self._head: _SieveNode | None = None
        self._tail: _SieveNode | None = None
        self._hand: _SieveNode | None = None
        self.stats = PolicyStats()

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        node = self._map.get(block_hash)
        return node.block if node is not None else None

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        # Drop any stale node first: leaving an orphan in the list causes a
        # KeyError on later eviction.
        if block_hash in self._map:
            self.remove(block_hash)
        node = _SieveNode(block_hash, block)
        self._map[block_hash] = node
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node
        if self._hand is None:
            self._hand = node

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        node = self._map.pop(block_hash, None)
        if node is not None:
            self._unlink(node)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        count = 0
        for block_hash in block_hashes:
            node = self._map.get(block_hash)
            if node is not None:
                node.visited = True
                count += 1
        self.stats.touch_blocks += count

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        self.stats.cache_size_at_last_evict = len(self._map)

        if n == 0:
            return []

        hand = self._hand
        if hand is None:
            return None

        candidates: list[_SieveNode] = []
        candidate_ids: set[int] = set()
        # Cleared visited bits to roll back if the scan fails.
        cleared: list[_SieveNode] = []

        # Two full passes are enough: one to clear visited bits, one to
        # collect the now-unvisited entries.
        max_steps = 2 * len(self._map)
        scan_steps = 0

        for _ in range(max_steps):
            scan_steps += 1
            if len(candidates) >= n:
                break

            if hand.visited:
                hand.visited = False
                cleared.append(hand)
                hand = self._advance(hand)
            elif (
                hand.block.ref_cnt == 0
                and hand.block_hash not in protected
                and id(hand) not in candidate_ids
            ):
                candidates.append(hand)
                candidate_ids.add(id(hand))
                hand = self._advance(hand)
            else:
                hand = self._advance(hand)

        self.stats.evict_scan_steps += scan_steps

        if len(candidates) < n:
            self.stats.evict_failed += 1
            for node in cleared:
                if id(node) not in candidate_ids:
                    node.visited = True
            return None

        # The hand was advanced past every candidate during the scan, so we
        # must not touch self._hand inside the unlink loop.
        self._hand = hand
        result: list[tuple[BlockHash, BlockStatus]] = []
        for node in candidates:
            # pop tolerates duplicate-node corner cases.
            self._map.pop(node.block_hash, None)
            self._unlink_no_hand(node)
            result.append((node.block_hash, node.block))

        if not self._map:
            self._hand = None

        self.stats.evict_blocks += len(result)
        return result

    def _advance(self, node: _SieveNode) -> _SieveNode:
        if node.prev is not None:
            return node.prev
        assert self._tail is not None
        return self._tail

    def _unlink(self, node: _SieveNode) -> None:
        if self._hand is node:
            if node.prev is not None:
                self._hand = node.prev
            elif node.next is not None:
                self._hand = node.next
            else:
                self._hand = None

        self._unlink_no_hand(node)

    def _unlink_no_hand(self, node: _SieveNode) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next

        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev

        node.prev = node.next = None
