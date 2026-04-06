# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import (
    BlockStatus,
    CachePolicy,
    PolicyStats,
)


class LRUCachePolicy(CachePolicy):
    """LRU cache policy backed by a single OrderedDict."""

    def __init__(self, cache_capacity: int):
        # cache_capacity unused by LRU but accepted for a uniform constructor
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.stats = PolicyStats()

    def get(self, block_hash: BlockHash) -> BlockStatus | None:
        self.stats.get_calls += 1
        return self.blocks.get(block_hash)

    def insert(self, block_hash: BlockHash, block: BlockStatus) -> None:
        self.stats.insert_calls += 1
        self.blocks[block_hash] = block

    def remove(self, block_hash: BlockHash) -> None:
        self.stats.remove_calls += 1
        del self.blocks[block_hash]

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.stats.touch_calls += 1
        count = 0
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.blocks:
                self.blocks.move_to_end(block_hash)
                count += 1
        self.stats.touch_blocks += count

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> list[tuple[BlockHash, BlockStatus]] | None:
        self.stats.evict_calls += 1
        self.stats.cache_size_at_last_evict = len(self.blocks)

        if n == 0:
            return []
        candidates: list[tuple[BlockHash, BlockStatus]] = []
        scan_steps = 0
        for block_hash, block in self.blocks.items():
            scan_steps += 1
            if block.ref_cnt == 0 and block_hash not in protected:
                candidates.append((block_hash, block))
                if len(candidates) == n:
                    break
        self.stats.evict_scan_steps += scan_steps

        if len(candidates) < n:
            self.stats.evict_failed += 1
            return None
        for block_hash, _ in candidates:
            del self.blocks[block_hash]
        self.stats.evict_blocks += len(candidates)
        return candidates
