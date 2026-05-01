# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def test_block_pool_partition_ref_totals():
    pool = BlockPool(num_gpu_blocks=16, enable_caching=False, hash_block_size=16)
    a = pool.get_new_blocks(2, "partition-a")
    b = pool.get_new_blocks(1, "partition-b")
    assert pool.get_partition_block_ref_totals() == {"partition-a": 2, "partition-b": 1}
    pool.free_blocks(reversed(a), "partition-a")
    pool.free_blocks(reversed(b), "partition-b")
    assert pool.get_partition_block_ref_totals() == {}


def test_partition_metrics_skip_free_without_tracked_alloc():
    """Frees without a prior get_new_blocks/touch ref are ignored for partition totals."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=False, hash_block_size=16)
    orphan = KVCacheBlock(3)
    orphan.ref_cnt = 1
    pool.free_blocks([orphan], "never-allocated")
    assert pool.get_partition_block_ref_totals() == {}
