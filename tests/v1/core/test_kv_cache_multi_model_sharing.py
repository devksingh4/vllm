# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Viability tests for multi-model KV sharing policies on one BlockPool.

These are intentionally small, deterministic checks for:

1. **Two-level (quota)**: hard cap on per-partition ref totals.
2. **Cost-aware (single-level)**: victim choice differs from strict LRU when
   normalized staleness favors evicting a cheaper partition's block.
"""

from tests.v1.core.test_prefix_caching import make_kv_cache_config
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager


def test_two_level_partition_ref_cap_allows_within_budget():
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=False,
        hash_block_size=16,
        partition_ref_caps={"model_a": 3},
    )
    pool.get_new_blocks(3, "model_a")
    assert pool.get_partition_block_ref_totals() == {"model_a": 3}


def test_two_level_partition_ref_cap_soft_over_budget():
    """Exceeding the cap at the block-pool level is a soft warning, not a crash.

    Prefix-cache hits can push a partition over cap without going through
    allocation, so get_new_blocks must not raise.  The hard gate lives in
    KVCacheManager.can_allocate / allocate_slots instead.
    """
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=False,
        hash_block_size=16,
        partition_ref_caps={"model_a": 2},
    )
    pool.get_new_blocks(2, "model_a")
    assert pool.would_exceed_partition_cap(1, "model_a")
    # get_new_blocks still succeeds (soft cap) with a warning
    extra = pool.get_new_blocks(1, "model_a")
    assert len(extra) == 1
    assert pool.get_partition_block_ref_totals()["model_a"] == 3


def test_two_level_dynamic_cap_adjustment():
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=False,
        hash_block_size=16,
        partition_ref_caps={"model_a": 2},
    )
    pool.get_new_blocks(2, "model_a")
    assert pool.would_exceed_partition_cap(1, "model_a")
    pool.set_partition_ref_caps({"model_a": 4})
    assert not pool.would_exceed_partition_cap(1, "model_a")
    pool.get_new_blocks(1, "model_a")
    assert pool.get_partition_block_ref_totals()["model_a"] == 3


def test_cost_aware_prefers_evicting_higher_normalized_staleness():
    """Not strict LRU: among scanned free blocks, pick argmax idle/cost.

    We synthesize access sequences so the LRU head is an *expensive* block that
    is still more "valuable" (lower idle/cost) than a *cheaper* block deeper in
    the free list, which should be evicted first.
    """
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=False,
        hash_block_size=16,
        partition_eviction_cost={"expensive": 100.0, "cheap": 1.0},
    )
    b_exp = pool.get_new_blocks(1, "expensive")[0]
    b_cheap = pool.get_new_blocks(1, "cheap")[0]
    b_fill = pool.get_new_blocks(1, "expensive")[0]
    # Free in this order so the LRU head is expensive, then cheap (not an
    # untouched block that would otherwise sit at the head).
    pool.free_blocks([b_exp], "expensive")
    pool.free_blocks([b_cheap], "cheap")
    pool.free_blocks([b_fill], "expensive")
    head = pool.free_block_queue.fake_free_list_head.next_free_block
    second = head.next_free_block
    assert head is b_exp
    assert second is b_cheap
    # Make expensive head look freshly touched, cheap second look very stale.
    pool._kv_logical_time = 1000
    head.kv_access_seq = 990
    second.kv_access_seq = 1
    victim = pool.free_block_queue.popleft()
    assert victim is second
    assert victim is b_cheap


def test_kv_cache_manager_wires_partition_ref_caps():
    """End-to-end: KVCacheManager → coordinator → BlockPool two-level caps."""
    manager = KVCacheManager(
        make_kv_cache_config(16, 8),
        max_model_len=8192,
        enable_caching=False,
        hash_block_size=16,
        partition_ref_caps={"P": 2},
    )
    manager.block_pool.get_new_blocks(2, "P")
    assert manager.block_pool.would_exceed_partition_cap(1, "P")
    # Block-pool level is a soft cap — allocation proceeds with warning
    extra = manager.block_pool.get_new_blocks(1, "P")
    assert len(extra) == 1


def test_kv_cache_manager_set_partition_ref_caps_runtime():
    manager = KVCacheManager(
        make_kv_cache_config(16, 8),
        max_model_len=8192,
        enable_caching=False,
        hash_block_size=16,
        partition_ref_caps={"P": 1},
    )
    manager.block_pool.get_new_blocks(1, "P")
    assert manager.block_pool.would_exceed_partition_cap(1, "P")
    manager.set_partition_ref_caps({"P": 4})
    assert not manager.block_pool.would_exceed_partition_cap(1, "P")
    manager.block_pool.get_new_blocks(1, "P")
    assert manager.block_pool.get_partition_block_ref_totals()["P"] == 2


def test_kv_cache_manager_wires_cost_aware_eviction():
    manager = KVCacheManager(
        make_kv_cache_config(16, 4),
        max_model_len=8192,
        enable_caching=False,
        hash_block_size=16,
        partition_eviction_cost={"expensive": 100.0, "cheap": 1.0},
    )
    pool = manager.block_pool
    b_exp = pool.get_new_blocks(1, "expensive")[0]
    b_cheap = pool.get_new_blocks(1, "cheap")[0]
    b_fill = pool.get_new_blocks(1, "expensive")[0]
    pool.free_blocks([b_exp], "expensive")
    pool.free_blocks([b_cheap], "cheap")
    pool.free_blocks([b_fill], "expensive")
    head = pool.free_block_queue.fake_free_list_head.next_free_block
    second = head.next_free_block
    assert head is b_exp and second is b_cheap
    pool._kv_logical_time = 1000
    head.kv_access_seq = 990
    second.kv_access_seq = 1
    assert pool.free_block_queue.popleft() is b_cheap
