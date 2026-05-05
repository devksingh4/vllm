#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA-aware cache policy benchmark (pure-Python simulation, no GPU required).

Compares baseline policies (LRU, SIEVE, S3-FIFO) against LoRA-coupled
variants (tight and loose coupling) under four workload scenarios:

  Scenario A — Adapter Locality:
    Requests arrive in bursts per adapter.  Between bursts, other adapters
    fill the cache.  Tests whether coupling preserves adapter block groups.

  Scenario B — Adapter Thrashing:
    More adapters than GPU slots, cycled round-robin.  Tests whether
    coupling avoids caching blocks whose adapters are gone.

  Scenario C — Mixed Popularity:
    Some adapters are hot (frequent), others are cold (rare).  Tests the
    interaction of adapter popularity and block-level recency.

  Scenario D — Sticky + Ephemeral:
    A few sticky adapters that keep returning, interleaved with many
    one-shot ephemeral adapters.  Tests whether coupling preserves the
    sticky set while evicting ephemeral dead weight.

Metrics:
  - CPU cache hit rate  (higher is better)
  - Adapter reload count  (lower is better)
  - Composite cost = cache_misses * miss_cost + adapter_reloads * reload_cost

Usage:
  python benchmarks/benchmark_lora_policy.py
  python benchmarks/benchmark_lora_policy.py --scenario adapter_locality
  python benchmarks/benchmark_lora_policy.py --cache-capacity 200 --gpu-lora-slots 3
"""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.cpu.policies.abstract import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lora_aware import (
    LoRAAdaptiveBudgetPolicy,
    LoRABudgetPolicy,
    LoRACorrelatedTouchPolicy,
    LoRACostAwarePolicy,
    LoRAFrequencyWeightedPolicy,
    LoRAGhostListPolicy,
    LoRAHysteresisCouplingPolicy,
    LoRALooseCouplingPolicy,
    LoRAPositionAwarePolicy,
    LoRAPrefixTreePolicy,
    LoRASoftBoostCouplingPolicy,
    LoRATightCouplingPolicy,
    LoRATwoLevelLRUPolicy,
)
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.policies.lru_k import LRUKCachePolicy
from vllm.v1.kv_offload.cpu.policies.s3fifo import S3FIFOCachePolicy
from vllm.v1.kv_offload.cpu.policies.sieve import SIEVECachePolicy
from vllm.v1.kv_offload.cpu.policies.tinylfu import TinyLFUCachePolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block_hash(adapter_id: str, block_idx: int) -> BlockHash:
    """Deterministic block hash from adapter + index."""
    raw = hashlib.sha256(f"{adapter_id}:{block_idx}".encode()).digest()
    return BlockHash(raw)


@dataclass
class Request:
    adapter_id: str
    block_hashes: list[BlockHash]


@dataclass
class SimResult:
    policy_name: str
    scenario: str
    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    adapter_reloads: int = 0
    evict_failures: int = 0

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_lookups if self.total_lookups else 0.0

    def composite_cost(self, miss_cost: float = 10.0, reload_cost: float = 5.0) -> float:
        return self.cache_misses * miss_cost + self.adapter_reloads * reload_cost


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

def gen_adapter_locality(
    num_adapters: int = 6,
    blocks_per_request: int = 20,
    burst_size: int = 5,
    num_rounds: int = 4,
    rng: random.Random | None = None,
) -> list[Request]:
    """Scenario A: bursty per-adapter requests."""
    rng = rng or random.Random(42)
    requests: list[Request] = []
    adapters = [f"adapter_{i}" for i in range(num_adapters)]
    for _ in range(num_rounds):
        rng.shuffle(adapters)
        for adapter_id in adapters:
            for _ in range(burst_size):
                bhs = [_make_block_hash(adapter_id, j) for j in range(blocks_per_request)]
                requests.append(Request(adapter_id=adapter_id, block_hashes=bhs))
    return requests


def gen_adapter_thrashing(
    num_adapters: int = 10,
    blocks_per_request: int = 20,
    requests_per_adapter: int = 3,
    num_cycles: int = 3,
    rng: random.Random | None = None,
) -> list[Request]:
    """Scenario B: round-robin cycling through many adapters."""
    rng = rng or random.Random(43)
    requests: list[Request] = []
    adapters = [f"adapter_{i}" for i in range(num_adapters)]
    for _ in range(num_cycles):
        for adapter_id in adapters:
            for _ in range(requests_per_adapter):
                bhs = [_make_block_hash(adapter_id, j) for j in range(blocks_per_request)]
                requests.append(Request(adapter_id=adapter_id, block_hashes=bhs))
    return requests


def gen_mixed_popularity(
    num_hot: int = 2,
    num_cold: int = 6,
    blocks_per_request: int = 20,
    total_requests: int = 200,
    hot_probability: float = 0.7,
    rng: random.Random | None = None,
) -> list[Request]:
    """Scenario C: hot/cold adapter mix."""
    rng = rng or random.Random(44)
    hot_adapters = [f"hot_{i}" for i in range(num_hot)]
    cold_adapters = [f"cold_{i}" for i in range(num_cold)]
    requests: list[Request] = []
    for _ in range(total_requests):
        if rng.random() < hot_probability:
            adapter_id = rng.choice(hot_adapters)
        else:
            adapter_id = rng.choice(cold_adapters)
        bhs = [_make_block_hash(adapter_id, j) for j in range(blocks_per_request)]
        requests.append(Request(adapter_id=adapter_id, block_hashes=bhs))
    return requests


def gen_sparse_partial_access(
    num_adapters: int = 4,
    adapter_pool_size: int = 50,
    blocks_per_request: int = 10,
    bursts_per_adapter: int = 4,
    requests_per_burst: int = 4,
    rng: random.Random | None = None,
) -> list[Request]:
    """Scenario E: each adapter has a large pool, each request accesses a
    random subset.

    This reflects realistic chat workloads where an adapter serves many
    distinct conversations that share an underlying prompt template but
    diverge in their specific blocks.  Block-level scan resistance cannot
    see that blocks from the same adapter are related; adapter-level
    coupling (soft-boost) can.
    """
    rng = rng or random.Random(46)
    adapters = [f"adapter_{i}" for i in range(num_adapters)]
    pools = {
        aid: [_make_block_hash(aid, j) for j in range(adapter_pool_size)]
        for aid in adapters
    }
    requests: list[Request] = []
    for _ in range(bursts_per_adapter):
        rng.shuffle(adapters)
        for adapter_id in adapters:
            for _ in range(requests_per_burst):
                bhs = rng.sample(pools[adapter_id], blocks_per_request)
                requests.append(Request(adapter_id=adapter_id, block_hashes=bhs))
    return requests


def gen_sticky_ephemeral(
    num_sticky: int = 3,
    num_ephemeral: int = 8,
    blocks_per_request: int = 20,
    sticky_requests_each: int = 10,
    ephemeral_requests_each: int = 2,
    rng: random.Random | None = None,
) -> list[Request]:
    """Scenario D: sticky adapters interleaved with one-shot ephemeral adapters.

    Sticky adapters keep returning — their blocks should be preserved.
    Ephemeral adapters appear once with a handful of requests, then vanish
    forever — their blocks become dead weight until evicted.
    """
    rng = rng or random.Random(45)
    sticky = [f"sticky_{i}" for i in range(num_sticky)]
    ephemeral = [f"eph_{i}" for i in range(num_ephemeral)]

    requests: list[Request] = []
    eph_iter = iter(ephemeral)
    for sticky_round in range(sticky_requests_each):
        for adapter_id in sticky:
            bhs = [_make_block_hash(adapter_id, j) for j in range(blocks_per_request)]
            requests.append(Request(adapter_id=adapter_id, block_hashes=bhs))
        # Sprinkle an ephemeral burst after each sticky round.
        try:
            eph_id = next(eph_iter)
            for _ in range(ephemeral_requests_each):
                bhs = [_make_block_hash(eph_id, j) for j in range(blocks_per_request)]
                requests.append(Request(adapter_id=eph_id, block_hashes=bhs))
        except StopIteration:
            pass
    rng.shuffle(requests)  # mix them up a bit
    return requests


_SCENARIOS = {
    "adapter_locality": gen_adapter_locality,
    "adapter_thrashing": gen_adapter_thrashing,
    "mixed_popularity": gen_mixed_popularity,
    "sticky_ephemeral": gen_sticky_ephemeral,
    "sparse_partial_access": gen_sparse_partial_access,
}

# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _make_policy(
    name: str, capacity: int
) -> tuple[CachePolicy, str]:
    """Construct a policy by name. Supports 'lora_tight:lru' etc."""
    base_map: dict[str, type[CachePolicy]] = {
        "lru": LRUCachePolicy,
        "sieve": SIEVECachePolicy,
        "s3fifo": S3FIFOCachePolicy,
        "tinylfu": TinyLFUCachePolicy,
        "lru_k": LRUKCachePolicy,
        "lora_twolevel": LoRATwoLevelLRUPolicy,
    }
    coupling_map: dict[str, type] = {
        "lora_tight": LoRATightCouplingPolicy,
        "lora_loose": LoRALooseCouplingPolicy,
        "lora_hysteresis": LoRAHysteresisCouplingPolicy,
        "lora_soft": LoRASoftBoostCouplingPolicy,
        "lora_freqweighted": LoRAFrequencyWeightedPolicy,
        "lora_correlated": LoRACorrelatedTouchPolicy,
        "lora_budget": LoRABudgetPolicy,
        "lora_adabudget": LoRAAdaptiveBudgetPolicy,
        "lora_costaware": LoRACostAwarePolicy,
        "lora_ghost": LoRAGhostListPolicy,
        "lora_position": LoRAPositionAwarePolicy,
        "lora_prefixtree": LoRAPrefixTreePolicy,
    }
    if ":" in name:
        coupling_name, base_name = name.split(":", 1)
        return coupling_map[coupling_name](
            cache_capacity=capacity, inner_cls=base_map[base_name]
        ), name
    return base_map[name](cache_capacity=capacity), name


def simulate(
    policy_name: str,
    scenario_name: str,
    requests: list[Request],
    cache_capacity: int,
    gpu_lora_slots: int,
) -> SimResult:
    """Run one simulation: feed requests through the policy and collect stats."""
    policy, _ = _make_policy(policy_name, cache_capacity)
    is_lora_aware = hasattr(policy, "register_block_adapter")
    has_position = hasattr(policy, "register_block_position")
    has_parent = hasattr(policy, "register_block_parent")

    result = SimResult(policy_name=policy_name, scenario=scenario_name)

    # Track which adapters are "on GPU" (LRU with gpu_lora_slots capacity).
    gpu_adapters: list[str] = []  # ordered by recency (most recent last)
    # Track current cache size externally since CachePolicy has no size() method.
    cache_size = 0
    block_id_counter = 0

    for req in requests:
        adapter_id = req.adapter_id

        # --- GPU adapter slot management ---
        if adapter_id in gpu_adapters:
            gpu_adapters.remove(adapter_id)
            gpu_adapters.append(adapter_id)
        else:
            result.adapter_reloads += 1
            if len(gpu_adapters) >= gpu_lora_slots:
                gpu_adapters.pop(0)
            gpu_adapters.append(adapter_id)

        if is_lora_aware:
            policy.update_live_adapters(set(gpu_adapters))

        # --- CPU cache lookup (classify each block as hit or miss) ---
        hit_hashes: list[BlockHash] = []
        miss_hashes: list[BlockHash] = []
        for bh in req.block_hashes:
            result.total_lookups += 1
            block = policy.get(bh)
            if block is not None and block.ref_cnt >= 0:
                result.cache_hits += 1
                hit_hashes.append(bh)
            else:
                result.cache_misses += 1
                miss_hashes.append(bh)

        # Touch hits so the policy marks them recently used.
        if hit_hashes:
            policy.touch(hit_hashes)

        # Insert misses, evicting first if the cache would overflow.
        if miss_hashes:
            overflow = cache_size + len(miss_hashes) - cache_capacity
            if overflow > 0:
                protected = set(req.block_hashes)
                evicted = policy.evict(overflow, protected)
                if evicted is None:
                    result.evict_failures += 1
                else:
                    cache_size -= len(evicted)

            for bh in miss_hashes:
                if policy.get(bh) is not None:
                    continue
                if cache_size >= cache_capacity:
                    break  # still no room
                block = BlockStatus(block_id_counter)
                block.ref_cnt = 0  # mark as ready
                block_id_counter += 1
                policy.insert(bh, block)
                cache_size += 1
                if is_lora_aware:
                    policy.register_block_adapter(bh, adapter_id)
                if has_position:
                    policy.register_block_position(
                        bh, req.block_hashes.index(bh)
                    )
                if has_parent:
                    idx = req.block_hashes.index(bh)
                    parent = req.block_hashes[idx - 1] if idx > 0 else None
                    policy.register_block_parent(bh, parent)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA-aware cache policy simulation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        choices=list(_SCENARIOS) + ["all"],
        default="all",
        help="Workload scenario (default: all)",
    )
    p.add_argument(
        "--cache-capacity",
        type=int,
        default=150,
        help="CPU cache capacity in blocks (default: 150)",
    )
    p.add_argument(
        "--gpu-lora-slots",
        type=int,
        default=3,
        help="Number of GPU LoRA adapter slots (default: 3)",
    )
    p.add_argument(
        "--miss-cost",
        type=float,
        default=10.0,
        help="Cost multiplier for a cache miss (default: 10.0)",
    )
    p.add_argument(
        "--reload-cost",
        type=float,
        default=5.0,
        help="Cost multiplier for an adapter reload (default: 5.0)",
    )
    return p.parse_args()


_ALL_POLICIES = [
    # --- baselines ---
    "lru",
    "sieve",
    "s3fifo",
    # --- new classical baselines ---
    "tinylfu",
    "lru_k",
    # --- original coupling strategies ---
    "lora_tight:s3fifo",
    "lora_tight:sieve",
    "lora_loose:s3fifo",
    "lora_loose:sieve",
    "lora_hysteresis:s3fifo",
    "lora_hysteresis:sieve",
    "lora_soft:s3fifo",
    "lora_soft:sieve",
    # --- refined continuous-signal strategies ---
    "lora_freqweighted:lru",
    "lora_freqweighted:s3fifo",
    "lora_freqweighted:sieve",
    "lora_correlated:lru",
    "lora_correlated:sieve",
    "lora_correlated:s3fifo",
    "lora_budget:lru",
    "lora_budget:s3fifo",
    "lora_budget:sieve",
    "lora_budget:tinylfu",
    "lora_adabudget:lru",
    "lora_adabudget:s3fifo",
    "lora_adabudget:tinylfu",
    "lora_costaware:lru",
    "lora_costaware:s3fifo",
    "lora_costaware:sieve",
    "lora_ghost:lru",
    "lora_ghost:s3fifo",
    "lora_ghost:sieve",
    # --- structural (vllm-specific) ---
    "lora_position:lru",
    "lora_position:s3fifo",
    "lora_prefixtree:lru",
    "lora_prefixtree:s3fifo",
    # --- standalone policy ---
    "lora_twolevel",
]


def main() -> None:
    args = parse_args()
    scenarios = list(_SCENARIOS) if args.scenario == "all" else [args.scenario]

    for scenario_name in scenarios:
        print("=" * 72)
        print(f"Scenario: {scenario_name}")
        print(f"Cache capacity: {args.cache_capacity} blocks | "
              f"GPU LoRA slots: {args.gpu_lora_slots}")
        print("=" * 72)

        workload = _SCENARIOS[scenario_name]()
        num_unique_adapters = len({r.adapter_id for r in workload})
        num_unique_blocks = len({bh for r in workload for bh in r.block_hashes})
        print(f"Requests: {len(workload)} | "
              f"Unique adapters: {num_unique_adapters} | "
              f"Unique blocks: {num_unique_blocks}")
        print()

        results: list[SimResult] = []
        for policy_name in _ALL_POLICIES:
            res = simulate(
                policy_name=policy_name,
                scenario_name=scenario_name,
                requests=workload,
                cache_capacity=args.cache_capacity,
                gpu_lora_slots=args.gpu_lora_slots,
            )
            results.append(res)

        # Find winner(s) by hit rate and by cost.
        best_hit = max(res.hit_rate for res in results)
        best_cost = min(
            res.composite_cost(args.miss_cost, args.reload_cost)
            for res in results
        )

        header = (
            f"{'Policy':<24} {'Hit Rate':>10} {'Hits':>8} {'Misses':>8} "
            f"{'Reloads':>8} {'Cost':>10}  Marker"
        )
        print(header)
        print("-" * len(header))
        for res in results:
            cost = res.composite_cost(args.miss_cost, args.reload_cost)
            markers = []
            if abs(res.hit_rate - best_hit) < 1e-9:
                markers.append("best hit")
            if abs(cost - best_cost) < 1e-9:
                markers.append("best cost")
            marker_str = f" <-- {', '.join(markers)}" if markers else ""
            print(
                f"{res.policy_name:<24} {res.hit_rate:>9.1%} "
                f"{res.cache_hits:>8} {res.cache_misses:>8} "
                f"{res.adapter_reloads:>8} {cost:>10.0f}{marker_str}"
            )
        print()


if __name__ == "__main__":
    main()
