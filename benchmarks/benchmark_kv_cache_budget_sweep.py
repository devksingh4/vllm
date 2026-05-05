"""
Sweep eviction policies × cache budgets and collect TTFT / throughput / hit rate.

Produces a CSV table suitable for plotting policy comparison curves.
Calls benchmark_kv_cache.py as a subprocess for each (policy, budget) pair.

Usage (CPU, small scale):
    .venv/bin/python benchmarks/benchmark_kv_cache_budget_sweep.py \
        --policies lru sieve s3fifo \
        --budgets 2 4 8 \
        --workload scan-resistant \
        --num-batches 2 --batch-size 8 --prefix-words 400 \
        --max-tokens 10

Usage (GPU):
    .venv/bin/python benchmarks/benchmark_kv_cache_budget_sweep.py \
        --policies lru sieve s3fifo \
        --budgets 0.3 0.4 0.5 0.6 \
        --workload zipfian \
        --num-batches 4 --batch-size 32 --prefix-words 2000 \
        --max-tokens 50 --model Qwen/Qwen2.5-3B
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep eviction policies × cache budgets"
    )
    p.add_argument(
        "--policies",
        nargs="+",
        default=["lru", "sieve", "s3fifo"],
        help="Eviction policies to compare",
    )
    p.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[2, 4, 8],
        help="Cache budgets (GiB for CPU via VLLM_CPU_KVCACHE_SPACE, "
        "or gpu_memory_utilization fraction for GPU)",
    )
    # Workload params (forwarded to benchmark_kv_cache.py)
    p.add_argument("--workload", default="scan-resistant")
    p.add_argument("--num-batches", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prefix-words", type=int, default=400)
    p.add_argument("--suffix-words", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--num-prefixes", type=int, default=80)
    p.add_argument("--zipfian-alpha", type=float, default=1.5)
    p.add_argument("--working-set-size", type=int, default=15)
    p.add_argument("--model", type=str, default=None)
    p.add_argument(
        "--output-csv",
        type=str,
        default="budget_sweep_results.csv",
        help="Path for output CSV (default: budget_sweep_results.csv)",
    )
    return p.parse_args()


_TTFT_MEAN_RE = re.compile(r"Mean:\s+([\d.]+)\s+ms")
_THROUGHPUT_RE = re.compile(r"Total Token Throughput:\s+([\d.]+)")
_OUT_THROUGHPUT_RE = re.compile(r"Output Only Throughput:\s+([\d.]+)")
_HIT_RATE_RE = re.compile(r"Prefix cache hit rate:\s+([\d.]+)%")
_TOTAL_TIME_RE = re.compile(r"Total Time:\s+([\d.]+)")
_TTFT_P50_RE = re.compile(r"P50 \(median\):\s+([\d.]+)\s+ms")
_TTFT_P99_RE = re.compile(r"P99:\s+([\d.]+)\s+ms")


def _extract(pattern, text, default=""):
    m = pattern.search(text)
    return m.group(1) if m else default


def run_single(policy: str, budget: float, args) -> dict:
    """Run one (policy, budget) configuration and return parsed metrics."""
    import torch

    is_cpu = not torch.cuda.is_available()

    bench_script = str(
        Path(__file__).parent / "benchmark_kv_cache.py"
    )

    env = os.environ.copy()
    env["VLLM_KV_OFFLOAD_POLICY"] = policy

    cmd = [
        sys.executable,
        bench_script,
        "--workload", args.workload,
        "--num-batches", str(args.num_batches),
        "--batch-size", str(args.batch_size),
        "--prefix-words", str(args.prefix_words),
        "--suffix-words", str(args.suffix_words),
        "--max-tokens", str(args.max_tokens),
        "--num-prefixes", str(args.num_prefixes),
        "--zipfian-alpha", str(args.zipfian_alpha),
        "--working-set-size", str(args.working_set_size),
        "--enable-prefix-caching",
        "--enforce-eager",
    ]

    if is_cpu:
        env["VLLM_CPU_KVCACHE_SPACE"] = str(int(budget))
        cmd += ["--cpu-kv-cache-space", str(int(budget))]
    else:
        cmd += ["--gpu-memory-utilization", str(budget)]

    if args.model:
        cmd += ["--model", args.model]

    print(f"\n{'='*60}")
    print(f"  Policy={policy}  Budget={budget}  Workload={args.workload}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=600
    )
    wall = time.time() - start

    output = result.stdout + "\n" + result.stderr
    # Print abbreviated output
    for line in output.splitlines():
        if any(
            kw in line
            for kw in [
                "Batch ",
                "Benchmark Results",
                "Policy:",
                "Total Time:",
                "Throughput",
                "TTFT",
                "Mean:",
                "P50",
                "P95",
                "P99",
                "hit rate",
            ]
        ):
            print(f"  {line.strip()}")

    if result.returncode != 0:
        print(f"  *** FAILED (exit {result.returncode}) ***")
        # Print last 5 lines of stderr for debugging
        for line in result.stderr.strip().splitlines()[-5:]:
            print(f"  ERR: {line}")
        return {
            "policy": policy,
            "budget": budget,
            "status": "FAILED",
        }

    return {
        "policy": policy,
        "budget": budget,
        "workload": args.workload,
        "status": "OK",
        "total_time_s": _extract(_TOTAL_TIME_RE, output),
        "throughput_tok_s": _extract(_THROUGHPUT_RE, output),
        "out_throughput_tok_s": _extract(_OUT_THROUGHPUT_RE, output),
        "ttft_mean_ms": _extract(_TTFT_MEAN_RE, output),
        "ttft_p50_ms": _extract(_TTFT_P50_RE, output),
        "ttft_p99_ms": _extract(_TTFT_P99_RE, output),
        "prefix_hit_rate_pct": _extract(_HIT_RATE_RE, output),
        "wall_time_s": f"{wall:.1f}",
    }


def main():
    args = parse_args()

    results = []
    for policy in args.policies:
        for budget in args.budgets:
            row = run_single(policy, budget, args)
            results.append(row)

    # Print summary table
    if not results:
        return

    fields = [
        "policy", "budget", "workload", "status",
        "total_time_s", "throughput_tok_s", "out_throughput_tok_s",
        "ttft_mean_ms", "ttft_p50_ms", "ttft_p99_ms",
        "prefix_hit_rate_pct", "wall_time_s",
    ]

    print(f"\n{'='*80}")
    print("SWEEP RESULTS")
    print(f"{'='*80}")

    # Print as aligned table
    header = f"{'Policy':<10} {'Budget':>7} {'Workload':<16} " \
             f"{'TTFT_mean':>10} {'TTFT_p50':>10} {'TTFT_p99':>10} " \
             f"{'Tput':>10} {'HitRate':>8} {'Status':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.get('policy','?'):<10} "
            f"{r.get('budget','?'):>7} "
            f"{r.get('workload','?'):<16} "
            f"{r.get('ttft_mean_ms',''):>10} "
            f"{r.get('ttft_p50_ms',''):>10} "
            f"{r.get('ttft_p99_ms',''):>10} "
            f"{r.get('throughput_tok_s',''):>10} "
            f"{r.get('prefix_hit_rate_pct',''):>8} "
            f"{r.get('status','?'):<6}"
        )

    # Write CSV
    csv_path = args.output_csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
