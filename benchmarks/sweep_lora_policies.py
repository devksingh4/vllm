# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep driver that runs benchmark_lora_e2e.py for every (policy, scenario)
combination and aggregates results to a JSON file. Designed to be resumable:
re-running picks up where it left off based on results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

POLICIES_BASE = [
    "lru",
    # "arc",  # pre-existing bug: missing self.stats in __init__
    "sieve",
    "s3fifo",
    "tinylfu",
    "lru_k",
    "lora_twolevel",
]
LORA_COUPLINGS = [
    "lora_tight",
    "lora_loose",
    "lora_hysteresis",
    "lora_soft",
    "lora_freqweighted",
    "lora_correlated",
    "lora_budget",
    "lora_adabudget",
    "lora_costaware",
    "lora_ghost",
    "lora_position",
    "lora_prefixtree",
]
# lora_twolevel is standalone, not a decorator, so excluded from inner bases.
INNER_BASES = ["lru", "sieve", "s3fifo", "tinylfu", "lru_k"]  # arc excluded

ALL_POLICIES = POLICIES_BASE + [
    f"{c}:{b}" for c in LORA_COUPLINGS for b in INNER_BASES
]

SCENARIOS = ["adapter_thrashing", "adapter_locality", "mixed_popularity"]

COMMON_ARGS = [
    "--model", "Qwen/Qwen2.5-1.5B",
    "--lora-path", "kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter",
    "--num-adapters", "16",
    "--max-loras", "2",
    "--max-cpu-loras", "16",
    "--num-requests", "80",
    "--batch-size", "8",
    "--burst-len", "8",
    "--prefix-words", "800",
    "--suffix-words", "200",
    "--max-tokens", "8",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.5",
    "--kv-offloading-size", "0.5",
]

STATS_RE = re.compile(
    r"Policy stats \[([^\]]+)\]: "
    r"touch=(\d+) calls \((\d+) blocks\) \| "
    r"evict=(\d+) calls \((\d+) blocks, (\d+) failed\) \| "
    r"avg_scan_steps=(\S+) \| "
    r"touch:evict block ratio=(\S+) \| "
    r"inserts=(\d+) removes=(\d+) gets=(\d+) \| "
    r"cache_size_at_last_evict=(\d+) \| "
    r"cpu_offload_hit_rate=(\S+) \((\d+)/(\d+) blocks\)"
)
THROUGHPUT_RE = re.compile(r"Total Token Throughput:\s+([\d.]+)")
OUT_THROUGHPUT_RE = re.compile(r"Output Only Throughput:\s+([\d.]+)")
TIME_RE = re.compile(r"Total Time:\s+([\d.]+)")
P50_RE = re.compile(r"P50:\s+([\d.]+)")
P95_RE = re.compile(r"P95:\s+([\d.]+)")
P99_RE = re.compile(r"P99:\s+([\d.]+)")
INPUT_TOKENS_RE = re.compile(r"Total Input Tokens:\s+(\d+)")
OUTPUT_TOKENS_RE = re.compile(r"Total Output Tokens:\s+(\d+)")


def parse_log(text: str) -> dict:
    out: dict = {}
    if m := STATS_RE.search(text):
        out.update(
            policy_logged=m.group(1),
            touch_calls=int(m.group(2)),
            touch_blocks=int(m.group(3)),
            evict_calls=int(m.group(4)),
            evict_blocks=int(m.group(5)),
            evict_failed=int(m.group(6)),
            avg_scan_steps=m.group(7),
            touch_evict_ratio=m.group(8),
            inserts=int(m.group(9)),
            removes=int(m.group(10)),
            gets=int(m.group(11)),
            cache_size_at_last_evict=int(m.group(12)),
            hit_rate_pct=m.group(13),
            hit_blocks=int(m.group(14)),
            lookup_blocks=int(m.group(15)),
        )
    if m := THROUGHPUT_RE.search(text):
        out["throughput"] = float(m.group(1))
    if m := OUT_THROUGHPUT_RE.search(text):
        out["out_throughput"] = float(m.group(1))
    if m := TIME_RE.search(text):
        out["total_time"] = float(m.group(1))
    if m := P50_RE.search(text):
        out["p50_s"] = float(m.group(1))
    if m := P95_RE.search(text):
        out["p95_s"] = float(m.group(1))
    if m := P99_RE.search(text):
        out["p99_s"] = float(m.group(1))
    if m := INPUT_TOKENS_RE.search(text):
        out["input_tokens"] = int(m.group(1))
    if m := OUTPUT_TOKENS_RE.search(text):
        out["output_tokens"] = int(m.group(1))
    return out


def safe_name(s: str) -> str:
    return s.replace(":", "__").replace("/", "_")


def run_one(
    python: str,
    bench_path: Path,
    policy: str,
    scenario: str,
    outdir: Path,
    timeout_s: int,
) -> dict:
    log_path = outdir / f"{safe_name(policy)}__{scenario}.log"
    cmd = [
        python, str(bench_path),
        *COMMON_ARGS,
        "--scenario", scenario,
        "--policy", policy,
    ]
    env = os.environ.copy()
    env["VLLM_KV_OFFLOAD_POLICY"] = policy
    t0 = time.time()
    try:
        with open(log_path, "wb") as f:
            r = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_s,
            )
        ok = r.returncode == 0
        timed_out = False
    except subprocess.TimeoutExpired:
        ok = False
        timed_out = True
    elapsed = time.time() - t0
    text = log_path.read_text(errors="replace")
    parsed = parse_log(text)
    return {
        "policy": policy,
        "scenario": scenario,
        "ok": ok,
        "timed_out": timed_out,
        "elapsed_wallclock_s": elapsed,
        "log_path": str(log_path),
        **parsed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--python",
        default="/home/zhuang/fal/bin/python",
        help="Python interpreter with vllm installed",
    )
    p.add_argument(
        "--bench",
        default="benchmarks/benchmark_lora_e2e.py",
        help="Path to the benchmark script",
    )
    p.add_argument(
        "--outdir", default="/tmp/lora_sweep_logs",
        help="Where to put per-run logs and results.json",
    )
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument(
        "--policies-only",
        nargs="*",
        default=None,
        help="Restrict to these policy names (default: all)",
    )
    p.add_argument(
        "--scenarios-only",
        nargs="*",
        default=None,
        help="Restrict to these scenarios (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.json"

    policies = args.policies_only or ALL_POLICIES
    scenarios = args.scenarios_only or SCENARIOS

    # Load any existing results to skip completed runs.
    done: dict[tuple[str, str], dict] = {}
    if results_path.exists():
        try:
            for r in json.loads(results_path.read_text()):
                done[(r["policy"], r["scenario"])] = r
        except Exception as e:  # pragma: no cover
            print(f"[warn] could not load existing results: {e}")

    plan = [(p, s) for p in policies for s in scenarios]
    remaining = [t for t in plan if t not in done]
    print(
        f"Sweep: {len(plan)} runs total ({len(remaining)} remaining, "
        f"{len(done)} cached). Out: {outdir}"
    )
    sys.stdout.flush()

    results = list(done.values())
    started = time.time()
    for i, (p, s) in enumerate(plan):
        if (p, s) in done:
            continue
        idx = len(results) + 1
        wall = time.time() - started
        print(
            f"[{idx}/{len(plan)} t+{wall:.0f}s] running {p} | {s}",
            flush=True,
        )
        rec = run_one(
            args.python, Path(args.bench), p, s, outdir, args.timeout_s
        )
        results.append(rec)
        results_path.write_text(json.dumps(results, indent=2))
        evict = rec.get("evict_calls", "?")
        thr = rec.get("throughput", "?")
        hr = rec.get("hit_rate_pct", "?")
        out_thr = rec.get("out_throughput", "?")
        status = "OK" if rec["ok"] else ("TIMEOUT" if rec["timed_out"] else "FAIL")
        print(
            f"   {status} time={rec['elapsed_wallclock_s']:.0f}s "
            f"thr={thr} out_thr={out_thr} evict={evict} hit={hr}",
            flush=True,
        )
    print(f"DONE in {(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
