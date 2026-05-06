# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated GPU-prefix-cache eviction-policy sweep.

Holds CPU offload policy = lru constant; varies VLLM_GPU_PREFIX_CACHE_POLICY
across {lru, s3fifo, sieve} on the same workloads as sweep_lora_ttft.py.
Answers: does the GPU prefix-cache eviction policy itself affect TTFT,
independent of LoRA awareness or CPU offload policy?
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

GPU_POLICIES = ["lru", "s3fifo", "sieve"]
SCENARIOS = ["adapter_thrashing", "adapter_locality", "mixed_popularity"]

COMMON_ARGS = [
    "--model", "Qwen/Qwen2.5-3B",
    "--lora-path", "essam04/Qwen2.5-3B-FineTome-LoRA",
    "--num-adapters", "64",
    "--max-loras", "4",
    "--max-cpu-loras", "64",
    "--num-requests", "256",
    "--batch-size", "16",
    "--burst-len", "8",
    "--prefix-words", "1600",
    "--suffix-words", "200",
    "--max-tokens", "8",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--kv-offloading-size", "4.0",
    "--policy", "lru",  # CPU offload policy held constant
]

TTFT_RE = re.compile(
    r"TTFT\s+P50:\s+([\d\.\-]+)s\s+P95:\s+([\d\.\-]+)s\s+P99:\s+([\d\.\-]+)s\s+"
    r"mean:\s+([\d\.\-]+)s\s+min:\s+([\d\.\-]+)s\s+max:\s+([\d\.\-]+)s"
)
E2E_RE = re.compile(
    r"E2E\s+P50:\s+([\d\.\-]+)s\s+P95:\s+([\d\.\-]+)s\s+P99:\s+([\d\.\-]+)s\s+"
    r"mean:\s+([\d\.\-]+)s"
)
TIME_RE = re.compile(r"Total wallclock:\s+([\d\.]+)\s*s")
REQS_RE = re.compile(r"Requests measured:\s+(\d+)")
HIT_RE = re.compile(r"cpu_offload_hit_rate=(\S+)\s+\((\d+)/(\d+) blocks\)")


def parse_log(text: str) -> dict:
    out: dict = {}
    if m := TTFT_RE.search(text):
        out.update(
            ttft_p50=float(m.group(1)),
            ttft_p95=float(m.group(2)),
            ttft_p99=float(m.group(3)),
            ttft_mean=float(m.group(4)),
            ttft_min=float(m.group(5)),
            ttft_max=float(m.group(6)),
        )
    if m := E2E_RE.search(text):
        out.update(
            e2e_p50=float(m.group(1)),
            e2e_p95=float(m.group(2)),
            e2e_p99=float(m.group(3)),
            e2e_mean=float(m.group(4)),
        )
    if m := TIME_RE.search(text):
        out["total_time"] = float(m.group(1))
    if m := REQS_RE.search(text):
        out["requests_measured"] = int(m.group(1))
    if m := HIT_RE.search(text):
        out["hit_blocks"] = int(m.group(2))
        out["lookup_blocks"] = int(m.group(3))
        out["hit_rate_pct"] = m.group(1)
    return out


def run_one(
    python: str,
    bench_path: Path,
    gpu_policy: str,
    scenario: str,
    outdir: Path,
    timeout_s: int,
) -> dict:
    log_path = outdir / f"gpu_{gpu_policy}__{scenario}.log"
    cmd = [
        python, str(bench_path),
        *COMMON_ARGS,
        "--scenario", scenario,
    ]
    env = os.environ.copy()
    env["VLLM_GPU_PREFIX_CACHE_POLICY"] = gpu_policy
    env["VLLM_KV_OFFLOAD_POLICY"] = "lru"  # CPU policy held constant
    t0 = time.time()
    try:
        with open(log_path, "wb") as f:
            r = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                env=env, timeout=timeout_s,
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
        "gpu_policy": gpu_policy,
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
        default="/home/dsingh/source/devksingh4/vllm/.venv/bin/python",
    )
    p.add_argument("--bench", default="benchmarks/benchmark_lora_ttft.py")
    p.add_argument(
        "--outdir",
        default=(
            "/home/dsingh/source/devksingh4/vllm/lora_sweep_logs/gpu_policy_sweep"
        ),
    )
    p.add_argument("--timeout-s", type=int, default=900)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.json"

    done: dict[tuple[str, str], dict] = {}
    if results_path.exists():
        try:
            for r in json.loads(results_path.read_text()):
                done[(r["gpu_policy"], r["scenario"])] = r
        except Exception as e:
            print(f"[warn] could not load existing results: {e}")

    plan = [(p, s) for p in GPU_POLICIES for s in SCENARIOS]
    print(
        f"GPU policy sweep: {len(plan)} runs total "
        f"({len(plan) - len(done)} remaining, {len(done)} cached). Out: {outdir}",
        flush=True,
    )

    results = list(done.values())
    started = time.time()
    for i, (gp, sc) in enumerate(plan):
        if (gp, sc) in done:
            continue
        idx = len(results) + 1
        wall = time.time() - started
        print(
            f"[{idx}/{len(plan)} t+{wall:.0f}s] running gpu={gp} | {sc}",
            flush=True,
        )
        rec = run_one(
            args.python, Path(args.bench), gp, sc, outdir, args.timeout_s
        )
        results.append(rec)
        results_path.write_text(json.dumps(results, indent=2))
        ttft = rec.get("ttft_p50", "?")
        p95 = rec.get("ttft_p95", "?")
        hit = rec.get("hit_rate_pct", "?")
        status = (
            "OK" if rec["ok"]
            else ("TIMEOUT" if rec["timed_out"] else "FAIL")
        )
        print(
            f"   {status} time={rec['elapsed_wallclock_s']:.0f}s "
            f"ttft_p50={ttft} ttft_p95={p95} hit={hit}",
            flush=True,
        )
    print(f"DONE in {(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
