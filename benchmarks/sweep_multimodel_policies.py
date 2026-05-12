#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep driver that runs benchmark_kv_cache.py for every multi-model policy
and scenario combination and aggregates results to JSON.
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

BENCHMARK = str(Path(__file__).parent / "benchmark_kv_cache.py")

BLOCK_SIZE = 16  # vLLM default
TOKENS_PER_WORD = 1.5  # conservative estimate for English text

# ---------------------------------------------------------------------------
# Scenarios (workload shapes)
# ---------------------------------------------------------------------------
# Token budget estimates: prefix_words × TOKENS_PER_WORD × num_prefixes.
# Prefix blocks ≈ tokens / BLOCK_SIZE (each unique prefix occupies blocks).
# "cheap" = high-traffic tenant; "expensive" = costly-to-recompute tenant.

SCENARIOS: dict[str, dict] = {
    # Balanced: equal traffic, equal prefix length.  Policies should be
    # neutral — any TTFT difference here is noise or overhead.
    # ~2 × 25 × 900 = 45K tokens ≈ 2812 blocks
    "balanced": {
        "cheap": {"num_requests": 60, "prefix_words": 600, "num_prefixes": 25},
        "expensive": {"num_requests": 60, "prefix_words": 600, "num_prefixes": 25},
    },
    # Traffic skew only: cheap floods 5× more requests; both use the same
    # prefix length so cost-aware adds nothing — only caps should help.
    # cheap: 25 × 900 = 22.5K tok ≈ 1406 blk; expensive: 15 × 900 = 13.5K tok ≈ 843 blk
    # Total ≈ 2249 blocks → exceeds tight (2000).
    "traffic_skew": {
        "cheap": {"num_requests": 100, "prefix_words": 600, "num_prefixes": 25},
        "expensive": {"num_requests": 20, "prefix_words": 600, "num_prefixes": 15},
    },
    # Cost skew only: equal traffic but expensive prefixes are 10× longer.
    # cheap: 20 × 300 = 6K tok ≈ 375 blk; expensive: 10 × 3000 = 30K tok ≈ 1875 blk
    # Total ≈ 2250 blocks → exceeds tight (2000).
    "cost_skew": {
        "cheap": {"num_requests": 60, "prefix_words": 200, "num_prefixes": 20},
        "expensive": {"num_requests": 60, "prefix_words": 2000, "num_prefixes": 10},
    },
    # Full skew: cheap sends 5× more AND expensive has 10× longer prefixes.
    # cheap: 20 × 300 = 6K tok ≈ 375 blk; expensive: 10 × 3000 = 30K tok ≈ 1875 blk
    # Total ≈ 2250 blocks → exceeds tight (2000).
    "full_skew": {
        "cheap": {"num_requests": 100, "prefix_words": 200, "num_prefixes": 20},
        "expensive": {"num_requests": 20, "prefix_words": 2000, "num_prefixes": 10},
    },
}

# ---------------------------------------------------------------------------
# Derived: cache regimes (GPU blocks, not memory — GPU-independent)
# ---------------------------------------------------------------------------
# Computed from SCENARIOS so that tight always forces eviction and loose
# always fits everything.  Using --num-gpu-blocks-override avoids dependence
# on model size / GPU size.
#
# Tuning knobs (change these, not the block counts):
TIGHT_FRAC = 0.75  # tight cache = 75% of smallest working set → eviction guaranteed
LOOSE_MULT = 3.0  # loose cache = 3× largest working set → everything fits


def _scenario_blocks(scenario: dict) -> int:
    """Total unique prefix blocks for one scenario."""
    total = 0
    for cfg in scenario.values():
        tokens = cfg["prefix_words"] * TOKENS_PER_WORD * cfg["num_prefixes"]
        total += int(tokens / BLOCK_SIZE)
    return total


_WORKING_SETS = {name: _scenario_blocks(scen) for name, scen in SCENARIOS.items()}
_MIN_WS = min(_WORKING_SETS.values())
_MAX_WS = max(_WORKING_SETS.values())

CACHE_REGIMES: dict[str, int] = {
    "tight": int(_MIN_WS * TIGHT_FRAC),
    "loose": int(_MAX_WS * LOOSE_MULT),
}

# ---------------------------------------------------------------------------
# Engine args common to every run.
# --num-gpu-blocks-override is injected per-regime by the runner.
# ---------------------------------------------------------------------------

COMMON_ARGS = [
    "--model",
    "Qwen/Qwen2.5-1.5B",
    "--enable-prefix-caching",
    "--enforce-eager",
    "--max-model-len",
    "4096",
    "--batch-size",
    "16",
    "--suffix-words",
    "30",
    "--max-tokens",
    "8",
]

# ---------------------------------------------------------------------------
# Derived: policy conditions
# ---------------------------------------------------------------------------
# Caps are fractions of the tight cache.  cheap < 50% prevents monopolisation;
# expensive > 50% lets it retain long prefixes.  Sum > 100% because caps limit
# max, not reserve.  Under the loose regime caps rarely bind, confirming the
# mechanism is harmless when cache is abundant.
#
# Cost ratio is derived from the max prefix-length ratio across scenarios
# (i.e. how much more expensive the "expensive" partition's prefixes are to
# recompute relative to "cheap").

CHEAP_CAP_FRAC = 0.40  # cheap can pin at most 40% of tight cache
EXPENSIVE_CAP_FRAC = 0.70  # expensive can pin at most 70% of tight cache

_TIGHT = CACHE_REGIMES["tight"]
DEFAULT_CHEAP_CAP = int(_TIGHT * CHEAP_CAP_FRAC)
DEFAULT_EXPENSIVE_CAP = int(_TIGHT * EXPENSIVE_CAP_FRAC)


def _max_prefix_ratio() -> float:
    """Max ratio of expensive/cheap prefix_words across all scenarios."""
    best = 1.0
    for scen in SCENARIOS.values():
        words = {pid: cfg["prefix_words"] for pid, cfg in scen.items()}
        lo, hi = min(words.values()), max(words.values())
        if lo > 0:
            best = max(best, hi / lo)
    return best


DEFAULT_COST_RATIO = _max_prefix_ratio()


def _build_policies(
    cheap_cap: int,
    expensive_cap: int,
    cost_ratio: float,
) -> dict[str, list[str]]:
    """Return {policy_name: extra_cli_args} for each mechanism."""
    caps = json.dumps({"cheap": cheap_cap, "expensive": expensive_cap})
    cost = json.dumps({"cheap": 1.0, "expensive": cost_ratio})
    return {
        "baseline": [],
        "caps": ["--kv-cache-partition-ref-caps", caps],
        "cost": ["--kv-cache-partition-eviction-cost", cost],
        "caps_cost": [
            "--kv-cache-partition-ref-caps",
            caps,
            "--kv-cache-partition-eviction-cost",
            cost,
        ],
    }


# ---------------------------------------------------------------------------
# Working-set sanity checks
# ---------------------------------------------------------------------------


def _per_partition_blocks(scenario: dict) -> dict[str, int]:
    """Estimate unique prefix blocks per partition."""
    out = {}
    for pid, cfg in scenario.items():
        tokens = cfg["prefix_words"] * TOKENS_PER_WORD * cfg["num_prefixes"]
        out[pid] = int(tokens / BLOCK_SIZE)
    return out


def _print_pressure_table(scenarios: dict[str, dict], regimes: dict[str, int]) -> None:
    """Print a table showing working-set vs. cache size for each scenario.

    Because regimes are derived from scenarios, every scenario's working set
    is guaranteed to exceed the tight cache by construction.  This table is
    printed at startup so the operator can eyeball the numbers.
    """
    tight_blocks = regimes["tight"]
    loose_blocks = regimes["loose"]
    print("\nDerived block budgets:")
    print(f"  tight = {tight_blocks} blocks  (TIGHT_FRAC={TIGHT_FRAC})")
    print(f"  loose = {loose_blocks} blocks  (LOOSE_MULT={LOOSE_MULT})")
    print(f"  cheap_cap  = {DEFAULT_CHEAP_CAP} blocks  ({CHEAP_CAP_FRAC:.0%} of tight)")
    print(
        f"  expensive_cap = {DEFAULT_EXPENSIVE_CAP} blocks  ({EXPENSIVE_CAP_FRAC:.0%} of tight)"
    )
    print(f"  cost_ratio = {DEFAULT_COST_RATIO:.1f}x  (max prefix_words ratio)")
    print("\nWorking-set estimates (unique prefix blocks):")
    for name, scen in scenarios.items():
        blks = _per_partition_blocks(scen)
        total = sum(blks.values())
        parts = "  ".join(f"{pid}={b}" for pid, b in sorted(blks.items()))
        vs_tight = f"{total / tight_blocks:.1f}x tight"
        vs_loose = f"{total / loose_blocks:.2f}x loose"
        print(
            f"  {name:<16} total={total:>5} blocks  ({parts})  [{vs_tight}, {vs_loose}]"
        )
        assert total > tight_blocks, (
            f"BUG: {name} working set ({total}) <= tight cache ({tight_blocks}). "
            f"This should be impossible — check TIGHT_FRAC or scenario config."
        )
    print()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Per-partition line:  [cheap] n=80  mean=123.4ms  p50=100.0ms  p99=200.0ms
_PART_RE = re.compile(
    r"\[(\S+)\]\s+n=(\d+)\s+mean=([\d.]+)ms\s+p50=([\d.]+)ms\s+p99=([\d.]+)ms"
)
_P50_RE = re.compile(r"P50 \(median\):\s+([\d.]+) ms")
_P99_RE = re.compile(r"P99:\s+([\d.]+) ms")
_THROUGHPUT_RE = re.compile(r"Total Token Throughput:\s+([\d.]+)")
_HITRATE_RE = re.compile(r"cpu_offload_hit_rate=(\S+)\s+\((\d+)/(\d+) blocks\)")


def _parse_log(text: str) -> dict:
    out: dict = {}
    partitions: dict[str, dict] = {}
    for m in _PART_RE.finditer(text):
        partitions[m.group(1)] = {
            "n": int(m.group(2)),
            "mean_ms": float(m.group(3)),
            "p50_ms": float(m.group(4)),
            "p99_ms": float(m.group(5)),
        }
    if partitions:
        out["partitions"] = partitions
    if m := _P50_RE.search(text):
        out["overall_p50_ms"] = float(m.group(1))
    if m := _P99_RE.search(text):
        out["overall_p99_ms"] = float(m.group(1))
    if m := _THROUGHPUT_RE.search(text):
        out["throughput"] = float(m.group(1))
    if m := _HITRATE_RE.search(text):
        out["hit_blocks"] = int(m.group(2))
        out["lookup_blocks"] = int(m.group(3))
        out["hit_rate_pct"] = m.group(1)
    return out


# ---------------------------------------------------------------------------
# Runner (one combination)
# ---------------------------------------------------------------------------


def _run_key(scenario: str, regime: str, policy: str) -> str:
    return f"{scenario}__{regime}__{policy}"


def _run_one(
    python: str,
    scenario: str,
    regime: str,
    policy: str,
    policy_args: list[str],
    outdir: Path,
    timeout_s: int,
) -> dict:
    key = _run_key(scenario, regime, policy)
    log_path = outdir / f"{key}.log"
    partition_config = SCENARIOS[scenario]
    num_blocks = CACHE_REGIMES[regime]

    cmd = [
        python,
        BENCHMARK,
        "--workload",
        "multi-partition",
        "--partition-config",
        json.dumps(partition_config),
        "--num-gpu-blocks-override",
        str(num_blocks),
        *COMMON_ARGS,
        *policy_args,
    ]
    env = os.environ.copy()
    env["VLLM_KV_OFFLOAD_POLICY"] = "lru"

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
    parsed = _parse_log(text)
    return {
        "key": key,
        "scenario": scenario,
        "regime": regime,
        "policy": policy,
        "ok": ok,
        "timed_out": timed_out,
        "elapsed_wallclock_s": elapsed,
        "log_path": str(log_path),
        **parsed,
    }


# ---------------------------------------------------------------------------
# Summary table (grouped by scenario+regime)
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict]) -> None:
    pids = sorted(
        {pid for r in results for pid in r.get("partitions", {})},
    )
    if not pids:
        print("\n[WARN] No per-partition TTFT parsed — check logs.")
        return

    col = 12
    hdr = f"{'scenario':<16} {'regime':<7} {'policy':<12}"
    for pid in pids:
        hdr += f" {pid + ' P50':>{col}} {pid + ' P99':>{col}}"
    hdr += f" {'all P50':>{col}} {'all P99':>{col}} {'time':>6}"

    print(f"\n{'=' * len(hdr)}")
    print("MULTI-MODEL POLICY SWEEP RESULTS — PER-PARTITION TTFT (ms)")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    prev_group = ""
    for r in results:
        group = f"{r.get('scenario', '?')}__{r.get('regime', '?')}"
        if group != prev_group:
            if prev_group:
                print()  # blank line between scenario×regime groups
            prev_group = group

        status = "OK" if r.get("ok") else ("TIMEOUT" if r.get("timed_out") else "FAIL")
        row = f"{r.get('scenario', '?'):<16} {r.get('regime', '?'):<7} {r.get('policy', '?'):<12}"
        for pid in pids:
            p = r.get("partitions", {}).get(pid, {})
            p50 = p.get("p50_ms")
            p99 = p.get("p99_ms")
            row += f" {(f'{p50:.1f}' if p50 is not None else '-'):>{col}}"
            row += f" {(f'{p99:.1f}' if p99 is not None else '-'):>{col}}"
        op50 = r.get("overall_p50_ms")
        op99 = r.get("overall_p99_ms")
        row += f" {(f'{op50:.1f}' if op50 is not None else '-'):>{col}}"
        row += f" {(f'{op99:.1f}' if op99 is not None else '-'):>{col}}"
        row += f" {r.get('elapsed_wallclock_s', 0):>5.0f}s"
        if status != "OK":
            row += f"  {status}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EPILOG = f"""
Sweep axes (full cross-product = {len(SCENARIOS)} x {len(CACHE_REGIMES)} x 4 = {len(SCENARIOS) * len(CACHE_REGIMES) * 4} runs):

SCENARIOS:
  balanced       equal traffic, equal prefix lengths  (sanity baseline)
  traffic_skew   cheap sends 5x more requests         (tests caps)
  cost_skew      expensive prefixes 10x longer         (tests cost-aware)
  full_skew      both skews combined                   (worst case)

CACHE REGIMES (via --num-gpu-blocks-override, GPU-independent):
  tight   {CACHE_REGIMES["tight"]} blocks  (TIGHT_FRAC={TIGHT_FRAC} of smallest working set)
  loose   {CACHE_REGIMES["loose"]} blocks  (LOOSE_MULT={LOOSE_MULT}x of largest working set)

POLICIES:
  baseline    plain LRU, no partition awareness
  caps        per-partition ref caps
  cost        cost-aware LRU eviction
  caps_cost   both combined

All derived values (block budgets, caps, cost ratio) are computed
automatically from the scenario definitions.  VLLM_KV_OFFLOAD_POLICY=lru
is pinned for every run so cost-aware eviction can activate.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep multi-model cache partition policies (resumable)",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with vllm installed",
    )
    p.add_argument(
        "--outdir",
        default="/tmp/multimodel_sweep_logs",
        help="Directory for per-run logs and results.json",
    )
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument(
        "--cheap-cap",
        type=int,
        default=DEFAULT_CHEAP_CAP,
        help=f"Max block refs for 'cheap' partition "
        f"(default: {DEFAULT_CHEAP_CAP}, {CHEAP_CAP_FRAC * 100:.0f}%% of tight)",
    )
    p.add_argument(
        "--expensive-cap",
        type=int,
        default=DEFAULT_EXPENSIVE_CAP,
        help=f"Max block refs for 'expensive' partition "
        f"(default: {DEFAULT_EXPENSIVE_CAP}, {EXPENSIVE_CAP_FRAC * 100:.0f}%% of tight)",
    )
    p.add_argument(
        "--cost-ratio",
        type=float,
        default=DEFAULT_COST_RATIO,
        help=f"Eviction cost multiplier for 'expensive' "
        f"(default: {DEFAULT_COST_RATIO:.1f}, derived from max prefix ratio)",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default=",".join(SCENARIOS),
        help=f"Comma-separated scenarios (default: all). Choices: {list(SCENARIOS)}",
    )
    p.add_argument(
        "--regimes",
        type=str,
        default=",".join(CACHE_REGIMES),
        help=f"Comma-separated cache regimes. Choices: {list(CACHE_REGIMES)}",
    )
    p.add_argument(
        "--policies",
        type=str,
        default="baseline,caps,cost,caps_cost",
        help="Comma-separated policy conditions to run",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.json"

    policy_map = _build_policies(
        args.cheap_cap,
        args.expensive_cap,
        args.cost_ratio,
    )
    sel_scenarios = [s.strip() for s in args.scenarios.split(",")]
    sel_regimes = [r.strip() for r in args.regimes.split(",")]
    sel_policies = [p.strip() for p in args.policies.split(",")]

    for s in sel_scenarios:
        if s not in SCENARIOS:
            print(f"[error] unknown scenario: {s!r}", file=sys.stderr)
            sys.exit(2)
    for r in sel_regimes:
        if r not in CACHE_REGIMES:
            print(f"[error] unknown regime: {r!r}", file=sys.stderr)
            sys.exit(2)
    for p in sel_policies:
        if p not in policy_map:
            print(f"[error] unknown policy: {p!r}", file=sys.stderr)
            sys.exit(2)

    # Build full plan (scenario × regime × policy)
    plan: list[tuple[str, str, str]] = [
        (sc, rg, po)
        for sc in sel_scenarios
        for rg in sel_regimes
        for po in sel_policies
    ]

    # Resume: skip already-successful runs (failed/timed-out are retried).
    done: dict[str, dict] = {}
    if results_path.exists():
        try:
            for rec in json.loads(results_path.read_text()):
                if rec.get("ok"):
                    done[rec["key"]] = rec
        except Exception as e:
            print(f"[warn] could not load existing results: {e}")

    _print_pressure_table({s: SCENARIOS[s] for s in sel_scenarios}, CACHE_REGIMES)

    remaining = [t for t in plan if _run_key(*t) not in done]
    print("=" * 70)
    print("Multi-model partition policy sweep")
    print(f"  Scenarios:  {sel_scenarios}")
    print(f"  Regimes:    {sel_regimes}")
    print(f"  Policies:   {sel_policies}")
    print(f"  Total runs: {len(plan)} ({len(remaining)} remaining, {len(done)} cached)")
    print(f"  Caps:       cheap={args.cheap_cap}  expensive={args.expensive_cap}")
    print(f"  Cost ratio: expensive={args.cost_ratio}x")
    print(f"  Outdir:     {outdir}")
    print("=" * 70, flush=True)

    results: list[dict] = []
    started = time.time()
    n_ran = 0
    for sc, rg, po in plan:
        key = _run_key(sc, rg, po)
        if key in done:
            results.append(done[key])
            continue
        n_ran += 1
        wall = time.time() - started
        print(
            f"\n[{n_ran}/{len(remaining)} t+{wall:.0f}s] {sc} | {rg} | {po}",
            flush=True,
        )

        rec = _run_one(
            args.python,
            sc,
            rg,
            po,
            policy_map[po],
            outdir,
            args.timeout_s,
        )
        results.append(rec)
        if rec["ok"]:
            done[key] = rec
        results_path.write_text(json.dumps(results, indent=2))

        status = "OK" if rec["ok"] else ("TIMEOUT" if rec["timed_out"] else "FAIL")
        p50 = rec.get("overall_p50_ms", "?")
        parts = rec.get("partitions", {})
        part_str = "  ".join(
            f"{pid}={v.get('p50_ms', '?')}ms" for pid, v in sorted(parts.items())
        )
        print(
            f"   {status} time={rec['elapsed_wallclock_s']:.0f}s p50={p50}  {part_str}",
            flush=True,
        )

    elapsed = (time.time() - started) / 60
    print(f"\nDONE in {elapsed:.1f} min", flush=True)
    _print_summary(results)


if __name__ == "__main__":
    main()
