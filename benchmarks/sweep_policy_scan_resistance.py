#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sweep runner for benchmark_policy_scan_resistance.py.

Varies --num-oneshots across several levels for each of lru / sieve / s3fifo
and prints a comparison table at the end.

Usage:
    python benchmarks/sweep_policy_scan_resistance.py [args forwarded to benchmark]

Extra flags (not forwarded):
    --oneshot-levels  Comma-separated list of num-oneshots values to sweep
                      (default: 3,7,12)
    --policies        Comma-separated list of policies to test
                      (default: lru,sieve,s3fifo)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

BENCHMARK = Path(__file__).parent / "benchmark_policy_scan_resistance.py"
PYTHON = sys.executable


def parse_result(stdout: str) -> dict | None:
    """Extract key timing metrics from benchmark stdout."""
    patterns = {
        "phase1": r"Phase 1 — cold.*?:\s+([\d.]+)s",
        "phase2": r"Phase 2 — bridge.*?:\s+([\d.]+)s",
        "phase3": r"Phase 3 — Prompt A \(CPU load.*?\):\s+([\d.]+)s",
        "phase4": r"Phase 4 —.*?scan.*?:\s+([\d.]+)s",
        "phase5": r"Phase 5 — Prompt A \(survived.*?\):\s+([\d.]+)s",
        "ratio_vs_cpu": r"Ratio\s+Phase5 vs Phase3.*?:\s+([\d.]+)x",
        "ratio_vs_cold": r"Ratio\s+Phase5 vs Phase1.*?:\s+([\d.]+)x",
        "verdict": r"Verdict:\s+(.+)",
    }
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            result[key] = m.group(1).strip() if key == "verdict" else float(m.group(1))
    return result if len(result) >= 6 else None


def run_one(policy: str, num_oneshots: int, extra_args: list[str]) -> dict | None:
    cmd = [
        PYTHON,
        str(BENCHMARK),
        "--policy", policy,
        "--num-oneshots", str(num_oneshots),
        *extra_args,
    ]
    print(f"\n{'─'*60}")
    print(f"  policy={policy}  num-oneshots={num_oneshots}")
    print(f"{'─'*60}")
    sys.stdout.flush()

    # Stream output in real-time while also capturing it for later parsing.
    captured_lines: list[str] = []
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            sys.stdout.flush()
            captured_lines.append(line)

    stdout = "".join(captured_lines)
    result = parse_result(stdout)
    if result is None:
        print(f"  [WARNING] Could not parse result for policy={policy} oneshots={num_oneshots}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep num-oneshots across lru/sieve/s3fifo and print a table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--oneshot-levels",
        type=str,
        default="3,7,12",
        help="Comma-separated num-oneshots values (default: 3,7,12)",
    )
    p.add_argument(
        "--policies",
        type=str,
        default="lru,sieve,s3fifo",
        help="Comma-separated policies to test (default: lru,sieve,s3fifo)",
    )
    # Forward all remaining args to the benchmark unchanged
    args, extra = p.parse_known_args()
    # Strip any bare '--' separator the user may have passed
    extra = [a for a in extra if a != "--"]

    policies = [pol.strip() for pol in args.policies.split(",")]
    levels = [int(x.strip()) for x in args.oneshot_levels.split(",")]

    print("=" * 70)
    print("Scan-resistance policy sweep")
    print(f"  Policies:      {policies}")
    print(f"  Oneshot sweep: {levels}")
    print(f"  Extra args:    {extra}")
    print("=" * 70)

    # Results dict: (policy, num_oneshots) -> parsed result
    results: dict[tuple[str, int], dict] = {}

    for policy in policies:
        for n in levels:
            r = run_one(policy, n, extra)
            if r:
                results[(policy, n)] = r

    # --- Summary table ---
    print("\n")
    print("=" * 70)
    print("SWEEP RESULTS — Phase5/Phase3 ratio (1.0=survived, >2=evicted)")
    print("=" * 70)

    col_w = 14
    header = f"{'Policy':<10}" + "".join(f"{'N='+str(n):>{col_w}}" for n in levels)
    print(header)
    print("-" * len(header))

    for policy in policies:
        row_cpu   = f"{policy:<10}"
        row_cold  = f"{'':10}"
        for n in levels:
            r = results.get((policy, n))
            if r:
                ratio_cpu  = r.get("ratio_vs_cpu", float("inf"))
                ratio_cold = r.get("ratio_vs_cold", float("inf"))
                # Primary: ratio Phase5/Phase3 (CPU load baseline)
                emoji = "✓" if ratio_cpu <= 1.4 else ("~" if ratio_cpu <= 2.0 else "✗")
                row_cpu  += f"{emoji} {ratio_cpu:.2f}x p5/p3".rjust(col_w)
                row_cold += f"  {ratio_cold:.2f}x p5/p1".rjust(col_w)
            else:
                row_cpu  += f"{'ERR':>{col_w}}"
                row_cold += f"{'':>{col_w}}"
        print(row_cpu)
        print(row_cold)
        print()

    print("Legend:")
    print("  p5/p3 = Phase5/Phase3 ratio  (1.0=survived in CPU, >2=evicted)")
    print("  p5/p1 = Phase5/Phase1 ratio  (1.0=same as cold recompute)")
    print("  ✓ survived   ~ partial   ✗ evicted")

    print("\nDetailed timing:")
    print(f"  {'policy':<10} {'N':>4}  {'ph1':>7}  {'ph2':>7}  "
          f"{'ph3':>7}  {'ph4':>7}  {'ph5':>7}  verdict")
    print(f"  {'-'*75}")
    for (policy, n), r in sorted(results.items()):
        verdict_short = r.get("verdict", "?").split("—")[0].strip()
        print(
            f"  {policy:<10} {n:>4}  {r.get('phase1', 0):>7.3f}s  "
            f"{r.get('phase2', 0):>7.3f}s  {r.get('phase3', 0):>7.3f}s  "
            f"{r.get('phase4', 0):>7.3f}s  {r.get('phase5', 0):>7.3f}s  {verdict_short}"
        )


if __name__ == "__main__":
    main()
