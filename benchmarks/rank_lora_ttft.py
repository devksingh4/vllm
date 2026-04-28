# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reads ttft_sweep/results.json from sweep_lora_ttft.py and emits a ranked
markdown report focused on TTFT (P50/P95) and per-policy hit rate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SCENARIO_ORDER = ["adapter_thrashing", "adapter_locality", "mixed_popularity"]
SCENARIO_BLURB = {
    "adapter_thrashing": (
        "Round-robin across 16 LoRA adapters (max_loras=2). Highest "
        "CPU-offload reuse pressure."
    ),
    "adapter_locality": (
        "Bursts of consecutive requests on the same adapter (burst_len=8). "
        "Mostly serviced by GPU prefix cache."
    ),
    "mixed_popularity": (
        "Zipfian (alpha=1.2) over 16 adapters — a few hot adapters dominate."
    ),
}


def _hit_rate(r: dict) -> float:
    hb = r.get("hit_blocks")
    lb = r.get("lookup_blocks")
    if hb is not None and lb:
        return hb / lb
    s = r.get("hit_rate_pct")
    if s and s != "n/a":
        try:
            return float(s.rstrip("%")) / 100.0
        except ValueError:
            return float("nan")
    return float("nan")


def _f(r, key):
    v = r.get(key)
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt(x: float, digits: int = 3, nan_text: str = "  n/a") -> str:
    return f"{x:.{digits}f}" if x == x else nan_text


def fmt_pct(x: float) -> str:
    return f"{x*100:5.2f}%" if x == x else "  n/a"


def render_scenario_ttft(rows: list[dict], scenario: str) -> str:
    lines = [f"## Scenario: `{scenario}`", ""]
    lines.append(f"_{SCENARIO_BLURB.get(scenario, '')}_")
    lines.append("")
    lines.append(
        "Sorted ascending by **TTFT P50** (lower is better). "
        "Hit rate / E2E shown for context."
    )
    lines.append("")
    lines.append(
        "| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | "
        "Hit Rate |"
    )
    lines.append(
        "|-----:|:-------|---------:|---------:|---------:|--------:|"
        "---------:|"
    )
    ok = [r for r in rows if r.get("ok") and r.get("ttft_p50") is not None]
    ranked = sorted(ok, key=lambda r: _f(r, "ttft_p50"))
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | `{r['policy']}` | "
            f"{fmt(_f(r,'ttft_p50'))}s | "
            f"{fmt(_f(r,'ttft_p95'))}s | "
            f"{fmt(_f(r,'ttft_p99'))}s | "
            f"{fmt(_f(r,'e2e_p50'))}s | "
            f"{fmt_pct(_hit_rate(r))} |"
        )
    failed = [r for r in rows if not r.get("ok")]
    if failed:
        lines.append("")
        lines.append("**Failed/Timed-out runs:**")
        for r in failed:
            tag = "TIMEOUT" if r.get("timed_out") else "FAIL"
            lines.append(f"- `{r['policy']}` ({tag})")
    lines.append("")
    return "\n".join(lines)


def render_overall(per: dict[str, list[dict]]) -> str:
    """Average rank by TTFT P50 across thrashing + mixed_popularity (drop
    locality since it doesn't differentiate)."""
    keep = ["adapter_thrashing", "mixed_popularity"]
    ranks: dict[str, list[int]] = defaultdict(list)
    for s in keep:
        rows = per.get(s, [])
        for i, r in enumerate(rows, 1):
            ranks[r["policy"]].append(i)
    composite = []
    for policy, rs in ranks.items():
        if len(rs) != len(keep):
            continue
        composite.append((policy, sum(rs) / len(rs), rs))
    composite.sort(key=lambda t: t[1])

    lines = [
        "## Overall ranking — avg TTFT-P50 rank",
        "",
        "Average rank across the **two non-degenerate scenarios** "
        "(`adapter_thrashing` + `mixed_popularity`). Lower is better. "
        "`adapter_locality` is excluded because every policy's hit rate "
        "is 0% there.",
        "",
        "| Rank | Policy | Avg Rank | thrashing | mixed |",
        "|-----:|:-------|---------:|:---------:|:-----:|",
    ]
    for i, (policy, avg, rs) in enumerate(composite[:30], 1):
        lines.append(
            f"| {i} | `{policy}` | {avg:.1f} | {rs[0]} | {rs[1]} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        default="/tmp/lora_sweep_logs/ttft_sweep/results.json",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = json.loads(Path(args.results).read_text())

    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)

    parts = [
        "# LoRA-Aware Cache Policy TTFT Sweep — Results",
        "",
        f"Total runs: **{len(results)}** "
        f"({sum(1 for r in results if r.get('ok'))} ok, "
        f"{sum(1 for r in results if r.get('timed_out'))} timed out, "
        f"{sum(1 for r in results if not r.get('ok'))} failed).",
        "",
        "Per-request TTFT was measured via `LLMEngine.add_request` + "
        "`engine.step()` loop, recording `now()` when each request first "
        "emitted a token.",
        "",
    ]

    per_ranked: dict[str, list[dict]] = {}
    for s in SCENARIO_ORDER:
        rows = by_scenario.get(s, [])
        if not rows:
            continue
        parts.append(render_scenario_ttft(rows, s))
        per_ranked[s] = sorted(
            [r for r in rows if r.get("ok")],
            key=lambda r: _f(r, "ttft_p50"),
        )

    parts.append(render_overall(per_ranked))

    # Recommendations
    parts.append("## Recommendations\n")
    for s in SCENARIO_ORDER:
        rows = per_ranked.get(s, [])
        if not rows:
            continue
        top = rows[0]
        ttft = _f(top, "ttft_p50")
        hr = _hit_rate(top) * 100
        parts.append(
            f"- **{s}**: best TTFT P50 = `{top['policy']}` "
            f"({ttft:.3f}s, hit_rate={hr:.2f}%)."
        )

    report = "\n".join(parts)
    print(report)
    if args.out:
        Path(args.out).write_text(report)


if __name__ == "__main__":
    main()
