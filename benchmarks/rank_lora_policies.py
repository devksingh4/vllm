# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reads results.json produced by sweep_lora_policies.py and emits a markdown
ranking report (printed to stdout, optionally written to disk)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SCENARIO_ORDER = ["adapter_thrashing", "adapter_locality", "mixed_popularity"]
SCENARIO_BLURB = {
    "adapter_thrashing": (
        "Round-robin across all adapters: every request hits a different "
        "adapter than the previous, so blocks rarely repeat. Tests how well "
        "the policy avoids holding dead-adapter blocks."
    ),
    "adapter_locality": (
        "Bursts of consecutive requests on the same adapter. Tests whether "
        "the policy preserves grouped per-adapter blocks while pruning "
        "older adapter groups."
    ),
    "mixed_popularity": (
        "Zipfian adapter mix: a few hot adapters dominate. Tests "
        "interaction of adapter popularity with block recency."
    ),
}


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def is_lora_decorated(policy: str) -> bool:
    return ":" in policy


def base_of(policy: str) -> str:
    return policy.split(":", 1)[1] if ":" in policy else policy


def coupling_of(policy: str) -> str | None:
    return policy.split(":", 1)[0] if ":" in policy else None


def _safe_float(d: dict, key: str, default: float = float("nan")) -> float:
    v = d.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _hit_rate_value(r: dict) -> float:
    """Return cpu_offload_hit_rate as a fraction in [0,1], or NaN if absent."""
    hb = r.get("hit_blocks")
    lb = r.get("lookup_blocks")
    if hb is None or not lb:
        # Fall back to parsing percentage string like "12.34%"
        s = r.get("hit_rate_pct")
        if s and s != "n/a":
            try:
                return float(s.rstrip("%")) / 100.0
            except ValueError:
                return float("nan")
        return float("nan")
    return hb / lb


def fmt_thr(r: dict) -> str:
    v = r.get("out_throughput")
    return f"{v:6.1f}" if isinstance(v, (int, float)) else "   n/a"


def fmt_total_thr(r: dict) -> str:
    v = r.get("throughput")
    return f"{v:6.1f}" if isinstance(v, (int, float)) else "   n/a"


def fmt_hit(r: dict) -> str:
    hr = _hit_rate_value(r)
    return f"{hr*100:5.2f}%" if hr == hr else "   n/a"  # NaN check


def fmt_evict(r: dict) -> str:
    e = r.get("evict_blocks")
    return f"{e:5d}" if isinstance(e, int) else "   - "


def fmt_p95(r: dict) -> str:
    v = r.get("p95_s")
    return f"{v:5.2f}" if isinstance(v, (int, float)) else "   - "


def rank_scenario(
    rows: list[dict], primary: str = "hit_rate"
) -> list[dict]:
    """Sort rows so highest primary metric is first.

    `primary` may be 'hit_rate' or any numeric field on the result rows.
    Failed runs are dropped from the ranking.
    """
    ok = [r for r in rows if r.get("ok")]
    if primary == "hit_rate":
        return sorted(
            ok,
            key=lambda r: (
                -_hit_rate_value(r) if _hit_rate_value(r) == _hit_rate_value(r) else 1.0,
                -_safe_float(r, "out_throughput"),
            ),
        )
    return sorted(ok, key=lambda r: -_safe_float(r, primary))


def render_scenario_table(rows: list[dict], scenario: str) -> str:
    lines = []
    lines.append(f"## Scenario: `{scenario}`")
    lines.append("")
    lines.append(f"_{SCENARIO_BLURB.get(scenario, '')}_")
    lines.append("")
    lines.append(
        "| Rank | Policy | OutTok/s | TotTok/s | HitRate | Evicted | P95 (s) |"
    )
    lines.append(
        "|-----:|:-------|---------:|---------:|--------:|--------:|--------:|"
    )
    for i, r in enumerate(rank_scenario(rows), 1):
        lines.append(
            f"| {i} | `{r['policy']}` | {fmt_thr(r)} | {fmt_total_thr(r)} "
            f"| {fmt_hit(r)} | {fmt_evict(r)} | {fmt_p95(r)} |"
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


def render_lora_uplift(rows: list[dict], scenario: str) -> str:
    """Per-base table comparing each LoRA-coupling vs the bare base."""
    lines = []
    lines.append(f"### LoRA coupling uplift over bare base — `{scenario}`")
    lines.append("")
    lines.append(
        "Positive `Δ_thr` = LoRA decorator beats the bare inner base on "
        "output throughput. `n/a` means base run was missing/failed."
    )
    lines.append("")

    by_policy: dict[str, dict] = {r["policy"]: r for r in rows if r.get("ok")}
    bases = sorted({base_of(p) for p in by_policy if is_lora_decorated(p)})
    couplings = sorted({coupling_of(p) for p in by_policy if is_lora_decorated(p)})
    couplings = [c for c in couplings if c is not None]

    header = "| Coupling \\ Base | " + " | ".join(bases) + " |"
    sep = "|" + "---|" * (len(bases) + 1)
    lines.append(header)
    lines.append(sep)

    for coup in couplings:
        cells = [f"`{coup}`"]
        for b in bases:
            decorated = by_policy.get(f"{coup}:{b}")
            base = by_policy.get(b)
            if decorated is None or base is None:
                cells.append("n/a")
                continue
            d = _safe_float(decorated, "out_throughput")
            bt = _safe_float(base, "out_throughput")
            if bt != bt or d != d or bt == 0:
                cells.append("n/a")
                continue
            delta = d - bt
            pct = (delta / bt) * 100
            cells.append(f"{delta:+.1f} ({pct:+.1f}%)")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def render_overall(per_scenario_rankings: dict[str, list[dict]]) -> str:
    """Composite ranking using avg rank across scenarios."""
    rank_sum: dict[str, list[int]] = defaultdict(list)
    for rows in per_scenario_rankings.values():
        for i, r in enumerate(rows, 1):
            rank_sum[r["policy"]].append(i)

    composite = []
    for policy, ranks in rank_sum.items():
        if len(ranks) != len(per_scenario_rankings):
            continue  # only consider policies that ran in every scenario
        composite.append((policy, sum(ranks) / len(ranks), ranks))
    composite.sort(key=lambda t: t[1])

    lines = []
    lines.append("## Overall ranking (avg rank by output throughput)")
    lines.append("")
    lines.append("| Rank | Policy | Avg Rank | " + " | ".join(
        [f"{s} rank" for s in SCENARIO_ORDER]
    ) + " |")
    lines.append("|-----:|:-------|---------:|" + ":---:|" * len(SCENARIO_ORDER))
    for i, (policy, avg, ranks) in enumerate(composite, 1):
        per_scn = []
        # Map back per scenario in order
        for s in SCENARIO_ORDER:
            srows = per_scenario_rankings.get(s, [])
            r = next(
                (idx for idx, row in enumerate(srows, 1) if row["policy"] == policy),
                None,
            )
            per_scn.append(str(r) if r is not None else "—")
        lines.append(
            f"| {i} | `{policy}` | {avg:.1f} | " + " | ".join(per_scn) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_recommendations(per_scenario_rankings: dict[str, list[dict]]) -> str:
    lines = ["## Recommendations", ""]
    for s in SCENARIO_ORDER:
        rows = per_scenario_rankings.get(s)
        if not rows:
            lines.append(f"- **{s}**: no completed runs.")
            continue
        top = rows[0]
        thr = top.get("out_throughput")
        hr = _hit_rate_value(top) * 100
        lines.append(
            f"- **{s}**: best is `{top['policy']}` "
            f"(out_thr={thr:.1f} tok/s"
            + (f", hit_rate={hr:.2f}%" if hr == hr else "")
            + ")."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        default="/tmp/lora_sweep_logs/results.json",
    )
    p.add_argument("--out", default=None, help="Optional path to write report")
    args = p.parse_args()

    results = load_results(Path(args.results))
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)

    parts = ["# LoRA-Aware Cache Policy Sweep — Results", ""]
    parts.append(
        f"Total runs in results: **{len(results)}** "
        f"({sum(1 for r in results if r.get('ok'))} ok, "
        f"{sum(1 for r in results if r.get('timed_out'))} timed out, "
        f"{sum(1 for r in results if not r.get('ok'))} failed)."
    )
    parts.append("")

    per_scenario_rankings: dict[str, list[dict]] = {}
    for scenario in SCENARIO_ORDER:
        rows = by_scenario.get(scenario, [])
        if not rows:
            continue
        parts.append(render_scenario_table(rows, scenario))
        parts.append(render_lora_uplift(rows, scenario))
        per_scenario_rankings[scenario] = rank_scenario(rows)

    parts.append(render_overall(per_scenario_rankings))
    parts.append(render_recommendations(per_scenario_rankings))

    report = "\n".join(parts)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\n[wrote {args.out}]", flush=True)


if __name__ == "__main__":
    main()
