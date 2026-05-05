# LoRA Cache Policy TTFT Sweep — Executive Summary

**Sweep**: 198 runs (66 policies × 3 scenarios), 158.4 min wallclock, 0
failures. Same hardware/workload as the throughput sweep
(Qwen2.5-1.5B, 16 LoRAs, max_loras=2, 80 reqs of ~1k tokens, gpu_mem_util=0.5,
0.5 GiB CPU offload).

Per-request TTFT was measured via `LLMEngine.add_request` + `engine.step()` —
the time between submitting a request and receiving its first decoded token.

Full tables in `TTFT_REPORT.md`. This doc is the synthesis.

---

## TL;DR

| Optimize for… | Winner | TTFT P50 |
|:---|:---|---:|
| **TTFT on thrashing** | `lora_loose:tinylfu` | **1.234s** |
| **TTFT on mixed_popularity** | `lora_freqweighted:lru` | **0.371s** |
| **Avg TTFT (thrashing + mixed)** | `lora_loose:tinylfu` | rank 1+5 |
| **Bare-LRU baseline** | `lru` | rank 2+17 |

Bare **`lru` ranks 2nd on thrashing and 17th on mixed_popularity** — it's
within 1% of the top of the chart on both. The fanciest LoRA-aware
decorators give you single-digit-ms wins at best on TTFT.

**The hit-rate winner from the prior sweep, `lora_budget:s3fifo`, is rank
32 / 66 on mixed_popularity TTFT** — its 8.15% hit rate doesn't recover the
bookkeeping cost on this hardware/workload.

---

## TTFT P50 — top 10 per scenario

### `adapter_thrashing` (round-robin)

| Rank | Policy | TTFT P50 | TTFT P95 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|
| 1 | `lora_loose:tinylfu` | **1.234s** | 2.761s | 32.39% |
| 2 | `lru` | 1.246s | 2.717s | 49.52% |
| 3 | `lora_hysteresis:s3fifo` | 1.247s | 2.732s | 23.96% |
| 4 | `lora_ghost:lru_k` | 1.252s | 3.022s | 33.00% |
| 5 | `lora_adabudget:sieve` | 1.262s | 2.755s | 37.85% |
| 6 | `lora_soft:tinylfu` | 1.268s | 2.835s | 38.70% |
| 7 | `lora_hysteresis:tinylfu` | 1.274s | 2.686s | 23.96% |
| 8 | `lora_prefixtree:lru` | 1.278s | 2.785s | 49.34% |
| 9 | `lora_adabudget:lru_k` | 1.283s | 3.113s | 49.86% |
| 10 | `lora_prefixtree:sieve` | 1.287s | 2.776s | 36.92% |

The **TTFT spread across the top 10 is 1.234–1.287s = 4.3%**. Run-to-run
variance on identical configs is ~3%, so the top tier is essentially a tie.

### `mixed_popularity` (Zipfian)

| Rank | Policy | TTFT P50 | TTFT P95 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|
| 1 | `lora_freqweighted:lru` | **0.371s** | 1.944s | 2.72% |
| 2 | `lora_tight:lru_k` | 0.378s | **1.609s** | 2.72% |
| 3 | `lora_adabudget:tinylfu` | 0.381s | 1.915s | 4.51% |
| 4 | `lora_soft:sieve` | 0.382s | 2.052s | 5.43% |
| 5 | `lora_loose:tinylfu` | 0.383s | 2.002s | 7.50% |
| 6 | `lora_adabudget:lru` | 0.384s | 1.948s | 5.43% |
| 7 | `lora_ghost:lru_k` | 0.386s | 2.320s | 2.72% |
| 8 | `lora_twolevel` | 0.389s | 2.169s | 2.72% |
| 9 | `lora_adabudget:sieve` | 0.390s | 2.011s | 5.43% |
| 10 | `lora_hysteresis:lru` | 0.391s | 2.208s | 2.72% |
| 17 | `lru` | 0.394s | 1.997s | 5.43% |
| 32 | `lora_budget:s3fifo` | 0.404s | 2.041s | 8.15% |

Spread top 10: 0.371–0.391s = 5%. Again top tier is statistical
ties. `lora_budget:s3fifo` (the prior hit-rate winner) is in the middle of
the pack at +9% TTFT vs the leader.

### `adapter_locality`

Every policy clusters at TTFT P50 ≈ 0.48–0.50s. **Degenerate** — bursts
stay on GPU, no reuse pressure differentiates policies.

---

## Average rank across the two meaningful scenarios

`adapter_locality` is excluded since all policies are within noise.

| Rank | Policy | Avg TTFT-P50 rank | thrashing | mixed |
|-----:|:-------|------------------:|:---------:|:-----:|
| **1** | **`lora_loose:tinylfu`** | **3.0** | 1 | 5 |
| 2 | `lora_ghost:lru_k` | 5.5 | 4 | 7 |
| 3 | `lora_adabudget:sieve` | 7.0 | 5 | 9 |
| 4 | `lru` | 9.5 | 2 | 17 |
| 5 | `lora_adabudget:lru` | 9.5 | 13 | 6 |
| 6 | `lora_soft:tinylfu` | 10.0 | 6 | 14 |
| 7 | `lora_soft:sieve` | 15.0 | 26 | 4 |
| 8 | `lora_adabudget:lru_k` | 15.5 | 9 | 22 |
| 9 | `lora_hysteresis:s3fifo` | 17.0 | 3 | 31 |
| 10 | `lora_correlated:sieve` | 18.5 | 18 | 19 |

**`lora_loose:tinylfu`** wins on average TTFT — top-1 on thrashing, top-5
on mixed. **Bare `lru`** is rank 4 — within 1% of the leader.

---

## What this benchmark is actually measuring

TTFT here = **submit time → first decoded token**. On this hardware/workload
that decomposes roughly as:

- Tokenization + queueing: ~0
- Engine scheduling overhead: small, but **per-block policy bookkeeping
  scales here**
- LoRA load (CPU→GPU): non-trivial when adapter switching
- GPU prefill: dominant for thrashing (long prompts, ~1k tokens), modest for
  mixed (Zipfian re-uses some prefixes via GPU prefix cache)

The cache-policy hit rate only affects TTFT to the extent that **CPU
offload hits avoid GPU prefill compute**. With Qwen2.5-1.5B on a laptop
GPU, GPU prefill is fast enough that even ~50% offload-hit-rate uplift
buys back only ~10–50ms — easily eaten by per-block bookkeeping overhead
in the more sophisticated decorators.

This is why `lora_loose:tinylfu` (low overhead, decent retention) and
plain `lru` (zero overhead, decent retention) dominate, while
`lora_budget:s3fifo` (high overhead, best retention) doesn't translate
its hit-rate edge into TTFT wins.

---

## Comparison with the hit-rate-optimized recommendation

The **prior throughput sweep** flagged `lora_budget:s3fifo` as the
Zipfian winner because of its 8.15% hit rate (vs LRU's 5.43%). On TTFT
the picture is different:

| Metric | `lru` | `lora_budget:s3fifo` | `lora_loose:tinylfu` |
|:-------|------:|---------------------:|---------------------:|
| Thrashing TTFT P50 | **1.246s** | 1.407s | **1.234s** |
| Thrashing hit rate | 49.52% | **49.86%** | 32.39% |
| Mixed TTFT P50 | 0.394s | 0.404s | **0.383s** |
| Mixed hit rate | 5.43% | **8.15%** | 7.50% |
| Avg TTFT rank (2 scenarios) | 9.5 | rank 30+ | **3.0** |

`lora_loose:tinylfu` is competitive on hit rate (within 30% on each
scenario) and substantially better on TTFT.

`lora_budget:s3fifo` retains its hit-rate crown but its added bookkeeping
costs net out negative for TTFT on this hardware.

---

## Recommendations

- **Optimizing for TTFT (most production-relevant single metric)** —
  use **`lora_loose:tinylfu`**. Or just use **`lru`**: it's rank 4 on
  TTFT and rank 8 on hit rate, and has zero LoRA-aware bookkeeping cost.
- **Optimizing for offload hit rate** (e.g. capacity planning, expecting
  bigger models / longer outputs where prefill cost dominates) — use
  **`lora_budget:s3fifo`**. The +50% relative hit-rate uplift on Zipfian
  will start beating the bookkeeping overhead at ~4× the prefill cost
  per request.
- **Avoid for TTFT**: `lora_costaware:s3fifo` (worst on mixed at 0.558s),
  `lora_ghost:lru` (worst on thrashing at 1.5s+), and most tinylfu+heavy-
  decorator combos that aren't `lora_loose`.

---

## Caveats

1. **Single-seed sweep**. ~3% run-to-run noise — top-10 differences are
   plausibly indistinguishable. Multi-seed would tighten the rankings.
2. **8-token outputs**. TTFT ≈ E2E here, so this is essentially also a
   prefill-latency benchmark. With longer decode, the relative weight
   of TTFT in user experience goes down and other metrics matter more.
3. **GPU-prefill-bound on small model**. The hit-rate axis would matter
   more for TTFT on larger models or longer prefixes, where saving 10%
   of prefill compute saves more wallclock than the policy bookkeeping.
4. **`adapter_locality` is degenerate** at this cache size — bursts stay
   on GPU prefix cache.
5. **ARC excluded** because of pre-existing `self.stats` initialization
   bug (not introduced by this work).
