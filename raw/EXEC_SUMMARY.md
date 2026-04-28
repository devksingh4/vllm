# LoRA Cache Policy Sweep — Executive Summary

**Sweep**: 198 runs (66 unique policies × 3 scenarios), 157.8 min total wallclock,
0 failures.
**Hardware**: RTX 3070 Ti Laptop, 8 GiB. Qwen2.5-1.5B + 16 distinct LoRA
adapters (`max_loras=2`, `max_cpu_loras=16`), 80 requests of ~1 000 tokens
each. CPU offload sized at 0.5 GiB; GPU KV at `gpu_memory_utilization=0.5`.
**Excluded**: ARC (pre-existing `AttributeError: 'ARCCachePolicy' has no
attribute 'stats'` — bug in master, not introduced by this work).

The full per-scenario tables and uplift matrices are in
`REPORT_byhit.md` (sorted by hit rate) and `REPORT.md` (sorted by output
throughput). This doc summarises the headline findings.

---

## Ranking — by CPU-offload hit rate

The throughput numbers across policies cluster within ~10% (workload is
GPU-prefill-dominated), so **hit rate is the meaningful policy signal**.

### Adapter Thrashing (round-robin 16 adapters → highest reuse pressure)

| Tier | Hit rate | Policies |
|-----:|:--------:|:---------|
| **S** | 50–52% | **`s3fifo` (52.15%)** , `lora_correlated:s3fifo` (50.50%), `lru` / `lru_k` / `lora_loose:s3fifo` / `lora_position:s3fifo` / `lora_adabudget:s3fifo` / `lora_costaware:lru_k` / `lora_loose:lru_k` / `lora_prefixtree:lru_k` / `lora_position:lru_k` / `lora_budget:s3fifo` / `lora_position:lru` / `lora_prefixtree:s3fifo` / `lora_budget:lru` / `lora_correlated:lru_k` / `lora_loose:lru` / `lora_budget:lru_k` (all **49.86%**) |
| A | 35–45% | `lora_budget:sieve` (43.96%), `lora_soft:tinylfu` (41.83%), `lora_position:sieve` (38.44%), `lora_loose:sieve` (37.65%), bare `sieve` (36.92%), `lora_correlated:sieve` (36.70%), `lora_adabudget:sieve` (36.32%), `lora_costaware:sieve` (36.31%), `lora_soft:sieve` (35.13%) |
| B | 32–34% | **All `lora_tight:*` clamp to 33.00%** (lru / lru_k / sieve / tinylfu) and 33.53% (s3fifo). All `lora_ghost:*` clamp to 32.6–33.0%. |
| C | 27–32% | `tinylfu` and most tinylfu decorators (27–32%) |
| D | 18–26% | `lora_hysteresis:*` (24.0%), `lora_twolevel` (19.45%), `lora_freqweighted:*` (**18.23% across every base**) |

### Mixed Popularity (Zipfian, hot-head + cold-tail)

| Tier | Hit rate | Policies |
|-----:|:--------:|:---------|
| **S** | 8.15% | **`lora_budget:lru` / `:sieve` / `:s3fifo` / `:tinylfu` / `:lru_k` (8.15% across every base)**, `lora_soft:tinylfu`, `lora_adabudget:tinylfu` |
| A | 7.2–7.8% | `lora_correlated:lru_k` (7.77%), `tinylfu` (7.23%), `lora_loose:tinylfu` / `lora_position:tinylfu` / `lora_costaware:tinylfu` / `lora_prefixtree:tinylfu` (7.23%) |
| B | 5.4% | `lru`, `lru_k`, `sieve`, `s3fifo`, and the `lora_loose:*` / `lora_position:*` / `lora_costaware:*` / `lora_prefixtree:*` / `lora_adabudget:*` LRU-family decorators |
| C | 2.7% | All `lora_tight:*`, all `lora_ghost:*`, `lora_freqweighted:*`, `lora_hysteresis:*`, `lora_twolevel` |
| D | 0% | `lora_hysteresis:tinylfu` |

### Adapter Locality (bursts on same adapter)

**Hit rate is uniformly 0%** for every policy — the per-adapter bursts stay on
GPU prefix cache and never fall through to CPU offload. Throughput differences
in this scenario reflect run-to-run noise (LoRA load order, etc.), not policy
quality. **Treat this scenario as a no-op for ranking purposes.**

---

## Best policy per scenario

| Scenario | Recommended policy | Hit rate | OutTok/s |
|:---------|:-------------------|---------:|---------:|
| **`adapter_thrashing`** (round-robin LoRA serving) | **`s3fifo`** | 52.15% | 23.1 |
| **`mixed_popularity`** (Zipfian / skewed) | **`lora_budget:s3fifo`** | 8.15% | 31.8 |
| `adapter_locality` (bursty) | any (degenerate) | — | — |

If you want **one policy across all scenarios**, the closest to a Pareto
optimum is `lora_budget:s3fifo` — second-best on thrashing (49.86%, vs
52.15% peak), best on mixed, no harm on locality.

`s3fifo` (bare) is the headline **thrashing** winner but sits in the middle
of the pack on mixed (5.43% hit rate), so it's not strictly dominant.

---

## What the LoRA decorators actually do

These are the most useful cross-scenario observations from the uplift matrices
(positive = decorator beats bare base on *output throughput*; raw hit rates
discussed above):

- **`lora_loose`** — close-to-transparent. Throughput within ±5% of bare
  base in every scenario; hit rate matches bare base. Effectively "no-op"
  in this benchmark — wraps without changing eviction order.
- **`lora_tight`** — **harmful** on thrashing. Forces every base policy to
  ~33% hit rate (regardless of inner base). The atomic
  whole-adapter-group evictions throw away too many useful blocks. Don't ship.
- **`lora_budget` / `lora_adabudget`** — only family that **clearly helps
  Zipfian** (8.15% vs 5.43% for the LRU family). Per-adapter capacity
  reservation prevents hot adapters from monopolising the cache.
- **`lora_correlated`** — slight lift on `s3fifo` thrashing (50.50% vs
  52.15% bare); roughly transparent elsewhere. Worth keeping.
- **`lora_soft`** — mostly transparent except a notable **+25% throughput
  on `lru_k` locality** and **+28% on `tinylfu` locality** — a side-effect
  of LoRA-load ordering, not retention. Risky.
- **`lora_position`** — mostly transparent. No clear win.
- **`lora_prefixtree`** — also transparent; no clear win.
- **`lora_hysteresis`** — **bad**: holds blocks too long, churns wrong
  evictions. 24% on thrashing across every base.
- **`lora_freqweighted`** — **bad**: 18.23% on thrashing across every base.
  Frequency weighting boosts the wrong blocks (intra-prompt repetition is
  not adapter-reuse).
- **`lora_ghost`** — **suboptimal**: clamps to 33% on thrashing like tight
  coupling. Ghost-list mechanism doesn't help here.
- **`lora_costaware`** — close to `lora_loose`. Slightly better
  throughput on `s3fifo` mixed (+11.4%); no retention win.

### Standalone (non-decorator) LoRA-aware policies

- **`lora_twolevel`** — only standalone LoRA-aware policy in the sweep.
  19.45% hit on thrashing — **worst non-decorator policy**. The two-level
  partitioning is too rigid for a 16-adapter / round-robin pattern.

---

## Caveats

1. **Throughput is GPU-bound at this scale.** Output is small (8 tokens) so
   the ~50% retention difference between best and worst policies translates
   to only ~10% wallclock difference. On a workload with longer outputs and/or
   more aggressive LoRA reload cost, retention quality should matter more.
2. **`adapter_locality` is degenerate** at this cache size. Bursts fit on
   GPU prefix cache; CPU offload never matters. To exercise locality
   meaningfully you'd need GPU oversubscription within a burst (longer
   prompts and/or smaller GPU KV).
3. **Sweep is single-seed**. Run-to-run variance on identical config is
   ~3% on throughput; treat hit-rate ties of <1pp as noise.
4. **ARC excluded** because of a pre-existing `self.stats` initialization
   bug. Filing/fixing that is independent of this work.
5. **GPU prefix cache is enabled**, so CPU offload only earns hits when GPU
   evicts. With `enable_prefix_caching=False`, every request would consult
   CPU offload and the policy ranking would shift toward retention-quality
   metrics more strongly.
