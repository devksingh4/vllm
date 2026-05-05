# LoRA-Aware KV Cache Policy Benchmarks

Tools for evaluating CPU-offload KV cache eviction policies under LoRA
multi-adapter serving workloads. Combines a pure-Python simulator (fast,
no GPU) with three real-vLLM benchmarks (end-to-end, throughput, and
TTFT), plus sweep drivers and ranking utilities.

---

## What's in this directory

| Script | Purpose | GPU? | Typical runtime |
|:---|:---|:---:|:---|
| `benchmark_lora_policy.py` | Pure-Python policy simulator. Generates synthetic block-access traces and measures hit rate / reload count / composite cost without booting vLLM. | No | seconds |
| `benchmark_lora_e2e.py` | End-to-end vLLM benchmark: real `LLM.generate` over a configurable workload. Reports throughput, P50/P95/P99 batch latency, and (via the policy stats logged by `CPUOffloadingManager`) hit rate + eviction counts. | Yes | 30-60s per run |
| `benchmark_lora_ttft.py` | Per-request TTFT (Time-To-First-Token) benchmark. Uses `LLMEngine.add_request` + `engine.step()` to time each request from submission to first decoded token. | Yes | 30-60s per run |
| `sweep_lora_policies.py` | Driver that runs `benchmark_lora_e2e.py` for every (policy, scenario) combination, captures metrics into `results.json`. Resumable. | Yes | ~3h for full sweep |
| `sweep_lora_ttft.py` | Driver that runs `benchmark_lora_ttft.py` for every (policy, scenario). Resumable. | Yes | ~3h for full sweep |
| `rank_lora_policies.py` | Reads the `results.json` from `sweep_lora_policies.py` and emits a markdown report (per-scenario tables + LoRA-coupling uplift matrix + overall ranking). | No | seconds |
| `rank_lora_ttft.py` | Same idea for the TTFT sweep, focused on TTFT P50/P95 ranking. | No | seconds |

---

## Workload scenarios

All four real-vLLM benchmarks share the same workload generators. Each
produces a sequence of `(prompt, lora_request)` pairs over a fixed pool
of distinct LoRA adapters. The adapter access pattern is what makes a
scenario "easy" or "hard" for a given policy.

| Scenario | Pattern | What it stresses |
|:---|:---|:---|
| `adapter_thrashing` | Round-robin: req `i` uses adapter `i mod N`. | Highest CPU-offload reuse pressure. Each adapter cycles back after `N-1` other adapters' blocks have been inserted, so eviction quality directly determines whether old prefix blocks are still resident. |
| `adapter_locality` | Bursts of `burst_len` consecutive requests on the same adapter, then jump. | Per-adapter prefix blocks tend to stay on GPU prefix cache; CPU offload sees little reuse pressure. Mostly degenerate at small adapter counts. |
| `mixed_popularity` | Zipfian (alpha=1.2) over `N` adapters: a small head dominates, long tail of cold adapters. | The realistic multi-tenant LoRA-serving pattern. Tests the interaction of adapter popularity with block-level recency. |

The `benchmark_lora_policy.py` simulator additionally implements a
`sticky_ephemeral` scenario (sticky adapters interleaved with one-shot
ephemerals) — useful for unit-testing policies in isolation but not part
of the standard real-vLLM sweep.

---

## How the policy system is structured

The CPU-offload cache uses a pluggable `CachePolicy` abstraction from
`vllm/v1/kv_offload/cpu/policies/`. Two layers compose:

### Inner (base) policy — block-level eviction algorithm

Implements the actual "which block to evict next" decision. Inner
policies don't know about LoRA adapters; they see only opaque
`BlockHash` values.

| Name | Algorithm |
|:---|:---|
| `lru` | Least-recently-used. Single ordered map; touch moves to MRU end. |
| `sieve` | Scan-resistant LRU variant. New blocks must be touched at least twice to survive eviction (defends against one-shot scans). |
| `s3fifo` | Static segmented FIFO. Small admission queue + main queue + ghost list of recently-evicted blocks. Outperforms LRU on Zipfian workloads. |
| `tinylfu` | Frequency-weighted with admission filter (count-min sketch). Strong scan resistance, good on highly skewed access patterns. |
| `lru_k` | LRU keyed off the K-th most recent reference instead of the most recent. Distinguishes one-time hits from sustained reuse. |
| `arc` | Adaptive replacement cache (T1/T2 + B1/B2 ghost lists). Self-tunes recency vs frequency. *Currently broken on master — missing `self.stats` initialisation.* |
| `lora_twolevel` | Standalone LoRA-aware policy (not a decorator): LRU of adapters, each containing an LRU of its own blocks. Evicts the LRU block of the LRU adapter. |

### Outer (LoRA-coupling) decorator — adapter-aware overlay

Wraps an inner policy and injects LoRA-adapter signal into eviction
decisions. Outer policies subscribe to two hooks defined on
`OffloadingManager`:

- `register_block_adapters(mapping: dict[BlockHash, str | None])` — called
  by the scheduler when blocks are inserted, telling the policy which
  LoRA adapter produced each block. The default implementation is a
  no-op, so plain inner policies silently ignore it.
- `update_live_adapters(adapters: set[str])` — called by the scheduler
  on every meta-build, listing which LoRA adapters currently have
  in-flight requests (a proxy for "currently on GPU"). LoRA-aware
  policies use this to bias eviction toward blocks belonging to
  adapters that have left GPU residency.

The decorator's eviction strategy differs by family:

| Coupling | Strategy |
|:---|:---|
| `lora_loose` | Soft preference. First call `inner.evict(n, protected=protected ∪ live_adapter_blocks)`; if that returns `None` (no candidates), fall back to plain `inner.evict(n, protected)`. |
| `lora_tight` | Atomic group eviction. Find non-live adapters by ascending block count; remove all blocks for the smallest group(s) atomically via `inner.remove(block_hash)` until `n` collected. Falls back to `inner.evict()` if no full group fits. |
| `lora_hysteresis` | Tight coupling, but a recently-departed adapter must stay non-live for `N` consecutive `update_live_adapters()` calls before its blocks become eligible. Defends against flapping. |
| `lora_soft` | On the live→non-live transition, lightly *touches* the inner policy with non-live blocks (boosting their evict-priority via the inner policy's normal eviction path); never rewrites eviction decisions itself. |
| `lora_freqweighted` | Maintains a decaying per-adapter frequency score. On evict, group-evicts blocks belonging to the lowest-heat adapter first. |
| `lora_correlated` | Every block access lightly touches a sample of the same adapter's other cached blocks, encouraging block groups to age together. Continuous-signal sibling of soft-boost. |
| `lora_budget` | Soft per-adapter cache-capacity reservation (`capacity / num_adapters` slots each). When an adapter exceeds its share, its own oldest blocks are evicted before reaching for blocks of other adapters. |
| `lora_adabudget` | Same idea as `lora_budget`, but adjusts each adapter's share dynamically based on observed hit rate / re-reference patterns. |
| `lora_costaware` | Explicit cost model: `expected_saved_work = hit_rate × prefill_cost − reload_cost`. Evicts blocks with the smallest expected saved work. |
| `lora_ghost` | Tight coupling with ARC-style ghost lists per adapter. When a re-arriving adapter has blocks in its ghost list, those blocks get a recency boost on re-insertion. |
| `lora_position` | Biases eviction by position within the prompt (early/shared prefix blocks survive longer than tail blocks). |
| `lora_prefixtree` | Maintains a per-adapter trie of cached block hashes; uses tree depth + branching factor to decide which blocks are most "prefix-like" and worth retaining. |

### Naming convention

Concrete policies are named `outer:inner` in the
`VLLM_KV_OFFLOAD_POLICY` environment variable and on the benchmark
`--policy` flag. For example:

```bash
# Plain LRU (no outer decorator)
VLLM_KV_OFFLOAD_POLICY=lru

# Per-adapter cache budget wrapped around s3fifo
VLLM_KV_OFFLOAD_POLICY=lora_budget:s3fifo

# Frequency-weighted adapter scoring wrapped around tinylfu
VLLM_KV_OFFLOAD_POLICY=lora_freqweighted:tinylfu
```

The standalone `lora_twolevel` is *not* a decorator — it has no inner —
so it appears bare without a colon.

When a plain inner policy is used, the `register_block_adapters` and
`update_live_adapters` calls become no-ops; the inner policy is
unaware of LoRA at all and behaves exactly as it would for a
non-LoRA workload.

---

## Quickstart

### Single benchmark run

```bash
.venv/bin/python benchmarks/benchmark_lora_e2e.py \
  --model Qwen/Qwen2.5-1.5B \
  --lora-path kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter \
  --num-adapters 16 --max-loras 2 --max-cpu-loras 16 \
  --scenario adapter_thrashing \
  --num-requests 80 --batch-size 8 \
  --prefix-words 800 --suffix-words 200 \
  --max-tokens 8 --max-model-len 2048 \
  --gpu-memory-utilization 0.5 \
  --kv-offloading-size 0.5 \
  --policy lora_budget:s3fifo
```

The throughput, latency percentiles, and (logged on process exit by
`CPUOffloadingManager`) the policy's eviction counts and CPU-offload hit
rate are written to stderr.

### TTFT measurement (single run)

```bash
.venv/bin/python benchmarks/benchmark_lora_ttft.py \
  --policy lora_loose:tinylfu --scenario adapter_thrashing \
  --num-requests 80 --batch-size 8 \
  --num-adapters 16 --max-loras 2 \
  --prefix-words 800 --suffix-words 200 \
  --kv-offloading-size 0.5 --gpu-memory-utilization 0.5
```

Reports per-request TTFT P50/P95/P99 and E2E latency.

### Full sweep (~3 hours each)

```bash
# Throughput / hit-rate sweep
.venv/bin/python benchmarks/sweep_lora_policies.py --timeout-s 300
.venv/bin/python benchmarks/rank_lora_policies.py \
  --results /tmp/lora_sweep_logs/results.json \
  --out /tmp/lora_sweep_logs/REPORT.md

# TTFT sweep
.venv/bin/python benchmarks/sweep_lora_ttft.py --timeout-s 300
.venv/bin/python benchmarks/rank_lora_ttft.py \
  --results /tmp/lora_sweep_logs/ttft_sweep/results.json \
  --out /tmp/lora_sweep_logs/TTFT_REPORT.md
```

The sweep drivers are resumable: re-running picks up from any partial
`results.json` and only runs the remaining (policy, scenario) pairs.
Per-run logs are written next to `results.json` named
`{policy}__{scenario}.log`.

### Pure-Python simulator (no GPU)

```bash
.venv/bin/python benchmarks/benchmark_lora_policy.py \
  --scenario mixed_popularity --cache-capacity 200 --gpu-lora-slots 2
```

Useful for quick policy iteration without paying vLLM startup cost.
Produces hit rate, adapter reload count, and composite cost
(`cache_misses × miss_cost + adapter_reloads × reload_cost`) per
policy.

---

## Important caveats

1. **GPU prefix caching catches reuse first.** When a block is still
   resident on GPU, vLLM never queries the CPU offload manager. CPU
   offload only earns hits when GPU evicts a block AND the block is
   re-requested before CPU evicts it too. Workloads must therefore
   pressure both tiers — the sweep config does this with 16 adapters at
   `gpu_memory_utilization=0.5`, but smaller / less aggressive configs
   will report a flat 0% hit rate everywhere.
2. **Single-seed, no warmup discount.** Each run uses a fresh model
   load; the first batch absorbs LoRA download + JIT compilation cost.
   Run-to-run variance on identical configs is around 3% on throughput
   and ~5% on TTFT P50 — treat differences smaller than that as noise.
3. **`adapter_locality` is degenerate at small adapter counts.** With
   only `N=16` adapters and burst lengths of 8, each burst's prefix
   blocks fit on GPU and never reach CPU offload. Hit rate is uniformly
   0% in this scenario, so it can't differentiate policies.
4. **ARC is currently broken on master** (missing `self.stats`
   initialisation in `__init__`). The sweep driver excludes it; the
   simulator does not exercise it.
5. **TTFT vs hit-rate optimisation can disagree.** A policy with higher
   bookkeeping cost and higher hit rate can lose on TTFT because the
   per-block bookkeeping happens on every insert/touch, while the
   hit-rate uplift only kicks in occasionally. See `TTFT_EXEC_SUMMARY.md`
   for an example where `lora_budget:s3fifo` (best hit rate on Zipfian)
   loses TTFT to bare `lru` (zero bookkeeping, slightly worse hit rate).

---

## Where to read more

- `vllm/v1/kv_offload/cpu/policies/abstract.py` — `CachePolicy` ABC and
  `PolicyStats` dataclass.
- `vllm/v1/kv_offload/cpu/policies/lru.py` — simplest reference
  implementation; useful for understanding what an inner policy must
  provide.
- `vllm/v1/kv_offload/cpu/policies/lora_aware.py` — all 11 outer
  decorators + the `lora_twolevel` standalone, plus the shared
  `_LoRABookkeeping` mixin.
- `vllm/v1/kv_offload/cpu/manager.py` — `CPUOffloadingManager`: parses
  the `outer:inner` policy name, wires up the decorator, dispatches
  the `register_block_adapters` and `update_live_adapters` hooks,
  and emits the policy-stats line on shutdown.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` —
  the scheduler-side wiring: where each block-store learns its
  adapter, and where `live_adapters` is computed each step from
  in-flight requests.
