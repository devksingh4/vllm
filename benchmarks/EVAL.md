# KV Cache Eviction Policy Evaluation Plan

## Research: how are KV cache algorithms evaluated?

### Two distinct problem classes

The literature splits into two camps that are easy to conflate:

1. **Token-level eviction** (H2O, SnapKV, StreamingLLM, PyramidKV, NACL,
   DefensiveKV). These *permanently drop individual KV entries inside the
   attention computation* to shrink memory. Evaluation centres on **model
   quality degradation**: perplexity on PG-19/WikiText, accuracy on LongBench,
   Needle-in-a-Haystack retrieval, RULER synthetic tasks, and passkey retrieval
   at various context lengths and budget ratios (5%, 10%, 20% of full cache).
   Because the model output *changes*, quality metrics are mandatory.

2. **Block-level / page-level eviction** (our work: SIEVE, S3-FIFO, LRU, ARC
   in vLLM's `BlockPool` and `CPUOffloadingManager`). These evict *entire
   prefix-cached blocks* under memory pressure but recompute them on miss —
   **the model output is bit-identical**. Evaluation centres on **system
   performance**: TTFT, throughput, cache hit rate, and tail latency. Quality
   metrics are irrelevant because a miss just costs a recompute, not a wrong
   answer.

**Our project is firmly in camp 2.** We should not be measuring perplexity or
downstream accuracy — the model output is identical regardless of policy. Our
job is to prove that one eviction policy reduces redundant prefill (= lower
TTFT, higher throughput) better than another under realistic workloads.

### Key metrics (system-level)

| Metric | Why it matters | Measured by |
|--------|---------------|-------------|
| **TTFT** (time-to-first-token) | Directly reflects prefix cache hits — a hit skips prefill | `o.metrics.first_token_latency` in vLLM |
| **Throughput** (tok/s) | Higher hit rate → less wasted compute → more capacity | Total tokens / wall time |
| **Cache hit rate** | The fundamental signal of eviction quality | vLLM's `Prefix cache hit rate` log line |
| **TPOT** (time per output token) | Should stay flat across policies (sanity check) | Per-request decode latency |
| **ITL** (inter-token latency) | Tail latency during decode; contention-sensitive | P95/P99 of per-token gaps |
| **Request latency** | End-to-end per-request | Measured at client |

### Evaluation approaches from the literature

**H2O (NeurIPS 2023):** Sweeps KV budget (5/10/20% of full cache), measures
perplexity on OPT models, accuracy on downstream tasks. Not directly
applicable to us since they change model output.

**SCBench (NeurIPS 2024, Microsoft):** The most relevant benchmark for
*shared-context* KV cache evaluation. Introduces two modes:
- **Multi-turn:** Shared context + follow-up queries in a conversation.
  Tests whether a policy retains the right blocks when attention focus shifts.
- **Multi-request:** Same long context, independent queries from different
  "users." Tests prefix caching directly — the exact scenario we care about.
SCBench is open-source: `microsoft/MInference` repo on GitHub, dataset on
HuggingFace (`microsoft/SCBench`). Tasks include string retrieval, semantic
retrieval, global info extraction, and multi-hop reasoning — all over shared
long contexts. **We should integrate SCBench tasks.**

**vLLM's built-in benchmarks:** The repo already has highly relevant scripts:
- `benchmark_prefix_caching.py` — Fixed or ShareGPT prompts with repeat
  count, measures throughput with prefix caching on/off.
- `benchmark_long_document_qa_throughput.py` — Long document with repeated QA,
  three repeat modes (random/tile/interleave), directly tests cache reuse.
- `benchmark_serving.py` — Full online serving benchmark with Poisson
  arrivals, TTFT/TPOT/ITL reporting, supports ShareGPT/sonnet/synthetic
  datasets, and `--prefix-repetition-*` flags for prefix caching stress tests.
- `benchmark_block_pool.py` — Micro-benchmark of alloc/free latency per
  eviction policy.

**ShareGPT traces:** Real user conversation data from LMSYS Chatbot Arena.
Widely used in LLM serving benchmarks. Natural prefix sharing from system
prompts and multi-turn conversations.

**LMSYS-Chat-1M:** 1M real conversations with prefix-sharing statistics
analyzed by the LMSYS team. Gold standard for realistic workload distributions.

### Open-source tools we should use

| Tool | What it does | How we use it |
|------|-------------|---------------|
| **`vllm bench serve`** | vLLM's built-in serving benchmark CLI | Online benchmark with ShareGPT, Poisson arrivals, TTFT/TPOT/ITL |
| **`vllm bench throughput`** | Offline throughput benchmark | Batch throughput with ShareGPT/sonnet/synthetic data |
| **`benchmark_prefix_caching.py`** | Prefix caching throughput | Vary `--num-prompts` and `--repeat-count` per policy |
| **`benchmark_long_document_qa_throughput.py`** | Long-doc prefix reuse | Test random/tile/interleave repeat modes |
| **`benchmark_block_pool.py`** | Micro-benchmark alloc/free latency | Measure per-policy overhead (LRU vs SIEVE vs S3-FIFO) |
| **GuideLLM** (`vllm-project/guidellm`) | Production benchmarking framework | Recommended by vLLM docs over `vllm bench serve` for production evals; has live progress updates and automatic report generation |
| **SCBench** (`microsoft/MInference`) | Multi-request shared context | Plug in SCBench tasks with our modified vLLM |
| **ShareGPT dataset** | Real conversation traces | Feed to `benchmark_serving.py` |
| **BurstGPT** | Real Azure ChatGPT workload traces | Realistic bursty arrival patterns; supported by `vllm bench serve` natively via `--dataset-name burstgpt` |

### `vllm bench serve` — details

Implemented in `vllm/benchmarks/serve.py` (~1970 lines). The core
`benchmark()` async function sends requests via aiohttp with configurable
arrival patterns and collects per-request TTFT, TPOT, ITL, and E2EL.

Key features relevant to our work (from `docs/benchmarking/cli.md`):

- **Prefix repetition dataset:** Built-in synthetic dataset that generates
  requests sharing N prefixes with unique suffixes — directly tests prefix
  cache eviction:
  ```bash
  vllm bench serve --dataset-name prefix_repetition \
      --prefix-repetition-prefix-len 512 \
      --prefix-repetition-suffix-len 128 \
      --prefix-repetition-num-prefixes 5 \
      --prefix-repetition-output-len 128
  ```

- **Load pattern control:** `--request-rate`, `--burstiness` (Gamma
  distribution), `--max-concurrency`. Burstiness < 1 creates bursty traffic
  that stresses eviction under memory pressure.

- **Ramp-up:** `--ramp-up-strategy linear|exponential` with
  `--ramp-up-start-rps` / `--ramp-up-end-rps` to find the throughput cliff
  where eviction policy matters most.

- **Goodput SLOs:** `--goodput ttft:3000 tpot:100` reports the fraction of
  requests meeting SLA targets — useful for showing that SIEVE maintains SLA
  compliance at higher load than LRU.

- **Timeline visualization:** `--plot-timeline` generates an interactive HTML
  plot showing per-request execution with ITL colour thresholds.

- **Custom datasets:** Any `.jsonl` file with `{"prompt": "..."}` entries.
  We can convert agent traces or SCBench tasks to this format.

- **LoRA benchmarking:** `--lora-modules` with `--lora-assignment
  random|round-robin` — relevant if we pursue multi-LoRA cache coupling.

### Multi-model KV cache partitioning (cache-partition branch)

The `cache-partition` merge adds experimental support for **multi-model
(multi-tenant) KV cache sharing** on a single GPU block pool. This is
orthogonal to the base eviction policy (LRU/SIEVE/S3-FIFO) — it layers
*partition-awareness* on top.

**Core concept:** Each request carries a `cache_partition_id` (set via
`SamplingParams.extra_args["cache_partition_id"]` or auto-resolved from
the served model name). The `BlockPool` tracks per-partition ref-count
contributions and exposes two new sharing policies:

#### 1. Two-level quota (`--kv-cache-partition-ref-caps`)

Hard caps on how many block refs each partition can hold. Prevents one
model/tenant from monopolising the cache.

```bash
vllm serve model_a model_b \
    --kv-cache-partition-ref-caps '{"model_a": 500, "model_b": 300}'
```

- `BlockPool.get_new_blocks()` raises `ValueError` if a partition
  would exceed its cap.
- Caps are **dynamically adjustable** at runtime via
  `set_partition_ref_caps()` — no engine restart needed.
- Implementation: `_partition_note_ref()` / `_partition_block_ref_totals`
  in `vllm/v1/core/block_pool.py`.

#### 2. Cost-aware eviction (`--kv-cache-partition-eviction-cost`)

Single-level GreedyDual-style victim selection. Each partition gets a
relative "prefill cost" multiplier. When evicting, the pool scans a
window of free blocks and picks `argmax((now - last_access) / cost)` —
i.e., stale blocks from cheap-to-recompute partitions are evicted first.

```bash
vllm serve expensive_model cheap_model \
    --kv-cache-partition-eviction-cost '{"expensive_model": 10.0, "cheap_model": 1.0}'
```

- Only composes with the **LRU base policy** (not SIEVE/S3-FIFO).
  The free block queue becomes `CostAwareFreeKVCacheBlockQueue`.
- Scan window default: 32 blocks. `popleft()` does a linear scan
  of `min(scan_window, num_free)` blocks.
- Implementation: `CostAwareFreeKVCacheBlockQueue` in
  `vllm/v1/core/kv_cache_utils.py`.

#### Plumbing changes

- `BlockPool.get_new_blocks(n, cache_partition_id)` — partition ID
  is now a required arg. Same for `touch()` and `free_blocks()`.
- `KVCacheBlock` gained `kv_eviction_tag` (last partition) and
  `kv_access_seq` (logical clock) fields for the cost-aware policy.
- `Request` carries `cache_partition_id` resolved by `InputProcessor`.
- **Metrics:** New Prometheus gauge `vllm:kv_cache_partition_block_refs`
  (per-partition ref counts). Also logged in the stat logger.
- `benchmark_block_pool.py` updated: `get_new_blocks` / `free_blocks`
  now require a partition ID string (`"__bench__"`).
- `benchmark_prefix_caching.py` updated: prints Prometheus-based
  prefix cache hit rate after generation; docstring shows the new
  `--kv-cache-partition-ref-caps` / `--kv-cache-partition-eviction-cost`
  CLI flags.

#### Implications for our evaluation

1. **New policy dimension:** We now have 5 policies to compare, not 3:
   LRU, SIEVE, S3-FIFO, LRU+quota, LRU+cost-aware. The budget sweep
   should cover the new ones.

2. **Multi-partition workloads needed:** The existing synthetic workloads
   all use a single implicit partition. We need workloads where multiple
   "models" or "tenants" compete for the same cache pool, with different
   access frequencies and prefix sizes.

3. **Fairness metrics:** Beyond TTFT and throughput, we should measure
   per-partition hit rate and per-partition TTFT to show that quota/cost
   policies improve fairness vs. unpartitioned LRU.

4. **Quota reapportionment:** Test dynamic `set_partition_ref_caps()`
   under load — does changing caps mid-flight cause latency spikes?

### Agentic workload evaluation

Agent workloads are an emerging and highly relevant evaluation dimension for
KV cache eviction. Two recent papers directly address this:

**"Don't Break the Cache" (2025):** First comprehensive evaluation of prompt
caching for long-horizon agentic tasks. Key findings:
- Agents execute 30-50+ tool calls per session, accumulating tens of thousands
  of tokens. Context grows dynamically and unpredictably.
- **Full-context caching can hurt latency** because dynamic tool results
  pollute the cache with content that won't be reused across sessions.
- Strategic cache boundary control (cache system prompt only, exclude tool
  results) outperforms naive full-context caching.
- 45-80% cost savings, 13-31% TTFT improvement across providers.
- Uses **DeepResearchBench** (multi-turn web search agent, 100 PhD-level
  research questions, open-source via `deep-agents` library).
- Implication for us: eviction policies that distinguish between stable
  prefixes (system prompt) and volatile content (tool results) should win.
  SIEVE's visited-bit mechanism could naturally handle this — system prompt
  blocks get visited repeatedly, tool result blocks don't.

**Continuum (NeurIPS 2025 submission):** Addresses KV cache management for
agentic workloads where tool calls create pauses that break cache retention.
- Current inference engines evict KV caches for paused requests, forcing full
  recomputation when the agent resumes after a tool call.
- Introduces KV cache TTL (time-to-live) to retain caches across tool-call
  gaps, with adaptive TTL based on expected tool duration.
- Collected **SWE-Agent inference traces** (to be open-sourced) — real
  multi-turn coding agent workloads with tool calls of varying duration.
- Implication for us: our eviction policies should be tested under
  interleaved request patterns where some requests pause (tool call) and
  resume, competing with new requests for cache space.

**How to evaluate on agent workloads:**

1. **Synthetic agent traces** (easiest, do first): Generate prompts that mimic
   the agent pattern: large shared system prompt (2-10K tokens) + growing
   conversation history + tool call/result pairs. Use `vllm bench serve` with
   a custom `.jsonl` dataset where prompts share a common prefix but diverge
   at different points (simulating different tool call depths).

2. **DeepResearchBench** (medium effort): Use `deep-agents` library to run
   real agent sessions against our vLLM fork. Measure TTFT per turn as
   context grows. Compare policies on whether they retain the system prompt
   and early conversation turns.

3. **SWE-Agent traces** (when available): Replay real coding agent traces
   with tool-call pauses. Measure cache retention across pauses under
   different policies.

4. **BurstGPT traces** (already supported): Real Azure ChatGPT API workload
   traces with realistic bursty arrival patterns. vLLM's benchmark CLI
   natively supports this:
   ```bash
   # Download: wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
   vllm bench serve --dataset-name burstgpt \
       --dataset-path BurstGPT_without_fails_2.csv \
       --num-prompts 500
   ```

## What we already measure (and gaps)

### Current benchmarks

| Script | Workloads | Metrics | Gap |
|--------|-----------|---------|-----|
| `benchmark_kv_cache.py` | Synthetic (uniform/zipfian/temporal/scan-resistant) + HELM few-shot (copa/piqa/winogrande) | TTFT, throughput, batch latency | No online serving mode, no realistic traces, no multi-turn |
| `benchmark_kv_cache_budget_sweep.py` | Policy × budget grid | TTFT, throughput (CSV output) | Wraps `benchmark_kv_cache.py`; same workload limitations |

### Gaps to fill

1. **No online serving benchmark** — current scripts use offline `LLM.generate()`.
   Real serving has concurrent requests with Poisson arrivals, which creates
   memory pressure the eviction policy must handle.

2. **No realistic trace data** — ShareGPT conversations have natural
   prefix-sharing from system prompts. We use only synthetic word salad.

3. **No multi-turn / multi-request** — SCBench shows this is critical for
   evaluating whether policies retain the right blocks across conversation
   turns.

4. **No cache budget sweep** — ✅ Now implemented via
   `benchmark_kv_cache_budget_sweep.py`.

5. **No hit-rate tracking** — vLLM logs `Prefix cache hit rate` but our
   scripts don't capture it programmatically.

6. **No micro-benchmark of policy overhead** — `benchmark_block_pool.py`
   exists but doesn't vary by policy. We should measure alloc/free latency
   for LRU vs SIEVE vs S3-FIFO.

7. **No agentic workloads** — Agent tool-calling patterns create unique
   cache pressure (growing context, interleaved pauses, mix of stable and
   volatile content). See agentic workload section above.

8. **No multi-partition / multi-tenant workloads** — The new two-level
   quota and cost-aware eviction policies from `cache-partition` need
   workloads where ≥2 partitions with different access patterns and
   prefill costs compete for the same block pool. Existing benchmarks
   use a single implicit partition.

## New benchmarks to implement

### 1. `benchmark_kv_cache_budget_sweep.py` ✅ IMPLEMENTED

Sweeps cache budgets and eviction policies, producing a CSV table of
(policy × budget → TTFT, throughput, hit_rate). This is the core experiment
for the paper: does SIEVE beat LRU at tight budgets?

Uses our existing workload generators (Zipfian, temporal, scan-resistant)
but automates the cross-product sweep and captures hit rate from vLLM logs.

### 2. Online serving with ShareGPT (use existing vLLM infra)

For GPU runs, use `vllm bench serve` with prefix repetition or ShareGPT:
```bash
# Start server with SIEVE policy
VLLM_KV_OFFLOAD_POLICY=sieve vllm serve $MODEL \
    --enable-prefix-caching --gpu-memory-utilization 0.6

# Prefix repetition stress test (directly tests eviction)
vllm bench serve --backend vllm --model $MODEL \
    --dataset-name prefix_repetition \
    --num-prompts 500 \
    --prefix-repetition-prefix-len 1024 \
    --prefix-repetition-suffix-len 256 \
    --prefix-repetition-num-prefixes 20 \
    --prefix-repetition-output-len 64

# ShareGPT real traces
vllm bench serve --backend vllm --model $MODEL \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500

# BurstGPT realistic arrival patterns
vllm bench serve --backend vllm --model $MODEL \
    --dataset-name burstgpt \
    --dataset-path BurstGPT_without_fails_2.csv \
    --num-prompts 500

# With goodput SLOs (show which policy meets SLA better)
vllm bench serve --backend vllm --model $MODEL \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --goodput ttft:3000 tpot:100

# With ramp-up to find throughput cliff
vllm bench serve --backend vllm --model $MODEL \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --ramp-up-strategy linear --ramp-up-start-rps 1 --ramp-up-end-rps 50
```
Repeat for `lru`, `s3fifo`. Compare TTFT and throughput distributions.

### 3. Agentic workload benchmark (NEW)

Generate a custom `.jsonl` dataset mimicking agent tool-calling patterns:
- Shared system prompt (large, stable)
- Growing conversation context per "session"
- Tool call/result interleaving (some content volatile, some stable)
- Multiple sessions sharing the same system prompt

Feed to `vllm bench serve --dataset-name custom --dataset-path agent_traces.jsonl`.
This tests whether SIEVE naturally retains system prompt blocks (frequently
visited) while evicting one-shot tool result blocks.

### 4. Multi-turn shared context (SCBench integration)

SCBench (`microsoft/SCBench` on HuggingFace) provides multi-request tasks
where many queries hit the same long context. This directly exercises prefix
cache eviction under realistic sharing patterns. Integration requires
`pip install MInference` and running the SCBench evaluation harness against
our modified vLLM.

### 5. Block pool micro-benchmark per policy

Extend `benchmark_block_pool.py` to sweep `VLLM_KV_OFFLOAD_POLICY` across
LRU/SIEVE/S3-FIFO and report alloc/free latency. This isolates policy
bookkeeping overhead from end-to-end effects.

### 6. Multi-partition fairness benchmark ✅ IMPLEMENTED

`benchmark_kv_cache.py --workload multi-partition` with asymmetric
per-partition configs via `--partition-config` (JSON).

Each partition gets its own prefix pool, request count, and prefix size.
Requests are interleaved across partitions and batched together, with
per-request `cache_partition_id` injected into `SamplingParams.extra_args`.
Results include **per-partition TTFT breakdown**.

**Ref-cap isolation experiment:**
```bash
# Baseline: plain LRU, hog monopolises cache
VLLM_KV_OFFLOAD_POLICY=lru python benchmarks/benchmark_kv_cache.py \
    --model $MODEL --enable-prefix-caching --enforce-eager \
    --workload multi-partition \
    --partition-config '{"hog": {"num_requests": 64, "prefix_words": 200},
                         "starved": {"num_requests": 16, "prefix_words": 2000}}'

# With quota: cap hog to 300 block refs
VLLM_KV_OFFLOAD_POLICY=lru python benchmarks/benchmark_kv_cache.py \
    --model $MODEL --enable-prefix-caching --enforce-eager \
    --workload multi-partition \
    --partition-config '{"hog": {"num_requests": 64, "prefix_words": 200},
                         "starved": {"num_requests": 16, "prefix_words": 2000}}' \
    --kv-cache-partition-ref-caps '{"hog": 300}'
```

**Cost-aware eviction experiment:**
```bash
# Baseline: plain LRU, expensive and cheap blocks treated equally
VLLM_KV_OFFLOAD_POLICY=lru python benchmarks/benchmark_kv_cache.py \
    --model $MODEL --enable-prefix-caching --enforce-eager \
    --workload multi-partition \
    --partition-config '{"cheap": {"num_requests": 64, "prefix_words": 200},
                         "expensive": {"num_requests": 16, "prefix_words": 2000}}'

# With cost-aware: protect expensive partition's blocks
VLLM_KV_OFFLOAD_POLICY=lru python benchmarks/benchmark_kv_cache.py \
    --model $MODEL --enable-prefix-caching --enforce-eager \
    --workload multi-partition \
    --partition-config '{"cheap": {"num_requests": 64, "prefix_words": 200},
                         "expensive": {"num_requests": 16, "prefix_words": 2000}}' \
    --kv-cache-partition-eviction-cost '{"cheap": 1.0, "expensive": 10.0}'
```

**Key metrics:** Compare per-partition TTFT and per-partition hit rate
between baseline and policy-enabled runs. The delta shows whether the
policy improves fairness (ref caps) or smart prioritization (cost-aware).

**Block pool micro-benchmark:**

`benchmark_block_pool.py` already passes `"__bench__"` as partition ID.
Extend to test:
- `CostAwareFreeKVCacheBlockQueue` with multiple partition tags and
  varying cost multipliers — measure victim selection overhead vs plain
  LRU `popleft()`.
- Quota enforcement overhead on `get_new_blocks()`.

## Implementation priorities

1. ✅ Synthetic workloads (done)
2. ✅ HELM few-shot (done)
3. ✅ Budget sweep (done — `benchmark_kv_cache_budget_sweep.py`)
4. ✅ Multi-partition offline benchmark (done — `--workload multi-partition`)
5. **Run partition experiments on GPU** — tight cache budget needed to force
   eviction; compare LRU vs LRU+quota vs LRU+cost-aware per-partition TTFT
6. **ShareGPT / BurstGPT / prefix_repetition online serving** — use existing
   `vllm bench serve`, just need GPU
7. **Synthetic agent traces** — generate `.jsonl` with agent-like patterns,
   feed to `vllm bench serve --dataset-name custom`
8. **Block pool micro-benchmark** — extend `benchmark_block_pool.py` to
   sweep policies (LRU/SIEVE/S3-FIFO/cost-aware) and measure overhead
9. **SCBench multi-request** — longer term, for final paper
10. **GuideLLM** — consider for production-grade evaluation and reporting
