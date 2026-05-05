# Running GPU benchmarks on CPU (macOS)

These KV-cache benchmarks were written for GPU with KV-cache offloading
(GPU → CPU). To run them on a CPU-only machine (e.g. Apple Silicon Mac),
the unified `benchmark_kv_cache.py` auto-detects CPU and sets
`VLLM_CPU_KVCACHE_SPACE` accordingly.

On CPU, KV cache size is controlled by `VLLM_CPU_KVCACHE_SPACE` (in GiB),
exposed via `--cpu-kv-cache-space`. GPU-specific engine args like
`--gpu-memory-utilization` are simply not passed on CPU runs.

Since `benchmark_kv_cache.py` uses `EngineArgs` / `LLM.from_engine_args()`
(the same pattern as `benchmark_prefix_caching.py`), all vLLM engine flags
are available from the CLI. The engine handles CPU vs GPU differences
natively.

## Quick start (macOS)

```bash
# One-time setup (from repo root)
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
uv pip install datasets   # for --workload helm

# Synthetic workload
VLLM_KV_OFFLOAD_POLICY=sieve .venv/bin/python benchmarks/benchmark_kv_cache.py \
    --model Qwen/Qwen2.5-0.5B --enable-prefix-caching \
    --workload scan-resistant --num-batches 2 --batch-size 4 \
    --prefix-words 200 --max-tokens 10

# HELM few-shot
VLLM_KV_OFFLOAD_POLICY=lru .venv/bin/python benchmarks/benchmark_kv_cache.py \
    --model Qwen/Qwen2.5-0.5B --enable-prefix-caching \
    --workload helm --helm-task copa --num-test 10 --batch-size 5 \
    --max-tokens 10
```
