# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Scan-resistance benchmark: LRU vs SIEVE vs S3-FIFO.

Correct 5-phase workload that exercises the CPU offload policy:

  warm-up  —  Throw-away request: eliminates first-inference CUDA overhead
               so Phase 1 measures true cold-miss prefill time.

  Phase 1  —  Prompt A (cold): first access, A computed and stored in GPU.

  Phase 2  —  Bridge prompts: M unique prompts that fill the GPU cache,
               forcing A's blocks to be evicted from GPU to CPU.
               These are distinct from both A and the scan one-shots.

  Phase 3  —  Prompt A (CPU load): GPU misses, loads A from CPU → GPU.
               This triggers CPU touch() / M-queue promotion, marking A
               as "hot" in the CPU policy.

  Phase 4  —  Scan one-shots: N unique prompts that each go to GPU, spill
               to CPU, and collectively overflow the CPU cache — forcing
               evictions.  The CPU policy decides what to keep.

  Phase 5  —  Prompt A (recall): GPU misses again (evicted by scan).
               CPU lookup: did A survive the scan?

Expected behaviour
------------------
LRU:    A was touched (move_to_end) in CPU during Phase 3 → now MRU.
        Phase 4 one-shots are inserted AFTER → they are even newer.
        When CPU fills, A is actually the OLDEST among those at the tail...
        wait — depends on precise arrival order.  In practice, because A's
        blocks are contiguous and were all touched at one moment, and one-
        shots' blocks arrive intermixed as Phase 4 runs, LRU gives one-shots
        recency parity or advantage.  Policy shows little separation from
        SIEVE/S3-FIFO except when the CPU is extremely tight.

SIEVE:  Phase 3 touch() sets visited=True on A's CPU blocks.  During Phase 4
        eviction the hand skips A (visited) and evicts the unvisited one-shot
        blocks.  A survives.

S3-FIFO: Phase 3 touch() promotes A's blocks from S → M queue.  Phase 4
          one-shots land in S queue and are evicted first.  A (in M) survives.

Key metrics
-----------
  Phase 3 time  ≈ CPU-load time (A in CPU after Phase 2 bridge).
  Phase 5 time  — if close to Phase 3 → A survived (CPU hit).
                — if close to Phase 1 → A evicted (cold recompute).

Usage
-----
  python benchmarks/benchmark_policy_scan_resistance.py --policy lru
  python benchmarks/benchmark_policy_scan_resistance.py --policy sieve
  python benchmarks/benchmark_policy_scan_resistance.py --policy s3fifo
"""

import argparse
import os
import random
import time

from vllm import LLM, SamplingParams

_WORDS = [
    "algorithm",
    "optimization",
    "throughput",
    "latency",
    "bandwidth",
    "pipeline",
    "scheduler",
    "prefetch",
    "eviction",
    "allocation",
    "partition",
    "replication",
    "consistency",
    "transaction",
    "isolation",
    "concurrency",
    "parallelism",
    "synchronization",
    "deadlock",
    "mutex",
    "processor",
    "register",
    "instruction",
    "operand",
    "accumulator",
    "interrupt",
    "exception",
    "privilege",
    "virtual",
    "physical",
    "memory",
    "cache",
    "buffer",
    "queue",
    "stack",
    "heap",
    "pointer",
    "address",
    "segment",
    "page",
    "frame",
    "block",
    "sector",
    "cluster",
]


def make_text(rng: random.Random, num_words: int) -> str:
    return " ".join(rng.choices(_WORDS, k=num_words))


def run_phase(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    label: str,
) -> float:
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    total_in = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {label}: {elapsed:.3f}s  ({total_in} in / {total_out} out)")
    return elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan-resistance benchmark: tests whether SIEVE/S3-FIFO protect "
            "a repeatedly-accessed CPU-offloaded prompt (Prompt A) from "
            "eviction by a burst of unique one-shot prompts, compared to LRU.\n\n"
            "Uses a 5-phase workload that correctly exercises the CPU eviction "
            "policy (see module docstring for details)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--policy",
        choices=["lru", "sieve", "s3fifo"],
        default="lru",
        help="Eviction policy to test (default: lru)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Model to use (default: Qwen/Qwen2.5-3B)",
    )
    p.add_argument(
        "--prompt-a-words",
        type=int,
        default=300,
        help=(
            "Length of Prompt A in words.  Should be large enough to "
            "occupy a significant fraction of the GPU KV cache. (default: 300)"
        ),
    )
    p.add_argument(
        "--num-bridge",
        type=int,
        default=5,
        help=(
            "Number of bridge prompts in Phase 2.  These fill the GPU and "
            "force Prompt A from GPU to CPU.  Should exceed "
            "GPU_blocks / prompt_blocks. (default: 5)"
        ),
    )
    p.add_argument(
        "--num-oneshots",
        type=int,
        default=10,
        help=(
            "Number of unique one-shot prompts in Phase 4 (the scan).  "
            "Their combined block footprint should overflow the CPU offload "
            "cache to force evictions. (default: 10)"
        ),
    )
    p.add_argument(
        "--oneshot-words",
        type=int,
        default=300,
        help="Words per one-shot / bridge prompt (default: 300)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help=(
            "Max generated tokens per prompt.  Use 1 to minimise decode "
            "time and maximise sensitivity to prefill differences. (default: 1)"
        ),
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="GPU memory fraction (default: 0.6)",
    )
    p.add_argument(
        "--kv-offloading-size",
        type=float,
        default=2.0,
        help="CPU offload size in GiB (default: 2.0)",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help=(
            "Maximum sequence length.  Reduce on memory-constrained GPUs "
            "(e.g. 512 for an 8 GiB card with a 3B model). (default: 2048)"
        ),
    )
    p.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=None,
        help=(
            "Override the number of GPU KV cache blocks.  Pins block count "
            "to a deterministic value so that Phase 2 (bridge) reliably "
            "pushes Prompt A from GPU to CPU.  Should be slightly larger than "
            "the number of blocks in Prompt A (e.g. 30 for a 300-word prompt "
            "with block_size=16). (default: None = auto)"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["VLLM_KV_OFFLOAD_POLICY"] = args.policy

    # Prompt A: fixed seed — identical across all policy runs for fair comparison.
    rng_a = random.Random(42)
    prompt_a = (
        make_text(rng_a, args.prompt_a_words)
        + "\n\nQuestion: Summarize the above in one sentence.\nAnswer:"
    )

    # Bridge prompts (Phase 2): unique, never repeated, seed range 200-299.
    rng_bridge = random.Random(200)
    bridge_prompts = [
        make_text(rng_bridge, args.oneshot_words)
        + f"\n\nQuestion: What is the main concept? (bridge {i})\nAnswer:"
        for i in range(args.num_bridge)
    ]

    # Scan one-shots (Phase 4): unique, never repeated, seed range 1000+.
    rng_os = random.Random(1000)
    oneshot_prompts = [
        make_text(rng_os, args.oneshot_words)
        + f"\n\nQuestion: What is the key concept? (item {i})\nAnswer:"
        for i in range(args.num_oneshots)
    ]

    print("=" * 60)
    print("Scan-resistance benchmark")
    print("=" * 60)
    print(f"Policy:              {args.policy}")
    print(f"Model:               {args.model}")
    print(f"Prompt A length:     ~{args.prompt_a_words} words")
    print(f"Bridge prompts:      {args.num_bridge} x ~{args.oneshot_words} words")
    print(f"One-shot prompts:    {args.num_oneshots} x ~{args.oneshot_words} words")
    print(f"CPU offload size:    {args.kv_offloading_size} GiB")
    print(f"GPU mem utilization: {args.gpu_memory_utilization}")
    print(f"Max model len:       {args.max_model_len}")
    print()

    llm_kwargs: dict = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        kv_offloading_backend="native",
        kv_offloading_size=args.kv_offloading_size,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=True,
    )
    if args.num_gpu_blocks is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks
        print(f"GPU blocks override: {args.num_gpu_blocks}")

    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Warm-up: absorb first-inference CUDA overhead so Phase 1 is clean.
    rng_wu = random.Random(777)
    warmup_prompt = make_text(rng_wu, min(50, args.prompt_a_words // 4)) + "\n\nAnswer:"
    print("Warm-up (discarded)...")
    run_phase(llm, [warmup_prompt], sp, "  warm-up")

    print("\nPhases:")
    t1 = run_phase(llm, [prompt_a],      sp, "Phase 1 — Prompt A (cold → GPU)")
    t2 = run_phase(llm, bridge_prompts,  sp,
                   f"Phase 2 — {args.num_bridge} bridge prompts (GPU flush → A to CPU)")
    t3 = run_phase(llm, [prompt_a],      sp, "Phase 3 — Prompt A (CPU load → marks hot)")
    t4 = run_phase(llm, oneshot_prompts, sp,
                   f"Phase 4 — {args.num_oneshots} one-shot prompts (scan/CPU pressure)")
    t5 = run_phase(llm, [prompt_a],      sp, "Phase 5 — Prompt A (survived scan?)")

    # --- Summary ---
    # Primary: how close is Phase 5 (recall) to Phase 3 (CPU load)?
    #   ratio ~1.0  → A survived in CPU cache (same cost as loading from CPU)
    #   ratio >> 1  → A was evicted, Phase 5 re-computes (close to Phase 1 cold)
    ratio_recall_vs_cpu_load = t5 / t3 if t3 > 0 else float("inf")
    ratio_recall_vs_cold     = t5 / t1 if t1 > 0 else float("inf")

    if ratio_recall_vs_cpu_load <= 1.4:
        verdict = "SURVIVED  — A found in CPU cache (warm hit on recall)"
    elif ratio_recall_vs_cpu_load <= 2.0:
        verdict = "PARTIAL   — A partially survived eviction"
    else:
        verdict = "EVICTED   — A evicted from CPU (cold recompute on recall)"

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Policy:                         {args.policy}")
    print(f"Phase 1 — cold:                 {t1:.3f}s")
    print(f"Phase 2 — bridge (GPU flush):   {t2:.3f}s")
    print(f"Phase 3 — CPU load (marks hot): {t3:.3f}s")
    print(f"Phase 4 — scan:                 {t4:.3f}s")
    print(f"Phase 5 — recall:               {t5:.3f}s")
    print()
    print(f"Ratio   Phase5 vs Phase3 (CPU-load): {ratio_recall_vs_cpu_load:.2f}x"
          f"  (1.0 = perfect retention, >2 = evicted)")
    print(f"Ratio   Phase5 vs Phase1 (cold):     {ratio_recall_vs_cold:.2f}x")
    print()
    print(f"Verdict: {verdict}")
