# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end benchmark for LoRA-aware CPU KV cache eviction policies.

Unlike ``benchmark_lora_policy.py`` (pure-Python simulation), this script
drives real ``LLM.generate`` calls with multiple LoRA adapters and an
offloaded CPU KV cache, so the policy is exercised through the actual
vLLM scheduler path — including the ``register_block_adapters`` /
``update_live_adapters`` hooks wired into the offloading scheduler.

Workload scenarios (chosen as the minimal set that differentiates
adapter-aware policies from baselines):

  * ``adapter_locality``
      Bursts of requests for one adapter, then another. Rewards policies
      that keep each adapter's blocks together when the adapter returns.

  * ``adapter_thrashing``
      Round-robin cycling through more adapters than fit in GPU LoRA
      slots. Stresses adapter churn and the "dead blocks of a departed
      adapter" problem.

  * ``mixed_popularity``
      Zipfian adapter selection. Realistic mixed workload where a few
      adapters dominate but a long tail of others appears.

Because loading a large base + many LoRAs is expensive, this benchmark
runs one ``(scenario, policy)`` combination per invocation. Wrap it in a
shell loop if you want to sweep.

Prerequisites:
  * A local or HF-downloadable base model.
  * One LoRA adapter path compatible with the base model (we replicate
    it under ``--num-adapters`` distinct names so every adapter is a
    separate identity to the scheduler — the KV-policy signal we are
    measuring is adapter *identity*, not weight diversity).

Example:
  VLLM_KV_OFFLOAD_POLICY=lora_budget:tinylfu \\
    .venv/bin/python benchmarks/benchmark_lora_e2e.py \\
      --model meta-llama/Llama-3.2-3B-Instruct \\
      --lora-path jeeejeee/llama32-3b-text2sql-spider \\
      --num-adapters 6 --scenario adapter_thrashing
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Short unique question fragments appended after the per-request prefix.
# Keeping the prefix large (and shared within an adapter) is what makes
# the KV cache behaviour interesting — otherwise nothing would be worth
# offloading.
_QUESTIONS = [
    "Summarize the key ideas in one sentence.",
    "List two trade-offs mentioned above.",
    "What is the most important concept here?",
    "Suggest a follow-up question a reader might ask.",
    "In two words, name the topic.",
    "Identify one concrete example from the passage.",
    "Rewrite the first clause in plain English.",
    "Which concept would you explain first to a beginner?",
]


def _adapter_prefix_words(adapter_idx: int, num_words: int) -> list[str]:
    """Generate a deterministic word list used as a per-adapter shared prefix.

    Different adapters get different prefixes so prefix caching cannot
    collapse them into a single logical block stream — we want each
    adapter's KV blocks to be distinct in the offload cache.
    """
    rng = random.Random(1_000 + adapter_idx)
    vocab = [
        "throughput", "latency", "scheduler", "prefetch", "allocation",
        "replication", "consistency", "isolation", "concurrency",
        "pipeline", "processor", "instruction", "operand", "interrupt",
        "virtual", "physical", "bandwidth", "deadlock", "partition",
        "mutex", "register", "accumulator", "privilege", "synchronization",
    ]
    return rng.choices(vocab, k=num_words)


def _build_prompt(
    adapter_idx: int, prefix_words: int, suffix_words: int, question_idx: int
) -> str:
    prefix = " ".join(_adapter_prefix_words(adapter_idx, prefix_words))
    # Per-request unique padding so the tail of each prompt differs even
    # within the same adapter — this is where *new* blocks are generated.
    rng = random.Random(hash((adapter_idx, question_idx)) & 0xFFFFFFFF)
    padding = " ".join(rng.choices(prefix.split(), k=suffix_words))
    question = _QUESTIONS[question_idx % len(_QUESTIONS)]
    return f"{prefix}\n\n{padding}\n\nQuestion: {question}\nAnswer:"


# --- Workload scenarios (return a list of adapter indices) ---


def gen_adapter_locality(
    num_requests: int, num_adapters: int, burst_len: int, seed: int
) -> list[int]:
    rng = random.Random(seed)
    out: list[int] = []
    while len(out) < num_requests:
        a = rng.randrange(num_adapters)
        take = min(burst_len, num_requests - len(out))
        out.extend([a] * take)
    return out


def gen_adapter_thrashing(num_requests: int, num_adapters: int) -> list[int]:
    return [i % num_adapters for i in range(num_requests)]


def gen_mixed_popularity(
    num_requests: int, num_adapters: int, alpha: float, seed: int
) -> list[int]:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, num_adapters + 1)
    probs = 1.0 / ranks**alpha
    probs /= probs.sum()
    return rng.choice(num_adapters, size=num_requests, p=probs).tolist()


_SCENARIOS = ("adapter_locality", "adapter_thrashing", "mixed_popularity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end LoRA-aware KV cache eviction benchmark"
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model (default: %(default)s)",
    )
    p.add_argument(
        "--lora-path",
        default="jeeejeee/llama32-3b-text2sql-spider",
        help="HF repo or local path to a LoRA adapter compatible with --model. "
        "Replicated under distinct names to simulate --num-adapters adapters.",
    )
    p.add_argument(
        "--num-adapters",
        type=int,
        default=6,
        help="Number of distinct LoRA adapter *identities* to simulate "
        "(all load the same weights, but scheduler sees them as separate).",
    )
    p.add_argument(
        "--max-loras",
        type=int,
        default=3,
        help="GPU LoRA slots. If --num-adapters > --max-loras you get "
        "adapter churn, which is what makes the LoRA-aware policies matter.",
    )
    p.add_argument(
        "--max-cpu-loras",
        type=int,
        default=None,
        help="CPU-side LoRA cache (default: --num-adapters)",
    )
    p.add_argument(
        "--max-lora-rank", type=int, default=16, help="Max LoRA rank"
    )
    p.add_argument(
        "--scenario",
        choices=_SCENARIOS,
        default="adapter_thrashing",
        help="Workload pattern (default: %(default)s)",
    )
    p.add_argument(
        "--policy",
        default=None,
        help="KV offload policy name (e.g. lru, sieve, tinylfu, "
        "lora_budget:tinylfu). Falls back to $VLLM_KV_OFFLOAD_POLICY, "
        "then 'lru'.",
    )
    p.add_argument("--num-requests", type=int, default=120)
    p.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Requests per llm.generate() call",
    )
    p.add_argument(
        "--prefix-words",
        type=int,
        default=500,
        help="Words in the per-adapter shared prefix",
    )
    p.add_argument(
        "--suffix-words",
        type=int,
        default=200,
        help="Unique padding words per request",
    )
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max sequence length (lower = less KV reserve, easier on VRAM)",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument(
        "--kv-offloading-size",
        type=float,
        default=2.0,
        help="CPU offload budget in GiB",
    )
    p.add_argument(
        "--burst-len",
        type=int,
        default=8,
        help="Burst length for adapter_locality",
    )
    p.add_argument(
        "--zipfian-alpha",
        type=float,
        default=1.2,
        help="Skew for mixed_popularity",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_lora_path(path: str) -> str:
    """Accept either a local path or a HF repo id; return a local path."""
    if os.path.isdir(path):
        return path
    return snapshot_download(repo_id=path)


def _build_lora_requests(
    local_path: str, num_adapters: int
) -> list[LoRARequest]:
    # Distinct name + int_id per adapter so the scheduler and the KV block
    # hasher treat them as different adapters, even though the weights are
    # identical on disk.
    return [
        LoRARequest(f"bench-lora-{i}", i + 1, local_path)
        for i in range(num_adapters)
    ]


def _build_adapter_indices(args: argparse.Namespace) -> list[int]:
    if args.scenario == "adapter_locality":
        return gen_adapter_locality(
            args.num_requests, args.num_adapters, args.burst_len, args.seed
        )
    if args.scenario == "adapter_thrashing":
        return gen_adapter_thrashing(args.num_requests, args.num_adapters)
    if args.scenario == "mixed_popularity":
        return gen_mixed_popularity(
            args.num_requests,
            args.num_adapters,
            args.zipfian_alpha,
            args.seed,
        )
    raise ValueError(f"Unknown scenario: {args.scenario}")


def main() -> None:
    args = parse_args()

    if args.policy:
        os.environ["VLLM_KV_OFFLOAD_POLICY"] = args.policy
    effective_policy = os.environ.get("VLLM_KV_OFFLOAD_POLICY", "lru")

    print(f"Policy:            {effective_policy}")
    print(f"Scenario:          {args.scenario}")
    print(f"Model:             {args.model}")
    print(f"LoRA path:         {args.lora_path}")
    print(
        f"Adapters:          {args.num_adapters} "
        f"(GPU slots: {args.max_loras})"
    )
    print(
        f"Requests:          {args.num_requests} "
        f"({args.num_requests // args.batch_size} batches of {args.batch_size})"
    )
    print(
        f"Prompt size:       ~{args.prefix_words} prefix + "
        f"{args.suffix_words} suffix words"
    )
    print(f"CPU offload:       {args.kv_offloading_size} GiB")

    # Resolve LoRA path up-front (downloads once).
    local_lora_path = _resolve_lora_path(args.lora_path)
    lora_requests = _build_lora_requests(local_lora_path, args.num_adapters)

    # Build the request stream.
    adapter_indices = _build_adapter_indices(args)
    adapter_counts = defaultdict(int)
    for a in adapter_indices:
        adapter_counts[a] += 1
    print(f"Adapter distribution: {dict(sorted(adapter_counts.items()))}")

    prompts_with_lora: list[tuple[str, LoRARequest]] = []
    for i, a in enumerate(adapter_indices):
        prompts_with_lora.append(
            (
                _build_prompt(a, args.prefix_words, args.suffix_words, i),
                lora_requests[a],
            )
        )

    print(f"\nLoading {args.model}...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        max_cpu_loras=args.max_cpu_loras or args.num_adapters,
        kv_offloading_backend="native",
        kv_offloading_size=args.kv_offloading_size,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Chunk into batches preserving scenario order (order is what creates
    # the cache-interesting access pattern).
    batches: list[list[tuple[str, LoRARequest]]] = []
    for start in range(0, len(prompts_with_lora), args.batch_size):
        batches.append(prompts_with_lora[start : start + args.batch_size])

    print(f"\nRunning {len(batches)} batches...")
    batch_times: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0

    overall_start = time.time()
    for i, batch in enumerate(batches):
        prompts = [p for p, _ in batch]
        lreqs = [lr for _, lr in batch]

        t0 = time.time()
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lreqs,
            use_tqdm=False,
        )
        elapsed = time.time() - t0
        batch_times.append(elapsed)

        batch_in = sum(len(o.prompt_token_ids) for o in outputs)
        batch_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_input_tokens += batch_in
        total_output_tokens += batch_out

        # Adapter histogram for this batch — useful for spot-checking
        # the scenario is doing what we expect.
        hist = defaultdict(int)
        for _, lr in batch:
            hist[lr.lora_int_id] += 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:3]
        top_str = ",".join(f"{aid}:{c}" for aid, c in top)

        print(
            f"  Batch {i + 1}/{len(batches)}: {elapsed:5.2f}s "
            f"({batch_in:5d} in / {batch_out:4d} out) "
            f"top_adapters=[{top_str}]"
        )

    overall_elapsed = time.time() - overall_start

    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / overall_elapsed
    out_throughput = total_output_tokens / overall_elapsed

    print("\n--- Benchmark Results ---")
    print(f"Policy:                 {effective_policy}")
    print(f"Scenario:               {args.scenario}")
    print(f"Total Time:             {overall_elapsed:.2f} s")
    print(f"Total Input Tokens:     {total_input_tokens}")
    print(f"Total Output Tokens:    {total_output_tokens}")
    print(f"Total Token Throughput: {throughput:.2f} tokens/s")
    print(f"Output Only Throughput: {out_throughput:.2f} tokens/s")

    times_sorted = sorted(batch_times)
    n = len(times_sorted)
    print("\nBatch Latency Percentiles:")
    print(f"  P50: {times_sorted[n // 2]:.2f} s")
    print(f"  P95: {times_sorted[min(n - 1, int(n * 0.95))]:.2f} s")
    print(f"  P99: {times_sorted[min(n - 1, int(n * 0.99))]:.2f} s")

    print(
        "\nPolicy statistics (touches, evictions, CPU cache hit rate) "
        "are logged on process exit by CPUOffloadingManager — check stderr."
    )


if __name__ == "__main__":
    main()
