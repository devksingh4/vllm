"""
Benchmark KV cache eviction policies with various workload patterns.

Unified benchmark for comparing eviction policies (LRU, SIEVE, S3-FIFO) under
prefix-caching workloads.  Uses the same EngineArgs / LLM.from_engine_args()
pattern as benchmark_prefix_caching.py, so every vLLM engine flag is available
from the CLI.

Workload modes:
  - uniform:        single shared prefix, unique suffixes
  - zipfian:        N prefixes accessed with Zipfian popularity
  - temporal:       phase-shifting hot working set
  - scan-resistant: small hot set + sequential scan (stresses LRU)
  - helm:           HELM few-shot evaluation (copa / piqa / winogrande)

Examples:

  # Scan-resistant synthetic workload (GPU)
  VLLM_KV_OFFLOAD_POLICY=sieve python benchmarks/benchmark_kv_cache.py \\
      --model Qwen/Qwen2.5-3B --enable-prefix-caching --enforce-eager \\
      --workload scan-resistant --num-batches 4 --batch-size 32

  # HELM few-shot (CPU, macOS)
  VLLM_KV_OFFLOAD_POLICY=lru python benchmarks/benchmark_kv_cache.py \\
      --model Qwen/Qwen2.5-0.5B --enable-prefix-caching \\
      --workload helm --helm-task copa --num-test 100

  # Budget sweep helper (see benchmark_kv_cache_budget_sweep.py)
"""

import os
import random
import time
from collections import defaultdict
from typing import List

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

# ---------------------------------------------------------------------------
# CPU / GPU compatibility
# ---------------------------------------------------------------------------

IS_CPU = not torch.cuda.is_available()


def _setup_cpu_env(cpu_kv_cache_gib: int = 4) -> None:
    if IS_CPU:
        os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", str(cpu_kv_cache_gib))


# ---------------------------------------------------------------------------
# Synthetic prefix / suffix generation
# ---------------------------------------------------------------------------

_PREFIX_WORDS = [
    "algorithm", "optimization", "throughput", "latency", "bandwidth",
    "pipeline", "scheduler", "prefetch", "eviction", "allocation",
    "partition", "replication", "consistency", "transaction", "isolation",
    "concurrency", "parallelism", "synchronization", "deadlock", "mutex",
    "processor", "register", "instruction", "operand", "accumulator",
    "interrupt", "exception", "privilege", "virtual", "physical",
]

_QUESTIONS = [
    "Summarize the key themes in one sentence.",
    "What optimization technique appears most often?",
    "List three concepts related to concurrency.",
    "Identify a potential trade-off mentioned in the text.",
    "Which scheduling concept is most relevant to latency?",
    "What is the relationship between throughput and bandwidth?",
    "Name a synchronization primitive from the passage.",
    "How does the pipeline concept relate to parallelism here?",
    "What memory-management concepts are discussed?",
    "Describe the main topic in exactly two words.",
]


def generate_shared_prefix(num_words: int, seed: int = 42) -> str:
    return " ".join(random.Random(seed).choices(_PREFIX_WORDS, k=num_words))


def _unique_padding(rng: random.Random, num_words: int) -> str:
    return " ".join(rng.choices(_PREFIX_WORDS, k=num_words))

# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------


def _make_prompt(prefix: str, padding: str, question: str) -> str:
    return f"{prefix}\n\n{padding}\n\nQuestion: {question}\nAnswer:"


def workload_uniform(
    num_batches: int, batch_size: int, prefix_words: int, suffix_words: int,
) -> list[list[str]]:
    """Single shared prefix, unique suffixes."""
    prefix = generate_shared_prefix(prefix_words)
    rng = random.Random(123)
    batches = []
    for b in range(num_batches):
        batch = []
        for i in range(batch_size):
            q = _QUESTIONS[(b * batch_size + i) % len(_QUESTIONS)]
            batch.append(_make_prompt(prefix, _unique_padding(rng, suffix_words), q))
        batches.append(batch)
    return batches


def _zipfian_indices(
    num_prefixes: int, num_requests: int, alpha: float = 1.2, seed: int = 42,
) -> list[int]:
    np.random.seed(seed)
    ranks = np.arange(1, num_prefixes + 1)
    probs = 1.0 / (ranks ** alpha)
    probs /= probs.sum()
    return np.random.choice(num_prefixes, size=num_requests, p=probs).tolist()


def _temporal_indices(
    num_prefixes: int, num_requests: int,
    working_set_size: int = 10, phase_length: int = 100, seed: int = 42,
) -> list[int]:
    np.random.seed(seed)
    indices: list[int] = []
    num_phases = (num_requests + phase_length - 1) // phase_length
    for phase in range(num_phases):
        start = (phase * working_set_size // 2) % max(
            1, num_prefixes - working_set_size
        )
        hot = list(range(start, min(start + working_set_size, num_prefixes)))
        cold = [i for i in range(num_prefixes) if i not in hot]
        phase_reqs = min(phase_length, num_requests - len(indices))
        for _ in range(phase_reqs):
            if np.random.random() < 0.8 and hot:
                indices.append(np.random.choice(hot))
            elif cold:
                indices.append(np.random.choice(cold))
            elif hot:
                indices.append(np.random.choice(hot))
    return indices[:num_requests]


def _scan_resistant_indices(
    num_prefixes: int, num_requests: int,
    working_set_size: int = 5, scan_size: int = 50, seed: int = 42,
) -> list[int]:
    np.random.seed(seed)
    hot = list(range(working_set_size))
    scan = list(range(working_set_size, min(working_set_size + scan_size, num_prefixes)))
    pos = 0
    indices: list[int] = []
    for i in range(num_requests):
        if i % 10 < 7:
            indices.append(np.random.choice(hot))
        else:
            indices.append(scan[pos % len(scan)])
            pos += 1
    return indices


def _batches_from_indices(
    prefixes: list[str], indices: list[int],
    num_batches: int, batch_size: int, suffix_words: int,
) -> list[list[str]]:
    rng = random.Random(123)
    batches: list[list[str]] = []
    idx = 0
    for b in range(num_batches):
        batch: list[str] = []
        for i in range(batch_size):
            if idx >= len(indices):
                break
            q = _QUESTIONS[(b * batch_size + i) % len(_QUESTIONS)]
            batch.append(_make_prompt(
                prefixes[indices[idx]], _unique_padding(rng, suffix_words), q,
            ))
            idx += 1
        if batch:
            batches.append(batch)
    return batches


def workload_patterned(
    pattern: str, num_batches: int, batch_size: int,
    prefix_words: int, suffix_words: int, num_prefixes: int,
    zipfian_alpha: float = 1.5, working_set_size: int = 15,
) -> list[list[str]]:
    """Multi-prefix workloads: zipfian, temporal, scan-resistant."""
    prefixes = [
        generate_shared_prefix(prefix_words + i * 10, seed=42 + i)
        for i in range(num_prefixes)
    ]
    n = num_batches * batch_size

    if pattern == "zipfian":
        indices = _zipfian_indices(num_prefixes, n, zipfian_alpha)
    elif pattern == "temporal":
        indices = _temporal_indices(num_prefixes, n, working_set_size)
    elif pattern == "scan-resistant":
        indices = _scan_resistant_indices(num_prefixes, n, working_set_size)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    counts = defaultdict(int)
    for i in indices:
        counts[i] += 1
    print(f"Prefix access distribution (top 5): "
          f"{dict(sorted(counts.items(), key=lambda x: -x[1])[:5])}")

    return _batches_from_indices(
        prefixes, indices, num_batches, batch_size, suffix_words,
    )


# ---------------------------------------------------------------------------
# HELM few-shot workload
# ---------------------------------------------------------------------------

def workload_helm(
    task: str, num_examples: int, num_test: int, batch_size: int,
) -> list[list[str]]:
    """HELM few-shot prompts.  All prompts share the same examples prefix."""
    from datasets import load_dataset

    task_datasets = {
        "copa": ("super_glue", "copa"),
        "piqa": ("piqa", None),
        "winogrande": ("winogrande", "winogrande_xl"),
    }
    if task not in task_datasets:
        raise ValueError(f"Unknown HELM task: {task}. Choose from {list(task_datasets)}")

    ds_name, config = task_datasets[task]
    load_kw = dict(name=config) if config else {}
    train = load_dataset(ds_name, split="train", **load_kw)
    test = load_dataset(ds_name, split="validation", **load_kw)

    # Build shared few-shot prefix
    prefix = "Answer the following questions.\n\n"
    for i, ex in enumerate(train):
        if i >= num_examples:
            break
        if task == "copa":
            ans = ex["choice1"] if ex["label"] == 0 else ex["choice2"]
            prefix += (f"Premise: {ex['premise']}\n"
                       f"Question: What is the {ex['question']}?\n"
                       f"Choice 1: {ex['choice1']}\nChoice 2: {ex['choice2']}\n"
                       f"Answer: {ans}\n\n")
        elif task == "piqa":
            ans = ex["sol1"] if ex["label"] == 0 else ex["sol2"]
            prefix += f"Goal: {ex['goal']}\nSolution: {ans}\n\n"
        elif task == "winogrande":
            correct = ex["option1"] if ex["answer"] == "1" else ex["option2"]
            prefix += f"Sentence: {ex['sentence']}\nAnswer: {correct}\n\n"

    # Build test prompts
    prompts: List[str] = []
    for i, ex in enumerate(test):
        if i >= num_test:
            break
        if task == "copa":
            prompts.append(
                prefix + f"Premise: {ex['premise']}\n"
                f"Question: What is the {ex['question']}?\n"
                f"Choice 1: {ex['choice1']}\nChoice 2: {ex['choice2']}\n"
                f"Answer:")
        elif task == "piqa":
            prompts.append(prefix + f"Goal: {ex['goal']}\nSolution:")
        elif task == "winogrande":
            prompts.append(prefix + f"Sentence: {ex['sentence']}\nAnswer:")

    # Chunk into batches
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]


# ---------------------------------------------------------------------------
# Benchmark harness (shared by all workloads)
# ---------------------------------------------------------------------------

def run_benchmark(
    llm: LLM, batches: list[list[str]], sampling_params: SamplingParams,
) -> dict:
    """Run batches through the LLM and return collected metrics."""
    total_in = 0
    total_out = 0
    batch_times: list[float] = []
    all_ttft: list[float] = []

    num_batches = len(batches)
    print(f"\nRunning {num_batches} batches...")
    overall_start = time.time()

    for i, batch in enumerate(batches):
        t0 = time.time()
        outputs = llm.generate(batch, sampling_params)
        elapsed = time.time() - t0
        batch_times.append(elapsed)

        b_in = sum(len(o.prompt_token_ids or []) for o in outputs)
        b_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_in += b_in
        total_out += b_out

        # Per-request TTFT
        b_ttft = [
            o.metrics.first_token_latency
            for o in outputs
            if o.metrics is not None and o.metrics.first_token_latency > 0
        ]
        all_ttft.extend(b_ttft)

        ttft_str = ""
        if b_ttft:
            ttft_str = f"  ttft_mean={sum(b_ttft)/len(b_ttft)*1000:.0f}ms"

        print(f"  Batch {i+1}/{num_batches}: {elapsed:.2f}s "
              f"({b_in} in / {b_out} out){ttft_str}")

    wall = time.time() - overall_start
    return {
        "wall_s": wall,
        "total_in": total_in,
        "total_out": total_out,
        "batch_times": batch_times,
        "all_ttft": all_ttft,
    }


def print_results(metrics: dict, policy: str, workload: str) -> None:
    """Pretty-print benchmark results."""
    wall = metrics["wall_s"]
    total_in = metrics["total_in"]
    total_out = metrics["total_out"]
    total = total_in + total_out

    print("\n--- Benchmark Results ---")
    print(f"Policy:                 {policy}")
    print(f"Workload:               {workload}")
    print(f"Total Time:             {wall:.2f} seconds")
    print(f"Total Input Tokens:     {total_in}")
    print(f"Total Output Tokens:    {total_out}")
    print(f"Total Token Throughput: {total / wall:.2f} tokens/s")
    print(f"Output Only Throughput: {total_out / wall:.2f} tokens/s")

    bt = sorted(metrics["batch_times"])
    n = len(bt)
    print("\nBatch Latency Percentiles:")
    print(f"  P50 (median):         {bt[n // 2]:.2f}s")
    print(f"  P95:                  {bt[int(n * 0.95)]:.2f}s")
    print(f"  P99:                  {bt[min(int(n * 0.99), n - 1)]:.2f}s")

    ttft = sorted(metrics["all_ttft"])
    if ttft:
        n = len(ttft)
        print(f"\nTime-to-First-Token (TTFT) — {n} requests:")
        print(f"  Min:                  {min(ttft) * 1000:.1f} ms")
        print(f"  Mean:                 {sum(ttft) / n * 1000:.1f} ms")
        print(f"  P50 (median):         {ttft[n // 2] * 1000:.1f} ms")
        print(f"  P95:                  {ttft[int(n * 0.95)] * 1000:.1f} ms")
        print(f"  P99:                  {ttft[min(int(n * 0.99), n - 1)] * 1000:.1f} ms")
        print(f"  Max:                  {max(ttft) * 1000:.1f} ms")
    else:
        print("\nTTFT: not available (metrics not populated)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> FlexibleArgumentParser:
    p = FlexibleArgumentParser(
        description="Benchmark KV cache eviction policies"
    )

    # Workload selection
    p.add_argument(
        "--workload",
        choices=["uniform", "zipfian", "temporal", "scan-resistant", "helm"],
        default="scan-resistant",
        help="Workload pattern (default: scan-resistant)",
    )

    # Synthetic workload params
    p.add_argument("--num-batches", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--prefix-words", type=int, default=2000)
    p.add_argument("--suffix-words", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=196)
    p.add_argument("--num-prefixes", type=int, default=80)
    p.add_argument("--zipfian-alpha", type=float, default=1.5)
    p.add_argument("--working-set-size", type=int, default=15)

    # HELM params
    p.add_argument("--helm-task", choices=["copa", "piqa", "winogrande"],
                    default="copa")
    p.add_argument("--num-examples", type=int, default=5,
                    help="Few-shot examples for HELM (default: 5)")
    p.add_argument("--num-test", type=int, default=500,
                    help="Test samples for HELM (default: 500)")

    # CPU compat
    p.add_argument("--cpu-kv-cache-space", type=int, default=4,
                    help="CPU KV cache space in GiB (VLLM_CPU_KVCACHE_SPACE)")

    # All vLLM engine args (model, enable-prefix-caching, etc.)
    p = EngineArgs.add_cli_args(p)

    return p


def main():
    parser = create_parser()
    args = parser.parse_args()

    policy = os.environ.get("VLLM_KV_OFFLOAD_POLICY", "unknown")
    _setup_cpu_env(args.cpu_kv_cache_space)

    print(f"Policy:           {policy}")
    print(f"Device:           {'CPU' if IS_CPU else 'GPU'}")
    print(f"Model:            {args.model}")
    print(f"Workload:         {args.workload}")

    # Generate workload
    if args.workload == "uniform":
        batches = workload_uniform(
            args.num_batches, args.batch_size,
            args.prefix_words, args.suffix_words,
        )
    elif args.workload == "helm":
        batches = workload_helm(
            args.helm_task, args.num_examples, args.num_test, args.batch_size,
        )
    else:
        batches = workload_patterned(
            args.workload, args.num_batches, args.batch_size,
            args.prefix_words, args.suffix_words, args.num_prefixes,
            args.zipfian_alpha, args.working_set_size,
        )

    # Build LLM from engine args (same pattern as benchmark_prefix_caching.py)
    engine_args = EngineArgs.from_cli_args(args)
    # Ensure prefix caching and stats are on
    if not engine_args.enable_prefix_caching:
        print("WARNING: --enable-prefix-caching not set; enabling it automatically")
        engine_args.enable_prefix_caching = True
    engine_args.disable_log_stats = False

    print(f"\nLoading {args.model}...")
    llm = LLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Run and report
    workload_label = args.workload
    if args.workload == "helm":
        workload_label = f"helm-{args.helm_task}"

    metrics = run_benchmark(llm, batches, sampling_params)
    print_results(metrics, policy, workload_label)


if __name__ == "__main__":
    main()
