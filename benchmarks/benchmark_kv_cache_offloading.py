import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np

from vllm import LLM, SamplingParams

_PREFIX_WORDS = [
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
]

# Short diverse questions appended after the shared prefix.
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


def generate_shared_prefix(num_words: int) -> str:
    rng = random.Random(42)
    return " ".join(rng.choices(_PREFIX_WORDS, k=num_words))


def generate_unique_padding(rng: random.Random, num_words: int) -> str:
    return " ".join(rng.choices(_PREFIX_WORDS, k=num_words))


def make_batches(
    prefix: str,
    num_batches: int,
    batch_size: int,
    suffix_words: int,
) -> list[list[str]]:
    rng = random.Random(123)
    batches = []
    for b in range(num_batches):
        batch = []
        for i in range(batch_size):
            question = _QUESTIONS[(b * batch_size + i) % len(_QUESTIONS)]
            padding = generate_unique_padding(rng, suffix_words)
            prompt = f"{prefix}\n\n{padding}\n\nQuestion: {question}\nAnswer:"
            batch.append(prompt)
        batches.append(batch)
    return batches


def generate_zipfian_prefix_indices(
    num_prefixes: int, num_requests: int, alpha: float = 1.2, seed: int = 42
) -> list[int]:
    """Generate Zipfian-distributed prefix indices. Higher alpha = more skewed."""
    np.random.seed(seed)
    ranks = np.arange(1, num_prefixes + 1)
    probs = 1.0 / (ranks**alpha)
    probs /= probs.sum()
    return np.random.choice(num_prefixes, size=num_requests, p=probs).tolist()


def generate_temporal_prefix_indices(
    num_prefixes: int,
    num_requests: int,
    working_set_size: int = 10,
    phase_length: int = 100,
    seed: int = 42,
) -> list[int]:
    """
    Generate indices with temporal locality (working set shifts over time).
    80% of requests hit a small working set that moves every phase_length requests.
    Tests how policies adapt to changing access patterns (ARC should excel).
    """
    np.random.seed(seed)
    indices = []
    num_phases = (num_requests + phase_length - 1) // phase_length

    for phase in range(num_phases):
        start_idx = (phase * working_set_size // 2) % max(
            1, num_prefixes - working_set_size
        )
        hot_set = list(
            range(start_idx, min(start_idx + working_set_size, num_prefixes))
        )
        cold_set = [i for i in range(num_prefixes) if i not in hot_set]

        phase_requests = min(phase_length, num_requests - len(indices))
        for _ in range(phase_requests):
            if np.random.random() < 0.8 and hot_set:
                indices.append(np.random.choice(hot_set))
            elif cold_set:
                indices.append(np.random.choice(cold_set))
            elif hot_set:
                indices.append(np.random.choice(hot_set))

    return indices[:num_requests]


def generate_scan_resistant_prefix_indices(
    num_prefixes: int,
    num_requests: int,
    working_set_size: int = 5,
    scan_size: int = 50,
    seed: int = 42,
) -> list[int]:
    """
    Generate scan-resistant workload (70% hot set, 30% sequential scan).

    Tests resistance to cache pollution from one-hit wonders.
    SIEVE should significantly outperform LRU (10-30% higher hit rate).
    """
    np.random.seed(seed)
    indices = []
    hot_set = list(range(working_set_size))
    scan_set = list(
        range(working_set_size, min(working_set_size + scan_size, num_prefixes))
    )

    scan_position = 0
    for i in range(num_requests):
        if i % 10 < 7:
            indices.append(np.random.choice(hot_set))
        else:
            indices.append(scan_set[scan_position % len(scan_set)])
            scan_position += 1

    return indices


def make_batches_with_pattern(
    prefixes: list[str],
    prefix_indices: list[int],
    num_batches: int,
    batch_size: int,
    suffix_words: int,
) -> list[list[str]]:
    rng = random.Random(123)
    batches = []
    request_idx = 0

    for b in range(num_batches):
        batch = []
        for i in range(batch_size):
            if request_idx >= len(prefix_indices):
                break
            prefix_idx = prefix_indices[request_idx]
            prefix = prefixes[prefix_idx]
            question = _QUESTIONS[(b * batch_size + i) % len(_QUESTIONS)]
            padding = generate_unique_padding(rng, suffix_words)
            prompt = f"{prefix}\n\n{padding}\n\nQuestion: {question}\nAnswer:"
            batch.append(prompt)
            request_idx += 1
        if batch:
            batches.append(batch)

    return batches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark KV cache offloading (SIEVE vs LRU)"
    )
    p.add_argument(
        "--policy",
        choices=["lru", "sieve", "arc", "s3fifo"],
        default="lru",
        help="Eviction policy (default: lru)",
    )
    p.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of sequential batches (default: 15)",
    )
    p.add_argument(
        "--batch-size", type=int, default=30, help="Prompts per batch (default: 30)"
    )
    p.add_argument(
        "--prefix-words",
        type=int,
        default=800,
        help="Words in shared prefix (default: 800)",
    )
    p.add_argument(
        "--suffix-words",
        type=int,
        default=400,
        help="Unique padding words per prompt (default: 400)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max generated tokens per prompt (default: 50)",
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
        default=1,
        help="CPU offload size in GiB (default: 1)",
    )
    p.add_argument(
        "--workload-pattern",
        choices=["uniform", "zipfian", "temporal", "scan-resistant"],
        default="uniform",
        help="Access pattern for prefixes (default: uniform)",
    )
    p.add_argument(
        "--num-prefixes",
        type=int,
        default=20,
        help="Number of distinct prefixes for non-uniform patterns (default: 20)",
    )
    p.add_argument(
        "--zipfian-alpha",
        type=float,
        default=1.2,
        help="Zipfian distribution alpha parameter (default: 1.2)",
    )
    p.add_argument(
        "--working-set-size",
        type=int,
        default=10,
        help="Working set size for temporal pattern (default: 10)",
    )
    p.add_argument(
        "--enable-kv-events",
        action="store_true",
        help="Enable KV cache event tracking for detailed statistics",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Model to use for benchmarking (default: Qwen/Qwen2.5-3B)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["VLLM_KV_OFFLOAD_POLICY"] = args.policy

    MODEL_ID = args.model

    print(f"Policy:           {args.policy}")
    print(f"Model:            {MODEL_ID}")
    print(f"Batches:          {args.num_batches} x {args.batch_size} prompts")
    print(f"Workload pattern: {args.workload_pattern}")
    print(f"Shared prefix:    ~{args.prefix_words} words")
    print(f"Unique suffix:    ~{args.suffix_words} words/prompt")
    print(f"CPU offload size: {args.kv_offloading_size} GiB")
    print(f"Max tokens:       {args.max_tokens}")

    if args.workload_pattern == "uniform":
        prefix = generate_shared_prefix(args.prefix_words)
        batches = make_batches(
            prefix,
            args.num_batches,
            args.batch_size,
            args.suffix_words,
        )
    else:
        prefixes = [
            generate_shared_prefix(args.prefix_words + i * 10)
            for i in range(args.num_prefixes)
        ]
        num_requests = args.num_batches * args.batch_size

        if args.workload_pattern == "zipfian":
            print(f"Zipfian alpha:    {args.zipfian_alpha}")
            prefix_indices = generate_zipfian_prefix_indices(
                args.num_prefixes, num_requests, args.zipfian_alpha
            )
        elif args.workload_pattern == "temporal":
            print(f"Working set size: {args.working_set_size}")
            prefix_indices = generate_temporal_prefix_indices(
                args.num_prefixes, num_requests, args.working_set_size
            )
        elif args.workload_pattern == "scan-resistant":
            print(f"Working set size: {args.working_set_size}")
            prefix_indices = generate_scan_resistant_prefix_indices(
                args.num_prefixes, num_requests, args.working_set_size
            )
        else:
            raise ValueError(f"Unknown workload pattern: {args.workload_pattern}")

        batches = make_batches_with_pattern(
            prefixes,
            prefix_indices,
            args.num_batches,
            args.batch_size,
            args.suffix_words,
        )

        prefix_counts = defaultdict(int)
        for idx in prefix_indices:
            prefix_counts[idx] += 1
        print(
            f"Prefix access distribution: {dict(sorted(prefix_counts.items())[:5])}..."
        )

    print(f"\nLoading {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=4096,
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

    total_input_tokens = 0
    total_output_tokens = 0
    batch_times = []

    print(f"\nRunning {args.num_batches} batches...")
    overall_start = time.time()
    for i, batch in enumerate(batches):
        batch_start = time.time()
        outputs = llm.generate(batch, sampling_params)
        batch_elapsed = time.time() - batch_start
        batch_times.append(batch_elapsed)

        batch_in = sum(len(o.prompt_token_ids) for o in outputs)
        batch_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_input_tokens += batch_in
        total_output_tokens += batch_out

        print(
            f"  Batch {i + 1}/{args.num_batches}: "
            f"{batch_elapsed:.2f}s "
            f"({batch_in} in / {batch_out} out)"
        )

    overall_elapsed = time.time() - overall_start

    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / overall_elapsed
    out_throughput = total_output_tokens / overall_elapsed

    print("\n--- Benchmark Results ---")
    print(f"Policy:                 {args.policy}")
    print(f"Workload:               {args.workload_pattern}")
    print(f"Total Time:             {overall_elapsed:.2f} seconds")
    print(f"Total Input Tokens:     {total_input_tokens}")
    print(f"Total Output Tokens:    {total_output_tokens}")
    print(f"Total Token Throughput: {throughput:.2f} tokens/s")
    print(f"Output Only Throughput: {out_throughput:.2f} tokens/s")

    batch_times_sorted = sorted(batch_times)
    p50_idx = len(batch_times_sorted) // 2
    p95_idx = int(len(batch_times_sorted) * 0.95)
    p99_idx = int(len(batch_times_sorted) * 0.99)
    print("\nLatency Percentiles:")
    print(f"  P50 (median):         {batch_times_sorted[p50_idx]:.2f}s")
    print(f"  P95:                  {batch_times_sorted[p95_idx]:.2f}s")
    print(f"  P99:                  {batch_times_sorted[p99_idx]:.2f}s")
