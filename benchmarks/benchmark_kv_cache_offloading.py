import argparse
import os
import random
import time

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark KV cache offloading (SIEVE vs LRU)"
    )
    p.add_argument(
        "--policy",
        choices=["lru", "sieve", "arc"],
        default="lru",
        help="Eviction policy (default: lru)",
    )
    p.add_argument(
        "--num-batches",
        type=int,
        default=15,
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["VLLM_KV_OFFLOAD_POLICY"] = args.policy

    MODEL_ID = "Qwen/Qwen2.5-3B"

    print(f"Policy:           {args.policy}")
    print(f"Model:            {MODEL_ID}")
    print(f"Batches:          {args.num_batches} x {args.batch_size} prompts")
    print(f"Shared prefix:    ~{args.prefix_words} words")
    print(f"Unique suffix:    ~{args.suffix_words} words/prompt")
    print(f"CPU offload size: {args.kv_offloading_size} GiB")
    print(f"Max tokens:       {args.max_tokens}")

    prefix = generate_shared_prefix(args.prefix_words)
    batches = make_batches(
        prefix,
        args.num_batches,
        args.batch_size,
        args.suffix_words,
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

    print(f"\nRunning {args.num_batches} batches...")
    overall_start = time.time()
    for i, batch in enumerate(batches):
        batch_start = time.time()
        outputs = llm.generate(batch, sampling_params)
        batch_elapsed = time.time() - batch_start

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
    print(f"Total Time:             {overall_elapsed:.2f} seconds")
    print(f"Total Input Tokens:     {total_input_tokens}")
    print(f"Total Output Tokens:    {total_output_tokens}")
    print(f"Total Token Throughput: {throughput:.2f} tokens/s")
    print(f"Output Only Throughput: {out_throughput:.2f} tokens/s")
