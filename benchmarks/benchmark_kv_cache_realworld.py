"""
Benchmark KV cache eviction policies on HELM few-shot datasets.

Usage:
    VLLM_KV_OFFLOAD_POLICY=lru python benchmarks/benchmark_kv_cache_realworld.py --task copa
"""

import argparse
import os
import time
from typing import List

from datasets import load_dataset

from vllm import LLM, SamplingParams


def load_helm_fewshot(
    task: str = "copa",
    num_examples: int = 5,
    num_test: int = 100,
) -> List[str]:
    """
    Load HELM-style few-shot prompts. All prompts share the same examples prefix.

    Returns list of prompts with shared few-shot examples.
    """
    task_datasets = {
        "copa": ("super_glue", "copa"),
        "piqa": ("piqa", None),
        "winogrande": ("winogrande", "winogrande_xl"),
    }

    if task not in task_datasets:
        raise ValueError(f"Unknown task: {task}. Choose from {list(task_datasets.keys())}")

    dataset_name, config = task_datasets[task]

    if config:
        train_data = load_dataset(dataset_name, config, split="train")
        test_data = load_dataset(dataset_name, config, split="validation")
    else:
        train_data = load_dataset(dataset_name, split="train")
        test_data = load_dataset(dataset_name, split="validation")

    # Build few-shot examples prefix
    examples_prefix = f"Answer the following questions.\n\n"

    for i, example in enumerate(train_data):
        if i >= num_examples:
            break

        if task == "copa":
            premise = example["premise"]
            choice1 = example["choice1"]
            choice2 = example["choice2"]
            question = example["question"]
            label = example["label"]
            answer = choice1 if label == 0 else choice2

            examples_prefix += f"Premise: {premise}\n"
            examples_prefix += f"Question: What is the {question}?\n"
            examples_prefix += f"Choice 1: {choice1}\n"
            examples_prefix += f"Choice 2: {choice2}\n"
            examples_prefix += f"Answer: {answer}\n\n"

        elif task == "piqa":
            goal = example["goal"]
            sol1 = example["sol1"]
            sol2 = example["sol2"]
            label = example["label"]
            answer = sol1 if label == 0 else sol2

            examples_prefix += f"Goal: {goal}\n"
            examples_prefix += f"Solution: {answer}\n\n"

        elif task == "winogrande":
            sentence = example["sentence"]
            option1 = example["option1"]
            option2 = example["option2"]
            answer = example["answer"]
            correct = option1 if answer == "1" else option2

            examples_prefix += f"Sentence: {sentence}\n"
            examples_prefix += f"Answer: {correct}\n\n"

    # Build test prompts with shared prefix
    prompts = []
    for i, example in enumerate(test_data):
        if i >= num_test:
            break

        if task == "copa":
            premise = example["premise"]
            choice1 = example["choice1"]
            choice2 = example["choice2"]
            question = example["question"]

            prompt = examples_prefix
            prompt += f"Premise: {premise}\n"
            prompt += f"Question: What is the {question}?\n"
            prompt += f"Choice 1: {choice1}\n"
            prompt += f"Choice 2: {choice2}\n"
            prompt += f"Answer:"

        elif task == "piqa":
            goal = example["goal"]
            prompt = examples_prefix + f"Goal: {goal}\nSolution:"

        elif task == "winogrande":
            sentence = example["sentence"]
            prompt = examples_prefix + f"Sentence: {sentence}\nAnswer:"

        prompts.append(prompt)

    return prompts




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark HELM few-shot tasks")

    p.add_argument(
        "--task",
        choices=["copa", "piqa", "winogrande"],
        default="copa",
        help="HELM task (default: copa)",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Few-shot examples (default: 5)",
    )
    p.add_argument(
        "--num-test",
        type=int,
        default=500,
        help="Test samples (default: 500)",
    )

    # Model options
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Model to use (default: Qwen/Qwen2.5-3B)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max generated tokens (default: 50)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
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

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    policy = os.environ.get("VLLM_KV_OFFLOAD_POLICY", "unknown")

    print(f"Policy:  {policy}")
    print(f"Task:    {args.task}")
    print(f"Model:   {args.model}")

    # Load HELM dataset
    print(f"\nLoading HELM {args.task} dataset...")
    prompts = load_helm_fewshot(
        args.task,
        args.num_examples,
        args.num_test,
    )
    print(f"Few-shot examples: {args.num_examples}")
    print(f"Test prompts: {len(prompts)}")
    print(f"All prompts share {args.num_examples}-shot prefix")

    # Initialize model
    print(f"\nLoading {args.model}...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=4096,
        kv_offloading_backend="native",
        kv_offloading_size=args.kv_offloading_size,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=True,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Run benchmark in batches
    total_input_tokens = 0
    total_output_tokens = 0
    batch_times: list[float] = []
    all_ttft: list[float] = []

    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    print(f"\nRunning {num_batches} batches (batch_size={args.batch_size})...")

    overall_start = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]

        batch_start = time.time()
        outputs = llm.generate(batch, sampling_params)
        batch_elapsed = time.time() - batch_start
        batch_times.append(batch_elapsed)

        batch_in = sum(len(o.prompt_token_ids or []) for o in outputs)
        batch_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_input_tokens += batch_in
        total_output_tokens += batch_out

        # Collect per-request TTFT from vLLM's built-in metrics.
        batch_ttft: list[float] = []
        for o in outputs:
            if o.metrics is not None and o.metrics.first_token_latency > 0:
                batch_ttft.append(o.metrics.first_token_latency)
        all_ttft.extend(batch_ttft)

        ttft_str = ""
        if batch_ttft:
            mean_ttft = sum(batch_ttft) / len(batch_ttft)
            ttft_str = f"  ttft_mean={mean_ttft * 1000:.0f}ms"

        batch_num = i // args.batch_size + 1
        print(
            f"  Batch {batch_num}/{num_batches}: "
            f"{batch_elapsed:.2f}s "
            f"({batch_in} in / {batch_out} out)"
            f"{ttft_str}"
        )

    overall_elapsed = time.time() - overall_start
    # Results
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / overall_elapsed
    out_throughput = total_output_tokens / overall_elapsed

    print("\n--- Benchmark Results ---")
    print(f"Policy:                 {policy}")
    print(f"Task:                   {args.task}")
    print(f"Total Time:             {overall_elapsed:.2f} seconds")
    print(f"Total Input Tokens:     {total_input_tokens}")
    print(f"Total Output Tokens:    {total_output_tokens}")
    print(f"Total Token Throughput: {throughput:.2f} tokens/s")
    print(f"Output Only Throughput: {out_throughput:.2f} tokens/s")

    bt = sorted(batch_times)
    p50 = bt[len(bt) // 2]
    p95 = bt[int(len(bt) * 0.95)]
    p99 = bt[min(int(len(bt) * 0.99), len(bt) - 1)]
    print("\nBatch Latency Percentiles:")
    print(f"  P50 (median):         {p50:.2f}s")
    print(f"  P95:                  {p95:.2f}s")
    print(f"  P99:                  {p99:.2f}s")

    if all_ttft:
        tf = sorted(all_ttft)
        n = len(tf)
        print(f"\nTime-to-First-Token (TTFT) — {n} requests:")
        print(f"  Mean:                 {sum(tf) / n * 1000:.1f} ms")
        print(f"  P50 (median):         {tf[n // 2] * 1000:.1f} ms")
        print(f"  P95:                  {tf[int(n * 0.95)] * 1000:.1f} ms")
        print(f"  P99:                  {tf[min(int(n * 0.99), n - 1)] * 1000:.1f} ms")
    else:
        print("\nTTFT: not available (metrics not populated)")
