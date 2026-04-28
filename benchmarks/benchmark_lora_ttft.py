# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request TTFT (Time-To-First-Token) benchmark for LoRA cache policies.

Uses the LLMEngine.add_request + step() loop directly to time when each
request's first decoded token comes back. Reuses prompt builders from
benchmark_lora_e2e.py.

Usage:
    benchmark_lora_ttft.py --policy lru --scenario adapter_thrashing
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def _import_e2e_helpers():
    """Reuse the prompt + scenario helpers from benchmark_lora_e2e.py."""
    here = Path(__file__).parent / "benchmark_lora_e2e.py"
    spec = spec_from_file_location("benchmark_lora_e2e", here)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument(
        "--lora-path",
        default="kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter",
    )
    p.add_argument("--num-adapters", type=int, default=16)
    p.add_argument("--max-loras", type=int, default=2)
    p.add_argument("--max-cpu-loras", type=int, default=16)
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument(
        "--scenario",
        choices=["adapter_thrashing", "adapter_locality", "mixed_popularity"],
        default="adapter_thrashing",
    )
    p.add_argument("--policy", default=None)
    p.add_argument("--num-requests", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prefix-words", type=int, default=800)
    p.add_argument("--suffix-words", type=int, default=200)
    p.add_argument("--burst-len", type=int, default=8)
    p.add_argument("--zipfian-alpha", type=float, default=1.2)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--kv-offloading-size", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    return s[idx]


def main() -> None:
    args = parse_args()
    helpers = _import_e2e_helpers()

    if args.policy:
        os.environ["VLLM_KV_OFFLOAD_POLICY"] = args.policy
    effective_policy = os.environ.get("VLLM_KV_OFFLOAD_POLICY", "lru")

    print(f"Policy:    {effective_policy}")
    print(f"Scenario:  {args.scenario}")

    local_lora = (
        args.lora_path
        if os.path.isdir(args.lora_path)
        else snapshot_download(repo_id=args.lora_path)
    )
    lora_requests = [
        LoRARequest(f"bench-lora-{i}", i + 1, local_lora)
        for i in range(args.num_adapters)
    ]

    if args.scenario == "adapter_thrashing":
        adapter_indices = helpers.gen_adapter_thrashing(
            args.num_requests, args.num_adapters
        )
    elif args.scenario == "adapter_locality":
        adapter_indices = helpers.gen_adapter_locality(
            args.num_requests, args.num_adapters, args.burst_len, args.seed
        )
    else:
        adapter_indices = helpers.gen_mixed_popularity(
            args.num_requests,
            args.num_adapters,
            args.zipfian_alpha,
            args.seed,
        )

    prompts_with_lora = [
        (
            helpers._build_prompt(
                a, args.prefix_words, args.suffix_words, i
            ),
            lora_requests[a],
        )
        for i, a in enumerate(adapter_indices)
    ]

    print(f"\nLoading {args.model}...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        max_cpu_loras=args.max_cpu_loras,
        kv_offloading_backend="native",
        kv_offloading_size=args.kv_offloading_size,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=True,
    )
    engine = llm.llm_engine

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True
    )

    batches: list[list[tuple[str, LoRARequest]]] = []
    for start in range(0, len(prompts_with_lora), args.batch_size):
        batches.append(prompts_with_lora[start : start + args.batch_size])

    ttfts: list[float] = []
    e2es: list[float] = []
    queue_waits: list[float] = []  # ttft - prefill_walltime, approximate

    print(f"\nRunning {len(batches)} batches (policy={effective_policy})...")
    overall_start = time.time()
    next_id = 0

    for bi, batch in enumerate(batches):
        submit_time: dict[str, float] = {}
        first_token_time: dict[str, float] = {}
        finished_time: dict[str, float] = {}
        active: set[str] = set()
        batch_start = time.time()

        for prompt, lr in batch:
            req_id = f"r{next_id}"
            next_id += 1
            submit_time[req_id] = time.time()
            engine.add_request(
                req_id,
                prompt,
                sampling_params,
                lora_request=lr,
            )
            active.add(req_id)

        while active:
            outs = engine.step()
            now = time.time()
            for o in outs:
                rid = o.request_id
                if rid not in submit_time:
                    continue
                if (
                    rid not in first_token_time
                    and o.outputs
                    and len(o.outputs[0].token_ids) >= 1
                ):
                    first_token_time[rid] = now
                if o.finished:
                    finished_time[rid] = now
                    active.discard(rid)

        # Tally per-request stats for this batch.
        for rid, st in submit_time.items():
            ft = first_token_time.get(rid)
            fn = finished_time.get(rid)
            if ft is not None:
                ttfts.append(ft - st)
            if fn is not None:
                e2es.append(fn - st)
            if ft is not None and fn is not None:
                queue_waits.append(0.0)  # placeholder

        elapsed = time.time() - batch_start
        if ttfts:
            print(
                f"  Batch {bi + 1}/{len(batches)}: "
                f"{elapsed:5.2f}s "
                f"batch P50 TTFT={percentile(ttfts[-len(batch):], 50):.3f}s "
                f"P95 TTFT={percentile(ttfts[-len(batch):], 95):.3f}s",
                flush=True,
            )

    overall = time.time() - overall_start

    print("\n--- TTFT Results ---")
    print(f"Policy:                    {effective_policy}")
    print(f"Scenario:                  {args.scenario}")
    print(f"Total wallclock:           {overall:.2f} s")
    print(f"Requests measured:         {len(ttfts)}")
    print()
    print(
        f"TTFT  P50: {percentile(ttfts, 50):.3f}s  "
        f"P95: {percentile(ttfts, 95):.3f}s  "
        f"P99: {percentile(ttfts, 99):.3f}s  "
        f"mean: {sum(ttfts)/len(ttfts):.3f}s  "
        f"min: {min(ttfts):.3f}s  "
        f"max: {max(ttfts):.3f}s"
    )
    print(
        f"E2E   P50: {percentile(e2es, 50):.3f}s  "
        f"P95: {percentile(e2es, 95):.3f}s  "
        f"P99: {percentile(e2es, 99):.3f}s  "
        f"mean: {sum(e2es)/len(e2es):.3f}s"
    )


if __name__ == "__main__":
    main()
