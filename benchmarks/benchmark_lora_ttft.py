# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request TTFT benchmark for LoRA cache policies."""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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


def gen_multi_turn_prompts(
    num_requests: int,
    num_sessions: int,
    num_adapters: int,
    turns_per_session: int,
    one_shot_fraction: float,
    system_prompt_words: int,
    user_msg_words: int,
    assistant_resp_words: int,
    one_shot_prefix_words: int,
    zipfian_alpha: float,
    seed: int = 42,
) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    system_prompt = " ".join(
        random.Random(seed + 1).choices(_PREFIX_WORDS, k=system_prompt_words)
    )

    session_to_adapter = [s % num_adapters for s in range(num_sessions)]

    sessions: list[list[str]] = []
    for sid in range(num_sessions):
        s_rng = random.Random(seed + sid + 1)
        history = ""
        turn_prompts: list[str] = []
        for _ in range(turns_per_session):
            user_msg = " ".join(s_rng.choices(_PREFIX_WORDS, k=user_msg_words))
            full_prompt = (
                f"{system_prompt}\n{history}\nUser: {user_msg}\nAssistant:"
            )
            turn_prompts.append(full_prompt)
            assistant_resp = " ".join(
                s_rng.choices(_PREFIX_WORDS, k=assistant_resp_words)
            )
            history += f"\nUser: {user_msg}\nAssistant: {assistant_resp}\n"
        sessions.append(turn_prompts)

    session_probs = 1.0 / (np.arange(1, num_sessions + 1) ** zipfian_alpha)
    session_probs /= session_probs.sum()
    session_cursors = [0] * num_sessions

    out: list[tuple[str, int]] = []
    one_shot_count = 0
    session_call_counts: dict[int, int] = defaultdict(int)
    for i in range(num_requests):
        if rng.random() < one_shot_fraction:
            unique = " ".join(
                rng.choices(_PREFIX_WORDS, k=one_shot_prefix_words)
            )
            q = _QUESTIONS[i % len(_QUESTIONS)]
            prompt = (
                f"{system_prompt}\n\n{unique}\n\nQuestion: {q}\nAnswer:"
            )
            out.append((prompt, rng.randrange(num_adapters)))
            one_shot_count += 1
        else:
            sid = int(np_rng.choice(num_sessions, p=session_probs))
            tid = session_cursors[sid] % turns_per_session
            session_cursors[sid] += 1
            session_call_counts[sid] += 1
            out.append((sessions[sid][tid], session_to_adapter[sid]))

    avg_session_tokens = (
        system_prompt_words
        + turns_per_session * (user_msg_words + assistant_resp_words)
    )
    print(
        f"Multi-turn workload: {num_sessions} sessions × "
        f"{turns_per_session} turns ({avg_session_tokens} words deep), "
        f"{one_shot_count}/{num_requests} one-shot "
        f"({one_shot_count / num_requests:.0%}), "
        f"system_prompt={system_prompt_words} words, "
        f"adapters={num_adapters}"
    )
    top_sessions = sorted(
        session_call_counts.items(), key=lambda kv: -kv[1]
    )[:5]
    print(
        f"Hottest 5 sessions (calls): "
        f"{ {f's{sid}->a{session_to_adapter[sid]}': c for sid, c in top_sessions} }"
    )
    return out


def _import_e2e_helpers():
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
        choices=[
            "adapter_thrashing",
            "adapter_locality",
            "mixed_popularity",
            "multi_turn",
        ],
        default="adapter_thrashing",
    )
    p.add_argument("--num-sessions", type=int, default=16,
                   help="Concurrent chat sessions (multi_turn)")
    p.add_argument("--turns-per-session", type=int, default=6,
                   help="Turns per session, prefix grows each turn (multi_turn)")
    p.add_argument("--one-shot-fraction", type=float, default=0.2,
                   help="Fraction of requests that are one-shot scan queries (multi_turn)")
    p.add_argument("--system-prompt-words", type=int, default=400,
                   help="Length of the shared system prompt prepended to all requests (multi_turn)")
    p.add_argument("--user-msg-words", type=int, default=40,
                   help="Words per user turn (multi_turn)")
    p.add_argument("--assistant-resp-words", type=int, default=80,
                   help="Words per assistant turn (multi_turn)")
    p.add_argument("--one-shot-prefix-words", type=int, default=300,
                   help="Words of unique padding for one-shot queries (multi_turn)")
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

    if args.scenario == "multi_turn":
        prompts_and_adapters = gen_multi_turn_prompts(
            num_requests=args.num_requests,
            num_sessions=args.num_sessions,
            num_adapters=args.num_adapters,
            turns_per_session=args.turns_per_session,
            one_shot_fraction=args.one_shot_fraction,
            system_prompt_words=args.system_prompt_words,
            user_msg_words=args.user_msg_words,
            assistant_resp_words=args.assistant_resp_words,
            one_shot_prefix_words=args.one_shot_prefix_words,
            zipfian_alpha=args.zipfian_alpha,
            seed=args.seed,
        )
        prompts_with_lora = [
            (prompt, lora_requests[a]) for prompt, a in prompts_and_adapters
        ]
    else:
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
        disable_log_stats=False,
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
