"""Sanity-check: repeat the same prompts, see if CPU offload hits."""
import os
os.environ["VLLM_KV_OFFLOAD_POLICY"] = "lru"

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=2048,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16,
        max_cpu_loras=2,
        kv_offloading_backend="native",
        kv_offloading_size=0.05,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=True,
    )
    lora_path = snapshot_download(
        repo_id="kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter"
    )
    lr = LoRARequest("hit-test", 1, lora_path)
    sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)

    long_prompt = (
        "optimization throughput latency bandwidth pipeline scheduler "
        "prefetch eviction allocation partition replication consistency "
        "transaction isolation concurrency parallelism synchronization "
        "deadlock mutex processor register instruction operand " * 30
    )

    print("--- Phase 1: warm with 20 unique prompts ---", flush=True)
    prompts1 = [long_prompt + f" Q: query {i}?" for i in range(20)]
    llm.generate(prompts1, sp, lora_request=[lr] * 20, use_tqdm=False)

    print("--- Phase 2: repeat same prompts, expect hits ---", flush=True)
    llm.generate(prompts1[:5], sp, lora_request=[lr] * 5, use_tqdm=False)
    print("--- Phase 2 done ---", flush=True)


if __name__ == "__main__":
    main()
