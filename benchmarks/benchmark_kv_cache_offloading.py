import time
import random
from vllm import LLM, SamplingParams


def generate_dummy_prompt(length):
    """Generates a random prompt of a specific word length to prevent caching."""
    words = [
        "apple",
        "system",
        "data",
        "compute",
        "matrix",
        "tensor",
        "cache",
        "block",
        "memory",
        "swap",
    ]
    return (
        "Return only the number of times the word 'apple' occurs in this string: "
        + " ".join(random.choices(words, k=length))
    )


if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-3B"

    print(f"Loading {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.6,  # Leave enough room to force KV cache to exhaust early
        enforce_eager=True,
        max_model_len=4096,  # Increase max length to allow larger KV allocations
        kv_offloading_backend="native",
        kv_offloading_size=8,  # Gigabytes to offload
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,  # Disable to force unique KV generation per sequence
    )

    # 2. Create a heterogeneous workload
    # Mix of very long prompts (heavy prefill KV) and shorter ones
    print("Generating heterogeneous prompts...")
    prompt_lengths = [random.randint(500, 3500) for _ in range(50)]
    prompts = [generate_dummy_prompt(length) for length in prompt_lengths]

    # Force long generation to stress decode KV cache growth
    sampling_params = SamplingParams(temperature=0.2, max_tokens=500, ignore_eos=True)

    print(f"Starting generation with {len(prompts)} sequences...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    # 3. Calculate detailed metrics
    total_input_tokens = sum(len(out.prompt_token_ids) for out in outputs)
    total_output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    total_tokens = total_input_tokens + total_output_tokens

    throughput = total_tokens / elapsed
    output_throughput = total_output_tokens / elapsed

    print("\n--- Benchmark Results ---")
    print(f"Total Time:             {elapsed:.2f} seconds")
    print(f"Total Input Tokens:     {total_input_tokens}")
    print(f"Total Output Tokens:    {total_output_tokens}")
    print(f"Total Token Throughput: {throughput:.2f} tokens/s")
    print(f"Output Only Throughput: {output_throughput:.2f} tokens/s")
