#!/bin/bash
set -e
cd /home/zhuang/vllm
PY=/home/zhuang/fal/bin/python
OUT=/tmp/lora_sweep_logs/ttft
mkdir -p "$OUT"

for policy in "lru" "lora_budget:s3fifo"; do
  for scenario in "adapter_thrashing" "adapter_locality" "mixed_popularity"; do
    safe=$(echo "$policy" | tr ':' '_')
    log="$OUT/${safe}__${scenario}.log"
    echo "=== $policy / $scenario ==="
    $PY benchmarks/benchmark_lora_ttft.py \
      --policy "$policy" --scenario "$scenario" \
      --num-requests 80 --batch-size 8 \
      --num-adapters 16 --max-loras 2 --max-cpu-loras 16 \
      --prefix-words 800 --suffix-words 200 \
      --max-tokens 8 --max-model-len 2048 \
      --gpu-memory-utilization 0.5 --kv-offloading-size 0.5 \
      > "$log" 2>&1
    grep -E "^TTFT|^E2E|^Total wall|^Policy:|^Scenario:" "$log"
  done
done
echo "DONE"
