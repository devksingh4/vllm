#!/usr/bin/env bash
# Multi-turn benchmark sweep across base + loose-coupling LoRA policies.
set -u
cd /home/dsingh/source/devksingh4/vllm

OUTDIR=/home/dsingh/source/devksingh4/vllm/lora_sweep_logs/multiturn_loose
mkdir -p "$OUTDIR"

PY=.venv/bin/python
BENCH=benchmarks/benchmark_lora_ttft.py

COMMON=(
  --model Qwen/Qwen2.5-1.5B
  --lora-path kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter
  --num-adapters 16
  --max-loras 2
  --max-cpu-loras 16
  --scenario multi_turn
  --num-requests 1280
  --batch-size 128
  --num-sessions 50
  --turns-per-session 6
  --one-shot-fraction 0.25
  --system-prompt-words 1500
  --user-msg-words 80
  --assistant-resp-words 250
  --zipfian-alpha 1.2
  --max-tokens 8
  --max-model-len 8192
  --gpu-memory-utilization 0.5
  --kv-offloading-size 0.5
)

BASES=(lru sieve s3fifo)
LOOSE=(lora_loose lora_loose_hysteresis lora_loose_freq lora_loose_ghost)

POLICIES=()
for b in "${BASES[@]}"; do POLICIES+=("$b"); done
for l in "${LOOSE[@]}"; do
  for b in "${BASES[@]}"; do POLICIES+=("${l}:${b}"); done
done

echo "Total policies: ${#POLICIES[@]}"
printf '  %s\n' "${POLICIES[@]}"

i=0
for p in "${POLICIES[@]}"; do
  i=$((i+1))
  safe=${p//:/__}
  log="$OUTDIR/${safe}.log"
  echo
  echo "=== [$i/${#POLICIES[@]}] policy=$p -> $log ==="
  start=$(date +%s)
  "$PY" "$BENCH" --policy "$p" "${COMMON[@]}" >"$log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -ne 0 ]; then
    echo "  FAILED (rc=$rc, ${dur}s)"
  else
    echo "  ok (${dur}s)"
    grep -E "^(TTFT|E2E|Total wallclock|Requests measured|Policy:|Multi-turn)" "$log" | sed 's/^/    /'
  fi
done

echo
echo "All done. Logs in $OUTDIR"
