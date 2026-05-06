#!/usr/bin/env bash
# Multi-turn benchmark sweep across all LoRA coupling modes × {lru, sieve, s3fifo}.
# 3 base policies + 15 couplings × 3 inner bases = 48 runs.
set -u
cd /home/dsingh14/vllm

OUTDIR=/home/dsingh14/vllm/lora_sweep_logs/multiturn_all
mkdir -p "$OUTDIR"

PY=.venv/bin/python
BENCH=benchmarks/benchmark_lora_ttft.py

COMMON=(
  --model Qwen/Qwen3-8B
  --lora-path maydixit/qwen3-8b-lora-self-preservation-rl
  --num-adapters 16
  --max-loras 2
  --max-cpu-loras 16
  --max-lora-rank 32
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
  # --gpu-memory-utilization 0.31
  # --kv-offloading-size 0.5
)

BASES=(lru sieve s3fifo)
COUPLINGS=(
  lora_budget
  lora_soft
  lora_loose
  lora_position
  lora_loose_hysteresis
  lora_loose_ghost
  lora_loose_freq
  lora_tight
  lora_hysteresis
  lora_freqweighted
  lora_correlated
  lora_adabudget
  lora_costaware
  lora_ghost
  lora_prefixtree
)

POLICIES=()
# for b in "${BASES[@]}"; do POLICIES+=("$b"); done
for c in "${COUPLINGS[@]}"; do
  for b in "${BASES[@]}"; do POLICIES+=("${c}:${b}"); done
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
    grep -E "^(TTFT|E2E|Total wallclock|Requests measured|Policy:|Multi-turn)" "$log" \
      | sed 's/^/    /'
    grep -oE "Policy \[[^]]+\] block hit rate: [0-9.]+% \([0-9]+/[0-9]+\)" "$log" \
      | tail -1 | sed 's/^/    /'
    grep -oE "Prefix cache hit rate: [0-9.]+%" "$log" | tail -1 | sed 's/^/    GPU /'
  fi
done

echo
echo "All done. Logs in $OUTDIR"
