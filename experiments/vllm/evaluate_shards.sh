#!/usr/bin/env bash
set -euo pipefail

# ─── Trap SIGINT/SIGTERM and kill all child jobs ─────────────────────
trap 'echo; echo "⚡️ Interrupted—killing all shards..."; jobs -p | xargs -r kill; exit 1' SIGINT SIGTERM

# ─── Configuration ──────────────────────────────────────────────────
read -ra MODELS <<< "${MODEL_OVERRIDE:-qwen3-8b qwen3-32b}"
read -ra DATASETS <<< "${DATASET_OVERRIDE:-OMNI-MATH JEEBENCH-MATH HHMT gpqa-physics JEEBENCH-PHYSICS OlympiadBench-physics JEEBENCH-CHEMISTRY gpqa-chemistry CRUXEVAL-O CRUXEVAL-I}"
TOTAL_GPUS="${TOTAL_GPUS:-4}"
DEFAULT_N_SAMPLE="${DEFAULT_N_SAMPLE:-10}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"


# ─── Main Loop ─────────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
  # pick tensor-parallel size based on model
  if [[ "$MODEL" == *"qwen3-32b"* ]]; then
    TP=2
  elif [[ "$MODEL" == *"8b"* ]]; then
    TP=1
  else
    TP=1
  fi
  
  NUM_SHARDS=$(( TOTAL_GPUS / TP ))
  echo "▶ Model: $MODEL (TP=$TP → $NUM_SHARDS shards over $TOTAL_GPUS GPUs)"

  for DATASET in "${DATASETS[@]}"; do
    N_SAMPLE=$DEFAULT_N_SAMPLE

    echo "
── Dataset: $DATASET | Model: $MODEL | N_SAMPLE=$N_SAMPLE ──"

      # launch each shard
      for sid in $(seq 0 $(( NUM_SHARDS - 1 ))); do
        gpu_start=$(( sid * TP ))
        gpu_end=$(( gpu_start + TP - 1 ))
        GPUS=$(seq -s, $gpu_start $gpu_end)

        logfile="$LOG_DIR/${MODEL}__${DATASET}_shard${sid}.log"
        echo "  → [shard $sid] GPUs=[$GPUS] → $logfile"

        CUDA_VISIBLE_DEVICES=$GPUS \
          python3 evaluate_reasoning_models_shard.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --n_sample "$N_SAMPLE" \
            --tensor_parallel_size "$TP" \
            --shard-id "$sid" \
            --num-shards "$NUM_SHARDS" \
        &> "$logfile" &
      done

      # wait for all shards of this run to finish
      echo "  ⇢ Waiting for all $NUM_SHARDS shards to complete…"
      wait
      echo "  ✓ All shards done for $DATASET / $MODEL"

      # gather results
      echo "  ⇢ Gathering results…"
      python3 gather_results.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --n_sample "$N_SAMPLE" \
        --num-shards "$NUM_SHARDS"
  done
  echo
done
