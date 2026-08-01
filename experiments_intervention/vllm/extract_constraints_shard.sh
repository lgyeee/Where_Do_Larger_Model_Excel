#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Trap SIGINT/SIGTERM and kill all child jobs ─────────────────────
trap 'echo; echo "⚡️ Interrupted—killing all shards..."; jobs -p | xargs -r kill; exit 1' SIGINT SIGTERM

# ─── Configuration ──────────────────────────────────────────────────
read -ra EXTRACTION_MODELS <<< "${EXTRACTION_MODEL_OVERRIDE:-qwen3-8b}"
read -ra DATASETS <<< "${DATASET_OVERRIDE:-OMNI-MATH JEEBENCH-MATH gpqa-physics JEEBENCH-PHYSICS OlympiadBench-physics JEEBENCH-CHEMISTRY gpqa-chemistry CRUXEVAL-O CRUXEVAL-I}"
TOTAL_GPUS="${TOTAL_GPUS:-1}"
DEFAULT_N_SAMPLE="${DEFAULT_N_SAMPLE:-1}"
MODEL_LARGE="${MODEL_LARGE:-qwen3-32b}"
MODEL_SMALL="${MODEL_SMALL:-qwen3-8b}"
ADVANTAGE_EXTRACTOR="${ADVANTAGE_EXTRACTOR:-gemini-3-pro}"
LOG_DIR="$SCRIPT_DIR/logs/extract_constraints"
mkdir -p "$LOG_DIR"

# ─── Main Loop ─────────────────────────────────────────────────────
for EXTRACTION_MODEL in "${EXTRACTION_MODELS[@]}"; do
  if [[ "$EXTRACTION_MODEL" == *"qwen3-32b"* ]]; then
    TP=2
  elif [[ "$EXTRACTION_MODEL" == *"8b"* ]]; then
    TP=1
  else
    TP=1
  fi

  NUM_SHARDS=$(( TOTAL_GPUS / TP ))
  echo "▶ Extraction model: $EXTRACTION_MODEL (TP=$TP → $NUM_SHARDS shards over $TOTAL_GPUS GPUs)"

  for DATASET in "${DATASETS[@]}"; do
    N_SAMPLE=$DEFAULT_N_SAMPLE

    echo "
── Dataset: $DATASET | Extraction model: $EXTRACTION_MODEL | N_SAMPLE=$N_SAMPLE ──"

    for sid in $(seq 0 $(( NUM_SHARDS - 1 ))); do
      gpu_start=$(( sid * TP ))
      gpu_end=$(( gpu_start + TP - 1 ))
      GPUS=$(seq -s, $gpu_start $gpu_end)

      logfile="$LOG_DIR/${EXTRACTION_MODEL}__${DATASET}_shard${sid}.log"
      echo "  → [shard $sid] GPUs=[$GPUS] → $logfile"

      CUDA_VISIBLE_DEVICES=$GPUS \
        python3 "$SCRIPT_DIR/extract_constraints_shard.py" \
          --extraction_model "$EXTRACTION_MODEL" \
          --dataset "$DATASET" \
          --n_sample "$N_SAMPLE" \
          --tensor_parallel_size "$TP" \
          --shard-id "$sid" \
          --num-shards "$NUM_SHARDS" \
          --model_large "$MODEL_LARGE" \
          --model_small "$MODEL_SMALL" \
          --advantage_extractor "$ADVANTAGE_EXTRACTOR" \
        &> "$logfile" &
    done

    echo "  ⇢ Waiting for all $NUM_SHARDS shards to complete…"
    wait
    echo "  ✓ All shards done for $DATASET / $EXTRACTION_MODEL"

    echo "  ⇢ Gathering results…"
    python3 "$SCRIPT_DIR/gather_constraints_shard.py" \
      --dataset "$DATASET" \
      --extraction_model "$EXTRACTION_MODEL" \
      --num-shards "$NUM_SHARDS" \
      --model_large "$MODEL_LARGE" \
      --model_small "$MODEL_SMALL"
  done
  echo
done
