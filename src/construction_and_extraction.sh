#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────
read -ra SUBJECTS <<< "${SUBJECTS:-math physics chemistry programming}"
read -ra MODEL_PAIRS <<< "${MODEL_PAIRS:-qwen3-32b:qwen3-8b gpt-oss-120b:gpt-oss-20b}"

GAP_VALUE="${GAP_VALUE:-0.6}"                 # e.g. 0.6
N_SAMPLE="${N_SAMPLE:-10}"                    # e.g. 10
ADVANTAGE_EXTRACTOR="${ADVANTAGE_EXTRACTOR:-gemini-3-pro}"  # maps to --advantage_extractor
# Set FROM_HF=1 to download HF traces into eval_outputs/ first (default: use local eval_outputs only).
FROM_HF="${FROM_HF:-0}"

LOG_DIR="api_logs"
mkdir -p "$LOG_DIR"

# ─── Main Loop ────────────────────────────────────────────────────
for SUBJECT in "${SUBJECTS[@]}"; do

  case "$SUBJECT" in
    programming)
      DATASETS=("CRUXEVAL-I" "CRUXEVAL-O")
      ;;
    math)
      DATASETS=( "OMNI-MATH" "HHMT" "JEEBENCH-MATH" ) 
      ;;
    chemistry)
      DATASETS=("JEEBENCH-CHEMISTRY" "gpqa-chemistry")
      ;;
    physics)
      DATASETS=("OlympiadBench-physics" "gpqa-physics" "JEEBENCH-PHYSICS") 
      ;;
    *)
      echo "Unknown subject: $SUBJECT"
      exit 1
      ;;
  esac

  mkdir -p "$LOG_DIR/$SUBJECT"

  for DATASET in "${DATASETS[@]}"; do
    for PAIR in "${MODEL_PAIRS[@]}"; do
      IFS=':' read -r MODEL_L MODEL_S <<< "$PAIR"
      if [[ -z "$MODEL_L" || -z "$MODEL_S" ]]; then
        echo "Invalid MODEL_PAIRS entry: '$PAIR' (expected large:small)"
        exit 1
      fi

        LOG_FILE="${LOG_DIR}/${SUBJECT}/${DATASET}_${MODEL_L}_vs_${MODEL_S}.log"
        echo "Logging to $LOG_FILE"

        cmd=(
          python3 -u construction_and_extraction.py
          --dataset "$DATASET"
          --model_large "$MODEL_L"
          --model_small "$MODEL_S"
          --advantage_extractor "$ADVANTAGE_EXTRACTOR"
          --n-sample "$N_SAMPLE"
          --gap-value "$GAP_VALUE"
        )
        if [[ "$FROM_HF" == "1" ]]; then
          cmd+=(--from-hf)
        fi
        "${cmd[@]}" 2>&1 | tee "$LOG_FILE"

    done
  done
done
