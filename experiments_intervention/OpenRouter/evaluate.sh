#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${DATASETS:-}" ]]; then
  DATASETS_LIST=($DATASETS)
elif [[ -n "${DATASET:-}" ]]; then
  DATASETS_LIST=($DATASET)
else
  DATASETS_LIST=(OMNI-MATH)
fi

MODEL_LARGE=${MODEL_LARGE:-gpt-oss-120b}
MODEL_SMALL=${MODEL_SMALL:-gpt-oss-20b}
EXTRACTION_MODEL=${EXTRACTION_MODEL:-gpt-oss-120b}
ADVANTAGE_EXTRACTOR=${ADVANTAGE_EXTRACTOR:-gemini-3-pro}
N_SAMPLE=${N_SAMPLE:-10}

if [[ -n "${MODES:-}" ]]; then
  MODES_LIST=($MODES)
elif [[ -n "${MODE:-}" ]]; then
  MODES_LIST=($MODE)
else
  MODES_LIST=(slm-guided llm-guided)
fi

if [[ -z "${REASONING_EFFORT:-}" ]]; then
  case "$MODEL_SMALL" in
    qwen3-32b)                                  REASONING_EFFORT="default" ;;
    gpt-oss-20b|gpt-oss-120b|gpt-oss-20b-free) REASONING_EFFORT="high" ;;
    *)                                          REASONING_EFFORT="high" ;;
  esac
fi

for MODE in "${MODES_LIST[@]}"; do
  for DATASET in "${DATASETS_LIST[@]}"; do
    echo "==> openrouter evaluate: dataset=$DATASET | mode=$MODE | small=$MODEL_SMALL | large=$MODEL_LARGE | extractor=$EXTRACTION_MODEL | n_sample=$N_SAMPLE | reasoning=$REASONING_EFFORT"
    python3 "$SCRIPT_DIR/evaluate.py" \
      --dataset "$DATASET" \
      --model_large "$MODEL_LARGE" \
      --model_small "$MODEL_SMALL" \
      --extraction_model "$EXTRACTION_MODEL" \
      --advantage_extractor "$ADVANTAGE_EXTRACTOR" \
      --mode "$MODE" \
      --n_sample "$N_SAMPLE" \
      --reasoning_effort "$REASONING_EFFORT"
  done
done
