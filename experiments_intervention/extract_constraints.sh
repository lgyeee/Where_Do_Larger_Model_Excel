#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============== Configuration ===============
# DATASETS can be a space-separated list. If not set, fallback to DATASET, else default OMNI-MATH.
if [[ -n "${DATASETS:-}" ]]; then
  DATASETS_LIST=($DATASETS)
elif [[ -n "${DATASET:-}" ]]; then
  DATASETS_LIST=($DATASET)
else
  DATASETS_LIST=(HHMT OMNI-MATH JEEBENCH-MATH gpqa-physics JEEBENCH-PHYSICS OlympiadBench-physics JEEBENCH-CHEMISTRY gpqa-chemistry CRUXEVAL-O CRUXEVAL-I)
fi

# OpenRouter API path (gpt-oss). For Qwen, use vllm/extract_constraints_shard.sh instead.
MODEL_LARGE=${MODEL_LARGE:-gpt-oss-120b}
MODEL_SMALL=${MODEL_SMALL:-gpt-oss-20b}
EXTRACTION_MODEL=${EXTRACTION_MODEL:-gpt-oss-120b}
ADVANTAGE_EXTRACTOR=${ADVANTAGE_EXTRACTOR:-gemini-3-pro}

# reasoning_effort accepted values differ across models:
#   gpt-oss-20b / gpt-oss-120b -> low | medium | high
if [[ -z "${REASONING_EFFORT:-}" ]]; then
  case "$EXTRACTION_MODEL" in
    gpt-oss-20b|gpt-oss-120b|gpt-oss-20b-free) REASONING_EFFORT="low" ;;
    *)                                          REASONING_EFFORT="low" ;;
  esac
fi

for DATASET in "${DATASETS_LIST[@]}"; do
  echo "==> extract constraints: dataset=$DATASET | large=$MODEL_LARGE | small=$MODEL_SMALL | extractor=$EXTRACTION_MODEL | reasoning=$REASONING_EFFORT"
  python3 "$SCRIPT_DIR/extract_constraints.py" \
    --dataset "$DATASET" \
    --model_large "$MODEL_LARGE" \
    --model_small "$MODEL_SMALL" \
    --extraction_model "$EXTRACTION_MODEL" \
    --advantage_extractor "$ADVANTAGE_EXTRACTOR" \
    --reasoning_effort "$REASONING_EFFORT"
done
