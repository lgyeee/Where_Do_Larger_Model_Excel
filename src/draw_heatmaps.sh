#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASETS_ALL="OMNI-MATH JEEBENCH-MATH HHMT gpqa-physics JEEBENCH-PHYSICS OlympiadBench-physics JEEBENCH-CHEMISTRY gpqa-chemistry CRUXEVAL-O CRUXEVAL-I"

for pair in qwen3-32b:qwen3-8b gpt-oss-120b:gpt-oss-20b; do
  MODEL_LARGE="${pair%%:*}"
  MODEL_SMALL="${pair##*:}"
  DATASETS_ARG="$DATASETS_ALL"
  case "${MODEL_LARGE}:${MODEL_SMALL}" in
    qwen3-32b:qwen3-8b)
      DIM=8 K=6
      DATASETS_ARG="${DATASETS_ARG//HHMT/}"
      DATASETS_ARG="$(echo "$DATASETS_ARG" | xargs)"
      ;;
    gpt-oss-120b:gpt-oss-20b)
      DIM=4 K=6
      ;;
    *)
      DIM=8 K=6
      ;;
  esac

  echo "=== export heatmap | ${MODEL_LARGE} vs ${MODEL_SMALL} | pcadim=${DIM} k=${K} ==="
  uv run python3 draw_heatmap.py \
    --model_large "$MODEL_LARGE" \
    --model_small "$MODEL_SMALL" \
    --pca_dim "$DIM" \
    --cluster_k "$K" \
    --datasets "$DATASETS_ARG"
done

echo "Done. Outputs: heatmaps_results/*.csv and heatmaps_results/*.png"
