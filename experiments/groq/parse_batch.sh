#!/usr/bin/env bash
set -euo pipefail

# =============== Configuration ===============
if [[ -n "${DATASETS:-}" ]]; then
  DATASETS_LIST=($DATASETS)
elif [[ -n "${DATASET:-}" ]]; then
  DATASETS_LIST=($DATASET)
else
  DATASETS_LIST=(HHMT)
fi

MODEL=${MODEL:-gpt-oss-20b}
N_SAMPLE=${N_SAMPLE:-1}

# =============== Retrieve batch results ===============
echo "==> Retrieving batch results..."
python3 reterive_batch_results.py

# =============== Parse batch results ===============
for DATASET in "${DATASETS_LIST[@]}"; do
  echo "==> parse: dataset=$DATASET | model=$MODEL | n_sample=$N_SAMPLE"
  python3 parse_batch_reasoning.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --n_sample "$N_SAMPLE"
done
