#!/usr/bin/env bash
set -euo pipefail

# =============== Configuration ===============
# DATASETS can be a space-separated list. If not set, fallback to DATASET, else default OMNI-MATH.
if [[ -n "${DATASETS:-}" ]]; then
  DATASETS_LIST=($DATASETS)
elif [[ -n "${DATASET:-}" ]]; then
  DATASETS_LIST=($DATASET)
else
  DATASETS_LIST=(OMNI-MATH)
fi
# HHMT JEEBENCH-MATH gpqa-physics JEEBENCH-PHYSICS OlympiadBench-physics
# JEEBENCH-CHEMISTRY gpqa-chemistry CRUXEVAL-O CRUXEVAL-I

MODEL=${MODEL:-gpt-oss-20b}
N_SAMPLE=${N_SAMPLE:-10}
REASONING_EFFORT=${REASONING_EFFORT:-high}


for DATASET in "${DATASETS_LIST[@]}"; do
  echo "==> create batch: dataset=$DATASET | model=$MODEL | n_sample=$N_SAMPLE | reasoning=$REASONING_EFFORT"
  python3 gen_batch_requests.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --n_sample "$N_SAMPLE" \
    --reasoning_effort "$REASONING_EFFORT"
done

for DATASET in "${DATASETS_LIST[@]}"; do
  echo "==> upload batch: dataset=$DATASET | model=$MODEL | n_sample=$N_SAMPLE | reasoning=$REASONING_EFFORT"
  python3 upload_batch.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --n_sample "$N_SAMPLE"
done


# read from batch_files/input_file_id.jsonl and upload the batch
for DATASET in "${DATASETS_LIST[@]}"; do
  echo "==> launch batch jobs: dataset=$DATASET | model=$MODEL | n_sample=$N_SAMPLE"
  python3 launch_batch.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --n_sample "$N_SAMPLE"
done

