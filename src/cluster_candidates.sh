#!/usr/bin/env bash
set -euo pipefail

# ─── Trap SIGINT/SIGTERM and kill all child jobs ─────────────────────
trap 'echo; echo "⚡️ Interrupted—killing clustering..."; jobs -p | xargs -r kill; exit 1' SIGINT SIGTERM

# ─── Configuration (override via env) ────────────────────────────────
read -ra DATASETS <<< "${DATASET_OVERRIDE:-}"

LOG_DIR="clustering_logs"
mkdir -p "$LOG_DIR"

# ─── Build dataset list if not provided ─────────────────────────────
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(
    "OMNI-MATH" "JEEBENCH-MATH" "HHMT"
    "gpqa-physics" "JEEBENCH-PHYSICS" "OlympiadBench-physics"
    "JEEBENCH-CHEMISTRY" "gpqa-chemistry"
    "CRUXEVAL-O" "CRUXEVAL-I"
  )
fi

DATASETS_ARG="${DATASETS[*]}"

run_fixed_k() {
  local model_large="$1"
  local model_small="$2"
  shift 2
  local pairs=("$@")
  local datasets_arg="$DATASETS_ARG"
  if [[ "$model_large" == "qwen3-32b" && "$model_small" == "qwen3-8b" ]]; then
    datasets_arg="${datasets_arg//HHMT/}"
    datasets_arg="$(echo "$datasets_arg" | xargs)"
  fi
  echo
  echo "── Model pair | LLM=$model_large | SLM=$model_small |"
  for pair in "${pairs[@]}"; do
    local d="${pair%%:*}"
    local k="${pair##*:}"
    local log_tag="${model_large}_vs_${model_small}_pcadim${d}_k${k}"
    echo "   ▶ clustering  pcadim=$d  k=$k  (log: ${log_tag}.log)"
    uv run --python 3.12 python adv_cluster.py \
      --datasets "$datasets_arg" \
      --model_large "$model_large" \
      --model_small "$model_small" \
      --use_cached_embeddings \
      --dr_method PCA \
      --dr_dim "$d" \
      --cluster_method KMeans \
      --cluster_k "$k" \
      > "$LOG_DIR/${log_tag}.log" 2>&1
  done
}

# Fixed-k candidates
run_fixed_k "qwen3-32b" "qwen3-8b" \
  "4:5" "4:6" "4:4" "4:3" \
  "8:7" "8:8" "8:9" "8:6"
run_fixed_k "gpt-oss-120b" "gpt-oss-20b" \
  "4:2" "4:5" "4:3" "4:6" \
  "8:2" "8:13" "8:14" "8:12" "8:3"
