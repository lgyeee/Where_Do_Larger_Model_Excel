#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ANALYSIS_MODEL="${ANALYSIS_MODEL:-openai/gpt-5.2}"
DATASETS_ARG="${SYNTH_DATASETS:-}"
LOG_DIR="${LOG_DIR:-clustering_logs}"
mkdir -p "$LOG_DIR"

run_pair() {
  local model_large="$1"
  local model_small="$2"
  shift 2
  local pairs=("$@")
  local datasets_arg="$DATASETS_ARG"
  if [[ -z "$datasets_arg" ]]; then
    datasets_arg="OMNI-MATH,JEEBENCH-MATH,HHMT,gpqa-physics,JEEBENCH-PHYSICS,OlympiadBench-physics,JEEBENCH-CHEMISTRY,gpqa-chemistry,CRUXEVAL-O,CRUXEVAL-I"
  fi
  if [[ "$model_large" == "qwen3-32b" && "$model_small" == "qwen3-8b" ]]; then
    datasets_arg="${datasets_arg//HHMT/}"
    datasets_arg="$(echo "$datasets_arg" | xargs)"
    datasets_arg="${datasets_arg// /,}"
  fi
  for pair in "${pairs[@]}"; do
    local dim="${pair%%:*}"
    local k="${pair##*:}"
    echo "── summarizing  pcadim=$dim  k=$k  | $model_large vs $model_small"
    extra_args=()
    [[ -n "$datasets_arg" ]] && extra_args+=(--datasets "$datasets_arg")
    uv run --python 3.12 python summarizing.py \
      "${extra_args[@]}" \
      --model_large "$model_large" \
      --model_small "$model_small" \
      --cluster_method KMeans \
      --pca_dim "$dim" \
      --cluster_k "$k" \
      --analysis_model "$ANALYSIS_MODEL" \
      > "$LOG_DIR/synth_${model_large}_vs_${model_small}_pcadim${dim}_k${k}.log" 2>&1
  done
}

# Hardcoded candidates (Qwen)
run_pair "qwen3-32b" "qwen3-8b" \
  "4:5" "4:6" "4:4" "4:3" \
  "8:7" "8:8" "8:9" "8:6"

# Hardcoded candidates (GPT-OSS)
run_pair "gpt-oss-120b" "gpt-oss-20b" \
  "4:2" "4:5" "4:3" "4:6" \
  "8:2" "8:13" "8:14" "8:12" "8:3"

echo "Done. Cluster-tag JSONs under clustering_tags/<model_family>/pcadim*/"
