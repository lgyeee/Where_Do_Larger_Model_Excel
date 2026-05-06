#!/usr/bin/env bash
# sweep_dim_k.sh — Sweep multiple pca_dim × k combos and summarize to clustering_results/*_k_sweep_metrics.csv + stdout.
# First pca_dim per model pair: no --use_cached_embeddings; later dims use cache. Re-run all cached: USE_CACHED_EMBEDDINGS=1.
# Env overrides: MODEL_PAIRS_OVERRIDE, MODEL_LARGE_OVERRIDE+MODEL_SMALL_OVERRIDE, PRELIM_DIMS_OVERRIDE, K_MIN/MAX_OVERRIDE, DATASET_OVERRIDE.
set -euo pipefail

trap 'echo; echo "⚡️ Interrupted—killing sweep..."; jobs -p | xargs -r kill; exit 1' SIGINT SIGTERM

# Model pairs config (override via env)
if [[ -n "${MODEL_PAIRS_OVERRIDE:-}" ]]; then
  read -ra MODEL_PAIRS <<< "${MODEL_PAIRS_OVERRIDE}"
elif [[ -n "${MODEL_LARGE_OVERRIDE:-}" && -n "${MODEL_SMALL_OVERRIDE:-}" ]]; then
  MODEL_PAIRS=( "${MODEL_LARGE_OVERRIDE}:${MODEL_SMALL_OVERRIDE}" )
else
  MODEL_PAIRS=( "qwen3-32b:qwen3-8b" "gpt-oss-120b:gpt-oss-20b" )
fi

read -ra DATASETS <<< "${DATASET_OVERRIDE:-}"

read -ra PRELIM_DIMS <<< "${PRELIM_DIMS_OVERRIDE:-256 128 64 16 8 4}"
K_MIN="${K_MIN_OVERRIDE:-2}"
K_MAX="${K_MAX_OVERRIDE:-15}"
USE_CACHED_EMBEDDINGS="${USE_CACHED_EMBEDDINGS:-0}"

LOG_DIR="clustering_logs"
RESULTS_DIR="clustering_results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Default datasets if not overridden
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(
    "OMNI-MATH" "JEEBENCH-MATH" "HHMT"
    "gpqa-physics" "JEEBENCH-PHYSICS" "OlympiadBench-physics"
    "JEEBENCH-CHEMISTRY" "gpqa-chemistry"
    "CRUXEVAL-O" "CRUXEVAL-I"
  )
fi
BASE_DATASETS_ARG="${DATASETS[*]}"

for PAIR in "${MODEL_PAIRS[@]}"; do
  IFS=':' read -r MODEL_L MODEL_S <<< "$PAIR"
  if [[ -z "$MODEL_L" || -z "$MODEL_S" ]]; then
    echo "Invalid MODEL_PAIRS entry: '$PAIR' (expected large:small)"
    exit 1
  fi

  # Remove HHMT for Qwen3 pair (avoid empty embedding / PCA error)
  datasets_arg="$BASE_DATASETS_ARG"
  if [[ "$MODEL_L" == "qwen3-32b" && "$MODEL_S" == "qwen3-8b" ]]; then
    datasets_arg="${datasets_arg//HHMT/}"
    datasets_arg="$(echo "$datasets_arg" | xargs)"
  fi

  echo -e "\n── Model pair | LLM=$MODEL_L | SLM=$MODEL_S |"

  first_dim=1
  for D in "${PRELIM_DIMS[@]}"; do
    LOG_TAG="${MODEL_L}_vs_${MODEL_S}_pcadim${D}_ksweep_${K_MIN}-${K_MAX}"
    log_path="$LOG_DIR/${LOG_TAG}.log"
    cache_opt=()

    if [[ "$USE_CACHED_EMBEDDINGS" == "1" || "$first_dim" -eq 0 ]]; then
      cache_opt=(--use_cached_embeddings)
      echo "   ▶ pca_dim=$D  k=${K_MIN}..${K_MAX}  [use_cached_embeddings]  (log: $LOG_TAG.log)"
    else
      echo "   ▶ pca_dim=$D  k=${K_MIN}..${K_MAX}  [embed + sweep; no cache]  (log: $LOG_TAG.log)"
    fi
    first_dim=0

    python adv_cluster.py \
      --datasets "$datasets_arg" \
      --model_large "$MODEL_L" \
      --model_small "$MODEL_S" \
      --dr_method PCA \
      "${cache_opt[@]}" \
      --dr_dim "$D" \
      --cluster_method KMeans \
      --cluster_k_min "$K_MIN" \
      --cluster_k_max "$K_MAX" \
      > "$log_path" 2>&1
  done

  # Output metrics summary for each pca_dim
  echo "   ✓ dim mean metrics (stdout)"
  printf "%-8s | %-12s | %-17s | %-4s\n" "pca_dim" "mean_dbi" "mean_silhouette" "n_k"
  printf -- "---------------------------------------------------------\n"
  for D in "${PRELIM_DIMS[@]}"; do
    METRIC_CSV="$RESULTS_DIR/${MODEL_L}_vs_${MODEL_S}_pcadim${D}_KMeans_k_sweep_metrics.csv"
    if [[ -f "$METRIC_CSV" ]]; then
      awk -F, -v d="$D" '
        NR>1 {ss+=$3; dbi+=$4; n+=1}
        END {if (n>0) printf "%-8s | %-12.6f | %-17.6f | %-4d\n", d, dbi/n, ss/n, n}
      ' "$METRIC_CSV"
    fi
  done
  echo
done
