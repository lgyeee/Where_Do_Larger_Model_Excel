#!/usr/bin/env bash
# Run select_cluster_candidates.py for both model pairs used in dim_selection.csv
# (selected grids: pca_dim 4 & 8 with their k_candidates).
#
# Requires: OPENROUTER_API_KEY in .env; cluster-tag JSONs under clustering_tags/{model_family}/pcadim*/
#   (from summarizing.py / run_synthesize_dim_k.sh for each dim:k).
#
# Optional: REVIEW_MODEL=openai/gpt-5.2 (default)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REVIEW_MODEL="${REVIEW_MODEL:-openai/gpt-5.2}"

# qwen3-32b_vs_qwen3-8b_dim_selection.csv — selected=1 rows: dim 4 → k 5 6 4 3; dim 8 → k 7 8 9 6
QWEN_CANDIDATES="4:3 4:4 4:5 4:6 8:6 8:7 8:8 8:9"

# gpt-oss-120b_vs_gpt-oss-20b_dim_selection.csv — selected=1 rows: dim 4 → k 2 5 3 6; dim 8 → k 2 13 14 12 3
GPT_OSS_CANDIDATES="4:2 4:3 4:5 4:6 8:2 8:3 8:12 8:13 8:14"

echo "=== Qwen3 (32b vs 8b) ==="
python3 select_cluster_candidates.py \
  --model_large qwen3-32b \
  --model_small qwen3-8b \
  --cluster_method KMeans \
  --candidates "$QWEN_CANDIDATES" \
  --review_model "$REVIEW_MODEL"

echo "=== GPT-OSS (120b vs 20b) ==="
python3 select_cluster_candidates.py \
  --model_large gpt-oss-120b \
  --model_small gpt-oss-20b \
  --cluster_method KMeans \
  --candidates "$GPT_OSS_CANDIDATES" \
  --review_model "$REVIEW_MODEL"

echo "Done. Outputs: select_results/*_select_k_candidate_review.json"
