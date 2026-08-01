#!/usr/bin/env bash
# Run select_cluster_candidates.py for model pairs used in dim_selection.csv
# (selected grids: pca_dim 4 & 8 with their k_candidates).
#
# Requires: OPENROUTER_API_KEY in .env; cluster-tag JSONs under clustering_tags/{model_family}/pcadim*/
#   (from summarizing.py / create_cluster_tag.sh for each dim:k).
#
# Optional: REVIEW_MODEL=openai/gpt-5.2 (default)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REVIEW_MODEL="${REVIEW_MODEL:-openai/gpt-5.2}"

# From each pair's *_dim_selection.csv (selected=1 rows)
QWEN_CANDIDATES="4:3 4:4 4:5 4:6 8:6 8:7 8:8 8:9"
GPT_OSS_CANDIDATES="4:2 4:3 4:5 4:6 8:2 8:3 8:12 8:13 8:14"
GEMMA4_CANDIDATES="4:5 4:14 4:13 4:7 4:12 8:9 8:11 8:10 8:13"

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

echo "=== Gemma4 (12b vs e4b) ==="
python3 select_cluster_candidates.py \
  --model_large gemma4-12b \
  --model_small gemma4-e4b \
  --cluster_method KMeans \
  --candidates "$GEMMA4_CANDIDATES" \
  --review_model "$REVIEW_MODEL"

echo "Done. Outputs: select_results/*_select_k_candidate_review.json"
