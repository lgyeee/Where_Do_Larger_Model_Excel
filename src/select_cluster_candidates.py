"""
Per-(pca_dim, K) taxonomy review via OpenRouter: one evaluation call per synthesis candidate.

Reads synthesis JSONs (cluster id, tag, definition), parses model JSON into the fixed schema.
"""
import os
import json
import argparse
import re

from dotenv import load_dotenv
from openai import OpenAI


# ===============================================
# Fixed prompt (taxonomy evaluation). Use .format(k=..., clusters_section=...).
# Literal braces in the JSON example must be doubled {{ }} for str.format.
# ===============================================
TAXONOMY_REVIEW_PROMPT = """You are evaluating a taxonomy of reasoning advantages induced from clustering.

Do not rewrite the taxonomy. Do not assume that fewer clusters are always better. Focus only on taxonomy quality.

You will be given K and, for each cluster, an id, a tag, and a short definition.

Evaluate the taxonomy using these criteria:
### 1. Mutual Exclusivity (Distinctness)
- [1-2] Redundant: Two or more clusters describe the same concept.
- [3-4] Overlap: Significant overlap (>40% chance of double-fitting).
- [5-6] Moderate: Different concepts, but boundaries are fuzzy/lack exclusion.
- [7-8] Sharp: Distinct features per cluster; very low ambiguity.
- [9-10] Exclusive: Logically impossible for an instance to belong to two clusters.

### 2. Conceptual Precision & Depth (Granularity)
- [1-2] Vague/Circular: Definitions offer no real insight or repeat the tag.
- [3-4] Surface-level: Describes ONLY the outcome (e.g., "the answer is wrong").
- [5-6] Functional: Describes the reasoning process/behavior in plain language.
- [7-8] Academic: Explains the underlying mechanism using formal logic/CS terminology.
- [9-10] Research-Ready: High-level synthesis of unique reasoning failures/patterns; publication quality.

## 3. Interpretability (Narrative Value)
- [1-2] Obscure: Hard to explain what kind of cases fit here.
- [3-4] Complex: Requires heavy mental effort to map to actual model behavior.
- [5-6] Clear: Understandable, but lacks a strong, cohesive "story."
- [7-8] Intuitive: A researcher can immediately picture the failure mode.
- [9-10] Insightful: Provides an "Aha!" moment; memorable and easy to communicate.

### 4. Cluster Balance & Utility (Granularity)
- [1-2] Failed: One dominant cluster (>80%) or useless fragmented noise.
- [3-4] Poor: Highly skewed; major categories have <1% samples.
- [5-6] Passable: Distribution allows for basic statistical observation.
- [7-8] Healthy: Balanced enough to reveal clear performance trends across models.
- [9-10] Optimal: Ideal for high-precision comparative analysis and scaling law research.

5. Taxonomy Resolution (Granularity)
- [1-3] Coarse: Too broad; merges distinct behaviors, losing diagnostic power.
- [4-6] Fragmented: Too noisy; over-splits clusters based on surface-level wording.
- [7-10] Functional: Just the right number of clusters to cover the main reasoning stages.
Input:
K = {k}
CLUSTERS =
{clusters_section}

Output one JSON object only.

{{
  "Distinctness_score": [1-10],
  "Granularity_score": [1-10],
  "Interpretability_score": [1-10],
  "Balance_score": [1-10],
  "Taxonomy_resolution_score": [1-10],
}}
"""

CLUSTER_BLOCK_TEMPLATE = """Cluster ID: {cluster_id}
Tag: {tag}
Definition: {definition}
Count: {count}
"""


def parse_dim_k_pairs(raw: str):
    pairs = []
    seen = set()
    for tok in raw.replace(",", " ").split():
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        a, b = tok.split(":", 1)
        key = (int(a.strip()), int(b.strip()))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _extract_model_family(model_name: str) -> str:
    return model_name.split("-")[0].split("_")[0]


def synthesis_path(model_large: str, model_small: str, cluster_method: str, pca_dim: int, k: int) -> str:
    if cluster_method != "KMeans":
        raise ValueError(f"Unsupported cluster_method for summarizing.py output: {cluster_method}")
    model_family = _extract_model_family(model_large)
    return os.path.join(
        "clustering_tags",
        model_family,
        f"pcadim{pca_dim}",
        f"{model_large}_vs_{model_small}_k{k}.json",
    )


def parse_llm_json(content: str):
    """
    Parse model output into dict. Tries raw JSON, then fenced ```json blocks, then outermost braces.
    Returns (parsed_or_none, error_message_or_none).
    """
    if not content or not content.strip():
        return None, "empty content"
    text = content.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip()), None
        except json.JSONDecodeError as e:
            return None, f"fenced JSONDecodeError: {e}"
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1]), None
        except json.JSONDecodeError as e:
            return None, f"brace-slice JSONDecodeError: {e}"

    return None, "could not parse JSON from model output"


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    parser = argparse.ArgumentParser(
        description="OpenRouter taxonomy review per (pca_dim, K) synthesis candidate."
    )
    parser.add_argument("--model_large", type=str, default="qwen3-32b")
    parser.add_argument("--model_small", type=str, default="qwen3-8b")
    parser.add_argument("--cluster_method", type=str, default="KMeans", choices=["KMeans", "HDBSCAN", "GMM"])
    parser.add_argument(
        "--candidates",
        type=str,
        default="4:3 4:4 4:5 4:6 8:6 8:7 8:8 8:9",
        help='Space or comma separated dim:k (dim only used to load files), e.g. "4:3,8:7"',
    )
    parser.add_argument("--review_model", type=str, default="openai/gpt-5.2")
    args = parser.parse_args()

    dim_k_list = parse_dim_k_pairs(args.candidates)
    if not dim_k_list:
        raise SystemExit("No valid dim:k in --candidates")

    # ===============================================
    # 1) Per candidate: build prompt, call API, parse JSON
    # ===============================================
    evaluations = []

    for dim, k in dim_k_list:
        spath = synthesis_path(
            args.model_large, args.model_small, args.cluster_method, dim, k
        )
        if not os.path.isfile(spath):
            raise FileNotFoundError(f"Missing cluster-tag JSON (run summarizing.py first): {spath}")
        with open(spath, "r", encoding="utf-8") as f:
            synth = json.load(f)
        clusters = synth.get("clusters") or []

        cluster_blocks = []
        per_cluster_meta = []
        for c in sorted(clusters, key=lambda x: int(x.get("cluster_id", 0))):
            cid = int(c.get("cluster_id", 0))
            tag = c.get("tag", "")
            definition = c.get("definition", "")
            count = int(c.get("count", 0))
            cluster_blocks.append(
                CLUSTER_BLOCK_TEMPLATE.format(
                    cluster_id=cid,
                    tag=tag,
                    definition=definition,
                    count=count,
                )
            )
            per_cluster_meta.append(
                {
                    "cluster_id": cid,
                    "tag": tag,
                    "definition": definition,
                    "count": count,
                }
            )

        clusters_section = "".join(cluster_blocks)
        prompt = TAXONOMY_REVIEW_PROMPT.format(k=k, clusters_section=clusters_section)

        completion = client.chat.completions.create(
            model=args.review_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw_response = completion.choices[0].message.content
        parsed, parse_error = parse_llm_json(raw_response)

        evaluations.append(
            {
                "pca_dim": dim,
                "k": k,
                "synthesis_path": spath,
                "prompt_sent": prompt,
                "raw_response": raw_response,
                "parsed": parsed,
                "parse_error": parse_error,
                "per_cluster_prompt_meta": per_cluster_meta,
            }
        )
        status = "ok" if parsed is not None else f"parse_fail: {parse_error}"
        print(f"[pca_dim={dim} k={k}] {status}")

    # ===============================================
    # 2) save file (same location pattern as before)
    # ===============================================
    os.makedirs("select_results/", exist_ok=True)
    out_path = os.path.join(
        "select_results",
        f"{args.model_large}_vs_{args.model_small}_select_k_candidate_review.json",
    )
    output = {
        "model_large": args.model_large,
        "model_small": args.model_small,
        "cluster_method": args.cluster_method,
        "candidates_dim_k": [{"pca_dim": d, "k": kk} for d, kk in dim_k_list],
        "review_model": args.review_model,
        "evaluations": evaluations,
        "summary": {
            "n_candidates": len(evaluations),
            "n_parse_ok": sum(1 for e in evaluations if e["parsed"] is not None),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved select-K results: {out_path}")