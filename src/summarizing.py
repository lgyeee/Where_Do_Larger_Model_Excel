"""LLM tag + definition per cluster from clustered advantages → clustering_tags/<model_family>/pcadim*/..._k*.json."""
import os
import json
import argparse
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI
from utils import DATASET_MAP


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    parser = argparse.ArgumentParser()
    # Keep in sync with run_clustering.sh default DATASETS so output paths find clustered JSONs.
    parser.add_argument(
        "--datasets",
        type=str,
        default="OMNI-MATH,JEEBENCH-MATH,HHMT,gpqa-physics,JEEBENCH-PHYSICS,OlympiadBench-physics,JEEBENCH-CHEMISTRY,gpqa-chemistry,CRUXEVAL-O,CRUXEVAL-I",
    )
    parser.add_argument("--model_large", type=str, default="gpt-oss-120b")
    parser.add_argument("--model_small", type=str, default="gpt-oss-20b")
    parser.add_argument("--cluster_method", type=str, default="KMeans", choices=["KMeans"])
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="KMeans: if set, load ..._pcadim{pca_dim}_k{cluster_k}_KMeans.json; else legacy dedup_pca_KMeans_k{k}.json",
    )
    parser.add_argument("--cluster_k", type=int, default=12)
    parser.add_argument("--analysis_model", type=str, default="openai/gpt-5.2")
    parser.add_argument("--max_examples_per_cluster", type=int, default=300)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.replace(",", " ").split() if d.strip()]
    cluster_key = "cluster_id_KMeans"

    # ===============================================
    # 1) load file
    # ===============================================
    cluster_data_by_ds = {}
    for ds in datasets:
        subject = DATASET_MAP[ds]["subject"]
        if args.pca_dim is not None:
            fname = (
                f"{args.model_large}_vs_{args.model_small}_pcadim{args.pca_dim}_k{args.cluster_k}_KMeans.json"
            )
        else:
            # Backward-compatible fallback for older KMeans output naming.
            fname = (
                f"{args.model_large}_vs_{args.model_small}_dedup_pca_KMeans_k{args.cluster_k}.json"
            )
        path = f"clustering_results/{subject}/{ds}/{fname}"
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing clustering output (run adv_cluster.py first): {path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            cluster_data_by_ds[ds] = json.load(f)

    # ===============================================
    # 2) group by cluster
    # ===============================================
    cluster_data_by_cluster_id = defaultdict(list)
    for ds in datasets:
        data = cluster_data_by_ds[ds]
        for q_idx, q in enumerate(data.get("questions", [])):
            for g in q.get("analysis", []):
                for item in g.get("analysis", []):
                    if item.get("type") != "advantage":
                        continue
                    if cluster_key not in item:
                        continue
                    cid = int(item[cluster_key])
                    cluster_data_by_cluster_id[cid].append(
                        {
                            "dataset": ds,
                            "question_idx": q_idx,
                            "llm_run_id": g.get("llm_run_id"),
                            "slm_run_id": g.get("slm_run_id"),
                            "advantage": item.get("advantage", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )

    # ===============================================
    # 3) OPENROUTER API (parse response)
    # ===============================================
    cluster_summaries = []
    for cid in sorted(cluster_data_by_cluster_id.keys()):
        records = cluster_data_by_cluster_id[cid]
        examples = [r["advantage"] for r in records if r["advantage"]][: args.max_examples_per_cluster]
        text_block = "\n".join([f"- {x}" for x in examples])
        prompt = (
            f"You are an expert in LLM Reasoning Analysis.\n\n"
            f"I will provide a cluster of similar reasoning advantages.\n\n"
            f"Please summarize the advantages and provide a tag and definition for the cluster.\n\n"
            f"INSTRUCTIONS:\n"
            f"1. DE-DOMAIN: Strip all subject-specific context (e.g., replace 'chemical valency' with 'structural constraints').\n"
f"2. ACTION-VERB TAGS: The tag MUST start with a strong action verb (e.g., Mapping, Verifying, Reducing, Deriving, Resolving). Avoid generic tags like 'Reasoning' or 'Logic'.\n"
f"3. DEFINITION: Describe *how* the model manipulates information to reach a conclusion. Use 2-4 clear sentences.\n"
f"4. LENGTH: Tag (2–4 words).\n\n"
            
            f"Advantages (bullet list):\n"
            f"{text_block}\n\n"
            f"RETURN JSON ONLY in this format:\n"
            "{\n"
            '  "tag": "...",\n'
            '  "definition": "..."\n'
            "}\n\n"
        )
        completion = client.chat.completions.create(model=args.analysis_model, messages=[{"role": "user", "content": prompt}], temperature=0)
        content = completion.choices[0].message.content
        try:
            parsed = json.loads(content)
            tag = parsed.get("tag", f"cluster_{cid}")
            definition = parsed.get("definition", "")
        except Exception:
            tag = f"cluster_{cid}"
            definition = content

        cluster_summaries.append(
            {
                "cluster_id": cid,
                "count": len(records),
                "tag": tag,
                "definition": definition,
            }
        )
        print(f"[cluster {cid}] {tag}")

    # ===============================================
    # 4) save file
    # ===============================================
    # Directory structure: clustering_tags/<model_family>/<pcadim>/<model_large>_vs_<model_small>_k<cluster_k>.json
    # model_family: extract it from model_large (before the first hyphen or underscore)
    def extract_model_family(model_name):
        return model_name.split("-")[0].split("_")[0]
    model_family = extract_model_family(args.model_large)
    base_dir = f"clustering_tags/{model_family}"
    if args.pca_dim is not None:
        dim_dir = f"{base_dir}/pcadim{args.pca_dim}"
    else:
        dim_dir = f"{base_dir}/no_pca"
    os.makedirs(dim_dir, exist_ok=True)

    out_path = (
        f"{dim_dir}/"
        f"{args.model_large}_vs_{args.model_small}_k{args.cluster_k}.json"
    )
    output = {
        "datasets": datasets,
        "model_large": args.model_large,
        "model_small": args.model_small,
        "cluster_method": args.cluster_method,
        "cluster_k": args.cluster_k,
        "pca_dim": args.pca_dim,
        "analysis_model": args.analysis_model,
        "clusters": cluster_summaries,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved synthesis: {out_path}")
