"""Embed advantages → dedup → PCA → KMeans; CLI entrypoint ``adv_cluster.py``."""
import os, json, argparse
import csv
from pathlib import Path
from glob import glob
import time
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import DATASET_MAP, MODEL_MAP
from embedding_utils import embed_text

_SRC_DIR = Path(__file__).resolve().parent
ADVANTAGE_ROOT = _SRC_DIR / "advantage_descriptions"

def load_advantage_description_file(datasets, model_large, model_small, advantage_extractor="gemini-3-pro") -> dict:
    data_by_ds = {}
    for ds in datasets:
        path = (
            ADVANTAGE_ROOT
            / DATASET_MAP[ds]["subject"]
            / ds
            / advantage_extractor
            / f"{model_large}_vs_{model_small}_analysis.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_by_ds[ds] = data
    return data_by_ds


def load_advantage_embedding_file(datasets, model_large, model_small) -> dict:
    data_by_ds = {}
    for ds in datasets:
        path = f"advantage_embeddings/{DATASET_MAP[ds]['subject']}/{ds}/{model_large}_vs_{model_small}_embeddings.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_by_ds[ds] = data
    return data_by_ds

def print_cluster_subject_distribution(cluster_labels, subject_refs, title="Cluster x Subject distribution (%)"):
    if len(cluster_labels) == 0 or len(subject_refs) == 0:
        return
    labels = np.asarray(cluster_labels)
    subjects = np.asarray(subject_refs)
    uniq_subjects = sorted(set(subjects.tolist()))
    subj_totals = {s: int((subjects == s).sum()) for s in uniq_subjects}
    uniq_clusters = sorted(set(labels.tolist()))

    print(f"\n{title}")
    header = "| cluster | " + " | ".join(uniq_subjects) + " |"
    sep = "| " + " | ".join(["---"] * (len(uniq_subjects) + 1)) + " |"
    print(header)
    print(sep)
    for cid in uniq_clusters:
        row = [str(cid)]
        mask_c = labels == cid
        for s in uniq_subjects:
            denom = max(1, subj_totals[s])
            pct = 100.0 * float(np.logical_and(mask_c, subjects == s).sum()) / denom
            row.append(f"{pct:.1f}%")
        print("| " + " | ".join(row) + " |")


def main():
    # pipeline is like embed -> dedup -> dimension reduction -> cluster
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="OMNI-MATH, JEEBENCH-MATH, HHMT, gpqa-physics, JEEBENCH-PHYSICS, OlympiadBench-physics, JEEBENCH-CHEMISTRY, gpqa-chemistry, CRUXEVAL-O, CRUXEVAL-I", help="ex: OMNI-MATH, JEEBENCH-MATH, HHMT, gpqa-physics, JEEBENCH-PHYSICS, OlympiadBench-physics, JEEBENCH-CHEMISTRY, gpqa-chemistry, CRUXEVAL-O, CRUXEVAL-I")
    parser.add_argument("--model_large", choices=MODEL_MAP.keys(), default="gpt-oss-120b")
    parser.add_argument("--model_small", choices=MODEL_MAP.keys(), default="gpt-oss-20b")
    parser.add_argument(
        "--advantage_extractor",
        type=str,
        default="gemini-3-pro",
        help="Subdir under src/advantage_descriptions/{subject}/{dataset}/ (default: gemini-3-pro)",
    )
    parser.add_argument("--use_cached_embeddings", action="store_true", help="use existing embedding files instead of regenerating embeddings")
    
    parser.add_argument("--dedup_threshold", type=float, default=0.95, help="cosine similarity threshold for embedding deduplication")
    
    parser.add_argument("--dr_method", type=str, default="PCA", choices=["PCA", "UMAP", "PCA+UMAP"])
    parser.add_argument("--dr_dim", type=int, default=64)

    
    parser.add_argument("--cluster_method", type=str, default="KMeans", choices=["KMeans"])
    parser.add_argument("--cluster_k", type=int, default=None, help="KMeans: single run with this k (omit if using --cluster_k_max)")
    parser.add_argument("--cluster_k_min", type=int, default=2, help="KMeans: with --cluster_k_max, sweep k from this value (inclusive)")
    parser.add_argument("--cluster_k_max", type=int, default=None, help="KMeans: if set, sweep k in [cluster_k_min, max]; one metrics CSV only; no UMAP/heatmaps/cluster JSON")
    args = parser.parse_args()
    
    # ===============================================
    # 1) validate arguments
    # ===============================================
    if args.cluster_method == "KMeans":
        if args.cluster_k_max is not None and args.cluster_k is not None:
            raise ValueError("Use either --cluster_k or --cluster_k_max, not both.")
        if args.cluster_k_max is None and args.cluster_k is None:
            raise ValueError("KMeans requires --cluster_k or --cluster_k_max.")
        if args.cluster_k_max is not None and args.cluster_k_max < args.cluster_k_min:
            raise ValueError("cluster_k_max must be >= cluster_k_min.")

    # ===============================================
    # 2) load advantage descriptions
    # ===============================================
    datasets = [ds.strip() for ds in args.datasets.replace(",", " ").split() if ds.strip()]
    advantage_description_data_by_ds = load_advantage_description_file(
        datasets, args.model_large, args.model_small, args.advantage_extractor
    )

    # ===============================================
    # 3) embed advantage descriptions
    # ===============================================
    if not args.use_cached_embeddings:
        # make sure the advantage_embeddings directory exists
        for ds in datasets:
            embedding_output_path = f'advantage_embeddings/{DATASET_MAP[ds]["subject"]}/{ds}/{args.model_large}_vs_{args.model_small}_embeddings.json'
            os.makedirs(os.path.dirname(embedding_output_path), exist_ok=True)

            data = advantage_description_data_by_ds[ds]
            for question in data.get("questions", []):
                for analysis_group in question.get("analysis", []):
                    if not analysis_group.get("analysis"):
                        continue
                    for item in analysis_group.get("analysis", []):
                        if item.get("type") == "advantage":
                            text_to_embed = item.get("advantage")
                            if not isinstance(text_to_embed, str) or not text_to_embed.strip():
                                continue
                            print(f"Embedding: {text_to_embed[:20]}...")
                            embedding_vector = embed_text(text_to_embed)
                            if embedding_vector is None:
                                continue   
                            item["advantage_embedding"] = embedding_vector
            with open(embedding_output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    advantage_embedding_data_by_ds = load_advantage_embedding_file(datasets, args.model_large, args.model_small)
    
    # ===============================================
    # 3) dedup
    # ===============================================
    # Among all embedding in one question, reduce embedddings which are too similar to each other 
    # by cosine similarity
    # apply greedy algorithm to reduce embeddings
    
    for ds in datasets:
        data = advantage_embedding_data_by_ds[ds]
        for question in data.get("questions", []):
            refs_to_compare = []
            for analysis_group in question.get("analysis", []):
                for item in analysis_group.get("analysis", []):
                    if item.get("type") == "advantage" and "advantage_embedding" in item:
                        refs_to_compare.append(item)
            
            # ===============================================
            # greedy algorithm to dedup embeddings
            # ===============================================
            kept_embeddings = []
            for item in refs_to_compare:
                curr_emb = item["advantage_embedding"]
                is_duplicate = False
                for kept_emb in kept_embeddings:
                    sim = np.dot(curr_emb, kept_emb) / (np.linalg.norm(curr_emb) * np.linalg.norm(kept_emb))
                    if sim > args.dedup_threshold:
                        is_duplicate = True
                        break
                
                item["is_redundant"] = is_duplicate # 標記
                if not is_duplicate:
                    kept_embeddings.append(curr_emb)

            # Remove redundant advantage items after marking.
            for analysis_group in question.get("analysis", []):
                filtered_items = []
                for item in analysis_group.get("analysis", []):
                    if item.get("type") == "advantage" and item.get("is_redundant", False):
                        continue
                    item.pop("is_redundant", None)
                    filtered_items.append(item)
                analysis_group["analysis"] = filtered_items
            question["analysis"] = [ag for ag in question.get("analysis", []) if ag.get("analysis")]

        dedup_output_path = f'advantage_embeddings/{DATASET_MAP[ds]["subject"]}/{ds}/{args.model_large}_vs_{args.model_small}_dedup.json'
        with open(dedup_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ===============================================
    # 4) dimension reduction 
    # ===============================================
    if args.dr_method == "PCA":
        # PCA is fitted on all deduped advantages across all datasets together.
        item_refs = []
        all_embeddings = []
        for ds in datasets:
            data = advantage_embedding_data_by_ds[ds]
            for question in data.get("questions", []):
                for analysis_group in question.get("analysis", []):
                    for item in analysis_group.get("analysis", []):
                        emb = item.get("advantage_embedding")
                        if item.get("type") != "advantage" or emb is None:
                            continue
                        emb_arr = np.asarray(emb, dtype=np.float32)
                        if emb_arr.ndim != 1 or emb_arr.size == 0:
                            continue
                        item_refs.append(item)
                        all_embeddings.append(emb_arr)

        if not all_embeddings:
            print("[WARN] No advantage embeddings available for PCA.")
            return

        # ===============================================
        # stack all embeddings together
        # ===============================================
        X = np.vstack(all_embeddings)
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
        if max_components < 1:
            print("[WARN] Invalid PCA n_components after bounds check.")
            return
        # ===============================================
        # fit PCA
        # ===============================================
        fixed_dim = min(args.dr_dim, max_components)
        pca = PCA(n_components=fixed_dim, svd_solver="full")
        X_pca = pca.fit_transform(X)
        explained_ratio = pca.explained_variance_ratio_

        # ===============================================
        # save PCA summary
        # ===============================================
        pca_summary_csv = f'advantage_embeddings/{args.model_large}_vs_{args.model_small}_pcadim{fixed_dim}_pca_summary.csv'
        with open(pca_summary_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([["requested_dr_dim","valid_dim","n_samples","n_features","max_allowed_components","cumulative_explained_variance"], [args.dr_dim, fixed_dim, n_samples, n_features, max_components, float(np.sum(explained_ratio))]])

        # ===============================================
        # save PCA output
        # ===============================================
        for item, reduced_vec in zip(item_refs, X_pca):
            item["advantage_embedding_pca"] = reduced_vec.tolist()

        for ds in datasets:
            pca_output_path = f'advantage_embeddings/{DATASET_MAP[ds]["subject"]}/{ds}/{args.model_large}_vs_{args.model_small}_dedup_pca.json'
            with open(pca_output_path, "w", encoding="utf-8") as f:
                json.dump(advantage_embedding_data_by_ds[ds], f, ensure_ascii=False, indent=2)
            
    # ===============================================
    # 5) cluster
    # ===============================================
    if args.cluster_method == "KMeans":
        all_pca_embeddings = []
        item_refs_pca = []
        subject_refs = []
        for ds in datasets:
            data = advantage_embedding_data_by_ds[ds]
            subject = DATASET_MAP[ds]["subject"]
            for question in data.get("questions", []):
                for analysis_group in question.get("analysis", []):
                    for item in analysis_group.get("analysis", []):
                        vec = item.get("advantage_embedding_pca")
                        if isinstance(vec, list) and len(vec) > 0:
                            all_pca_embeddings.append(vec)
                            item_refs_pca.append(item)
                            subject_refs.append(subject)

        # ===============================================
        # stack all PCA embeddings together
        # ===============================================
        X_cluster = np.vstack(all_pca_embeddings)
        n_samples = X_cluster.shape[0]
        os.makedirs("clustering_results", exist_ok=True)

        # ===============================================
        # sweep k
        # ===============================================
        if args.cluster_k_max is not None:
            # Run k-sweep for current dim 
            _sweep_rows = []
            for selected_k in range(int(args.cluster_k_min), int(args.cluster_k_max) + 1):
                if selected_k < 2 or selected_k > n_samples:
                    print(f"[skip] k={selected_k} out of range for n_samples={n_samples}")
                    continue
                # ===============================================
                # fit KMeans
                # ===============================================
                kmeans = KMeans(n_clusters=selected_k, random_state=0, n_init="auto")
                cluster_labels = kmeans.fit_predict(X_cluster)
                if len(set(cluster_labels)) < 2:
                    selected_ss = float("nan")
                    selected_dbi = float("nan")
                else:
                    selected_ss = float(silhouette_score(X_cluster, cluster_labels))
                    selected_dbi = float(davies_bouldin_score(X_cluster, cluster_labels))
                print(
                    f"KMeans k={selected_k}, silhouette={selected_ss:.6f}, dbi={selected_dbi:.6f}"
                )
                _sweep_rows.append([args.dr_dim, selected_k, selected_ss, selected_dbi])
            
            _csv_path = (
                f"clustering_results/{args.model_large}_vs_{args.model_small}_"
                f"pcadim{args.dr_dim}_KMeans_k_sweep_metrics.csv"
            )
            with open(_csv_path, "w", encoding="utf-8", newline="") as f:
                _w = csv.writer(f)
                _w.writerow(["pca_dim", "k", "silhouette_score", "dbi_score"])
                _w.writerows(_sweep_rows)
            print(f"Saved KMeans k-sweep metrics CSV: {_csv_path}")

            pair = f"{args.model_large}_vs_{args.model_small}"
            sweep_files = sorted(glob(f"clustering_results/{pair}_pcadim*_KMeans_k_sweep_metrics.csv"))
            all_rows = []
            for p in sweep_files:
                with open(p, "r", encoding="utf-8") as f:
                    all_rows += list(csv.DictReader(f))

            # For each dim: take DBI-top5 ks, remove SS outliers by (mean - std), compute mean DBI.
            by_dim = {}
            for r in all_rows:
                d = int(r["pca_dim"]); dbi = float(r["dbi_score"]); ss = float(r["silhouette_score"])
                if np.isfinite(dbi) and np.isfinite(ss):
                    by_dim.setdefault(d, []).append({"k": int(r["k"]), "dbi": dbi, "ss": ss})
            dim_rows = []
            for d, rows in sorted(by_dim.items()):
                top5 = sorted(rows, key=lambda x: x["dbi"])[:5]
                ss_arr = np.array([x["ss"] for x in top5], dtype=float); thr = float(ss_arr.mean() - (ss_arr.std(ddof=1) if len(ss_arr) > 1 else 0.0))
                kept = [x for x in top5 if x["ss"] >= thr] or top5
                dim_rows.append({"pca_dim": d, "mean_dbi": float(np.mean([x["dbi"] for x in kept])), "threshold_ss": thr, "k_candidates": " ".join(str(x["k"]) for x in kept)})
            dim_rows = sorted(dim_rows, key=lambda x: x["mean_dbi"])
            selected_dims = {x["pca_dim"] for x in dim_rows[:2]}

            # Persist final tables: dim selection + selected-dim full k table.
            with open(f"clustering_results/{pair}_dim_selection.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(["pca_dim", "mean_dbi", "threshold_ss", "selected", "k_candidates"]); [w.writerow([x["pca_dim"], x["mean_dbi"], x["threshold_ss"], 1 if x["pca_dim"] in selected_dims else 0, x["k_candidates"]]) for x in dim_rows]
            with open(f"clustering_results/{pair}_selected_dim_k_sweep_metrics.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(["pca_dim", "k", "silhouette_score", "dbi_score"]); [w.writerow([r["pca_dim"], r["k"], r["silhouette_score"], r["dbi_score"]]) for r in all_rows if int(r["pca_dim"]) in selected_dims]

        
        # ===============================================
        # single run with a fixed k as tool
        # ===============================================
        else:
            # ===============================================
            # validate k
            # ===============================================
            selected_k = int(args.cluster_k)
            if selected_k < 2 or selected_k > n_samples:
                raise ValueError(f"cluster_k must satisfy 2 <= k <= n_samples ({n_samples}), got {selected_k}")

            # ===============================================
            # fit KMeans
            # ===============================================
            kmeans = KMeans(n_clusters=selected_k, random_state=0, n_init="auto")
            cluster_labels = kmeans.fit_predict(X_cluster)
            if len(set(cluster_labels)) < 2:
                selected_ss = float("nan")
                selected_dbi = float("nan")
            else:
                selected_ss = float(silhouette_score(X_cluster, cluster_labels))
                selected_dbi = float(davies_bouldin_score(X_cluster, cluster_labels))
            print(
                f"KMeans k={selected_k}, silhouette={selected_ss:.6f}, dbi={selected_dbi:.6f}"
            )

            kmeans_out_stem = (
                f"{args.model_large}_vs_{args.model_small}_pcadim{args.dr_dim}_k{selected_k}_KMeans"
            )

            for item, cid in zip(item_refs_pca, cluster_labels):
                item["cluster_id_KMeans"] = int(cid)
            print_cluster_subject_distribution(cluster_labels, subject_refs, "KMeans cluster x subject distribution (%)")
            for ds in datasets:
                output_path = (
                    f'clustering_results/{DATASET_MAP[ds]["subject"]}/{ds}/{kmeans_out_stem}.json'
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(advantage_embedding_data_by_ds[ds], f, ensure_ascii=False, indent=2)
                print(f"Saved KMeans clustering output: {output_path}")
if __name__ == "__main__":
    main()