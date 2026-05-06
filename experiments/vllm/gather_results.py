# gather_results.py
#!/usr/bin/env python
#
# After running all shards, this script merges them in order,
# recomputes per-run accuracy/length, and writes an aggregated JSON.

import json
import glob
import numpy as np
import argparse
import sys
from pathlib import Path

base = Path(__file__).resolve()
src_dir = base.parents[2] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils import DATASET_MAP

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True)
    p.add_argument("--model",      required=True)
    p.add_argument("--n_sample",   type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    args = p.parse_args()

    base = str(src_dir / "eval_outputs" / DATASET_MAP[args.dataset]["subject"] / args.dataset / args.model)
    merged = {rid: [] for rid in range(args.n_sample)}

    # Load each shard in ascending order
    for sid in range(args.num_shards):
        path = f"{base}/{args.n_sample}_runs_shard{sid}.json"
        with open(path) as f:
            shard = json.load(f)
        for run in shard["runs"]:
            rid = run["run_id"]
            merged[rid].extend(run["records"])

    # Recompute per-run stats
    runs, accs, lens = [], [], []
    for rid, recs in merged.items():
        total = len(recs)
        correct = sum(1 for r in recs if r.get("correct"))
        acc = correct / total * 100.0
        avg_len = sum(r["reasoning_length"] for r in recs) / total
        runs.append({
            "run_id":     rid,
            "accuracy":   acc,
            "avg_length": avg_len,
            "records":    recs
        })
        accs.append(acc)
        lens.append(avg_len)

    # Per-question hit stats
    question_hits = {}
    for recs in merged.values():
        for r in recs:
            q = r.get("question")
            if q is None:
                continue
            if bool(r.get("correct")):
                question_hits[q] = question_hits.get(q, 0) + 1
            else:
                question_hits.setdefault(q, 0)

    # Per-question hit-rate mean/std: (#correct per question)/n_sample * 100
    hit_rates = [
        hits / args.n_sample * 100.0 for hits in question_hits.values()
    ] if question_hits else []

    # Annotate each record with its question's hit_count/hit_rate
    for run in runs:
        for rec in run["records"]:
            q = rec.get("question")
            if q is None:
                continue
            hits = question_hits.get(q, 0)
            rec["hit_count"] = hits
            rec["hit_rate"] = hits / args.n_sample * 100.0

    aggregate = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs, ddof=1)) if len(accs)>1 else 0.0,
        "mean_length":   float(np.mean(lens)),
        "std_length":    float(np.std(lens, ddof=1)) if len(lens)>1 else 0.0,
        "per_question_hit_rate_mean": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "per_question_hit_rate_std":  float(np.std(hit_rates, ddof=1)) if len(hit_rates)>1 else 0.0,
    }

    out = {"runs": runs, "aggregate": aggregate}
    out_path = f"{base}/{args.n_sample}_runs.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
    print("Wrote aggregated summary to", out_path)

if __name__ == "__main__":
    main()
