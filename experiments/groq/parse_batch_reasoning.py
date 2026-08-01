import os
import json
import hashlib
import argparse
import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset

base = Path(__file__).resolve()
for p in (base.parents[2] / "src", base.parents[3]):
    if p.exists():
        sys.path.insert(0, str(p))
        break

from utils import (
    DATASET_MAP, extract_answer, verify_answer,
    _extract_boxed, _literal_eval_safe, _eq_cruxeval, _normalize_gold, 
)

def build_question_map(dataset: str) -> dict:
    """Load dataset with same filters as gen_batch_requests.py,
    return {q_hash: {question, gold}}."""
    cfg = DATASET_MAP[dataset]
    ds_name, split = cfg["args"]
    config_name = cfg.get("config")
    ds = load_dataset(ds_name, config_name, split=split) if config_name else load_dataset(ds_name, split=split)

    question_key = cfg["question_key"]
    answer_key = cfg["answer_key"]

    if dataset == "OMNI-MATH":
        dk = cfg["difficulty_key"]
        ds = ds.filter(lambda ex: ex[dk] >= 7 and len(str(ex[answer_key])) <= 20)

    if dataset in ("gpqa-physics", "gpqa-chemistry"):
        fk, fv = cfg["filter_key"], cfg["filter_value"]
        ds = ds.filter(lambda ex: str(ex.get(fk, "")).lower() == str(fv).lower())

    if dataset in ("JEEBENCH-PHYSICS", "JEEBENCH-CHEMISTRY", "JEEBENCH-MATH"):
        ds = ds.filter(lambda ex: ex[cfg["filter_key"]] == cfg["filter_value"])

    if dataset in ("MMLU-Pro-math", "MMLU-Pro-physics", "MMLU-Pro-chemistry"):
        ds = ds.filter(lambda ex: ex[cfg["filter_key"]] == cfg["filter_value"])
        ds = ds.select(range(20))

    if dataset == "OlympiadBench-physics":
        ds = ds.filter(lambda ex: ex["answer_type"] == "Numerical")

    mapping = {}
    for ex in ds:
        q = ex[question_key]
        q_hash = hashlib.sha256(q.encode()).hexdigest()[:8]
        mapping[q_hash] = {"question": q, "gold": ex[answer_key]}
    return mapping


def read_batch_results(model: str, dataset: str, n_sample: int) -> list[dict]:
    subject = DATASET_MAP[dataset]["subject"]
    path = f"batch_results/{subject}/{dataset}/{model}/{n_sample}_runs.jsonl"
    if not os.path.exists(path):
        print(f"[SKIP] batch result not found yet: {path}")
        return []
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_sample", type=int, required=True)
    args = parser.parse_args()

    # 1) Build q_hash → {question, gold} mapping from dataset
    q_map = build_question_map(args.dataset)

    # 2) Read batch results
    data = read_batch_results(args.model, args.dataset, args.n_sample)
    if not data:
        return

    # 3) Parse each result, match by q_hash
    runs = {rid: {} for rid in range(args.n_sample)}

    for obj in data:
        custom_id = obj.get("custom_id", "")

        try:
            ds_and_hash, rid_str = custom_id.rsplit("-run-", 1)
            _, q_hash = ds_and_hash.rsplit("-", 1)
            rid = int(rid_str)
        except ValueError:
            print(f"[WARN] Cannot parse custom_id: {custom_id}")
            continue

        if q_hash not in q_map:
            print(f"[WARN] q_hash {q_hash} not found in dataset")
            continue

        resp = obj.get("response", {})
        body = resp.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            continue

        reasoning = choices[0].get("message", {}).get("reasoning", "")
        content = choices[0].get("message", {}).get("content", "")
        reasoning_length = body.get("usage", {}).get("completion_tokens", 0)
        full_response = f"<think>\n{reasoning}\n</think>\n{content}"

        gold = q_map[q_hash]["gold"]
        correct = False

        if args.dataset == "OlympiadBench-physics":
            gold = _normalize_gold(gold)

        if args.dataset in ("CRUXEVAL-O", "CRUXEVAL-I"):
            pred = _extract_boxed(full_response)
            gold_obj = _literal_eval_safe(gold)
            pred_obj = _literal_eval_safe(pred)
            try:
                correct = _eq_cruxeval(pred_obj, gold_obj)
            except Exception:
                pass
            pred_display = repr(pred_obj)
            gold_display = repr(gold_obj)
        else:
            pred = extract_answer(full_response)
            try:
                correct = verify_answer(gold, pred)
            except Exception:
                pass
            pred_display = pred
            gold_display = gold

        runs[rid][q_hash] = {
            "question": q_map[q_hash]["question"],
            "full_response": full_response,
            "reasoning_length": reasoning_length,
            "prediction": pred_display,
            "gold": gold_display,
            "correct": correct,
        }

    # 4) Per-question pass rate across n_sample
    all_q_hashes = set()
    for rid in runs:
        all_q_hashes.update(runs[rid].keys())

    for qh in all_q_hashes:
        pass_count = sum(
            1 for rid in runs if qh in runs[rid] and runs[rid][qh].get("correct")
        )
        pass_rate = pass_count / args.n_sample * 100.0
        for rid in runs:
            if qh in runs[rid]:
                runs[rid][qh]["hit_rate"] = pass_rate
                runs[rid][qh]["hit_count"] = pass_count

    # 5) Convert to list format and compute per-run stats
    sorted_hashes = sorted(all_q_hashes)
    summary = {"runs": [], "aggregate": {}}
    accs = []
    lengths = []

    for rid in range(args.n_sample):
        records = [runs[rid][qh] for qh in sorted_hashes if qh in runs[rid]]
        if not records:
            continue
        acc = sum(1 for r in records if r["correct"]) / len(records) * 100
        avg_len = sum(r["reasoning_length"] for r in records) / len(records)
        accs.append(acc)
        lengths.append(avg_len)
        summary["runs"].append({
            "run_id": rid,
            "accuracy": acc,
            "avg_length": avg_len,
            "records": records,
        })

    hit_rates = [
        sum(1 for rid in runs if qh in runs[rid] and runs[rid][qh].get("correct"))
        / args.n_sample * 100.0
        for qh in sorted_hashes
    ]

    summary["aggregate"]["mean_accuracy"] = float(np.mean(accs)) if accs else 0.0
    summary["aggregate"]["std_accuracy"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    summary["aggregate"]["mean_length"] = float(np.mean(lengths)) if lengths else 0.0
    summary["aggregate"]["std_length"] = float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0
    summary["aggregate"]["per_question_hit_rate_mean"] = float(np.mean(hit_rates)) if hit_rates else 0.0
    summary["aggregate"]["per_question_hit_rate_std"] = float(np.std(hit_rates, ddof=1)) if len(hit_rates) > 1 else 0.0

    # 6) Save
    subject = DATASET_MAP[args.dataset]["subject"]
    os.makedirs(f"eval_outputs/{subject}/{args.dataset}/{args.model}", exist_ok=True)
    output_path = f"eval_outputs/{subject}/{args.dataset}/{args.model}/{args.n_sample}_runs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Per-run accuracies: {accs}")
    print(f"Mean ± std accuracy: {summary['aggregate']['mean_accuracy']:.2f}% ± {summary['aggregate']['std_accuracy']:.2f}%")
    print(f"Mean ± std length: {summary['aggregate']['mean_length']:.1f} ± {summary['aggregate']['std_length']:.1f} tokens")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
