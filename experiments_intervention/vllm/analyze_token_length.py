#!/usr/bin/env python3
"""Summarize prompt_tokens + reasoning_length from vllm {mode}_results JSON files."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

INTERVENTION_DIR = Path(__file__).resolve().parents[1]

MODES = ["slm", "slm-normal", "slm-guided", "llm-guided"]
DEFAULT_MODEL = "qwen3-8b"
EXTRACTION_MODELS = {
    ("qwen3-8b", "slm-guided"): "qwen3-8b",
    ("qwen3-8b", "llm-guided"): "qwen3-32b",
    ("gpt-oss-20b", "slm-guided"): "gpt-oss-20b",
    ("gpt-oss-20b", "llm-guided"): "gpt-oss-120b",
}


def parse_result_path(path: Path) -> dict:
    rel = path.relative_to(INTERVENTION_DIR)
    parts = rel.parts
    mode = parts[0].removesuffix("_results")
    subject, dataset, model = parts[1], parts[2], parts[3]
    extraction_model = parts[4] if len(parts) > 5 else None
    return {
        "mode": mode,
        "subject": subject,
        "dataset": dataset,
        "model": model,
        "extraction_model": extraction_model,
        "path": str(path),
    }


def load_token_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = parse_result_path(path)
    rows = []
    for run in data.get("runs", []):
        run_id = run.get("run_id")
        for rec in run.get("records", []):
            prompt_tokens = rec.get("prompt_tokens") or 0
            reasoning_length = rec.get("reasoning_length") or 0
            rows.append({
                **meta,
                "run_id": run_id,
                "question": rec.get("question"),
                "prompt_tokens": prompt_tokens,
                "reasoning_length": reasoning_length,
                "total_tokens": prompt_tokens + reasoning_length,
            })
    return rows


def find_result_files() -> list[Path]:
    files = []
    for d in sorted(INTERVENTION_DIR.glob("*_results")):
        if d.is_dir():
            files.extend(sorted(d.rglob("*_runs.json")))
    return files


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def rows_for_mode(per_question: list[dict], mode: str, model: str) -> list[dict]:
    if mode in ("slm-guided", "llm-guided"):
        extractor = EXTRACTION_MODELS.get((model, mode), "")
        return [
            q for q in per_question
            if q["mode"] == mode and q["model"] == model and q["extraction_model"] == extractor
        ]
    return [
        q for q in per_question
        if q["mode"] == mode and q["model"] == model and not q["extraction_model"]
    ]


def print_master_table(per_question: list[dict], model: str) -> None:
    print(f"==== tokens (all questions pooled equally; model={model}) ====")
    print(f"{'mode':<14} {'n_q':>5} {'prompt':>8} {'reason':>8} {'tokens':>10}")
    for mode in MODES:
        qs = rows_for_mode(per_question, mode, model)
        if not qs:
            print(f"{mode:<14} {'NA':>5} {'NA':>8} {'NA':>8} {'NA':>10}")
            continue
        n_q = len(qs)
        prompt_m = round(mean(q["prompt_tokens"] for q in qs), 1)
        reason_m = round(mean(q["reasoning_length_mean"] for q in qs), 1)
        total_m = round(mean(q["total_tokens_mean"] for q in qs), 1)
        print(f"{mode:<14} {n_q:>5} {prompt_m:>8.1f} {reason_m:>8.1f} {total_m:>10.1f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="filter by mode, e.g. slm-normal, slm-guided, llm-guided")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model for master table (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", help="filter by dataset")
    parser.add_argument("--output-csv", help="write per-question summary to CSV")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print per-dataset and overall breakdowns",
    )
    args = parser.parse_args()

    paths = find_result_files()
    if not paths:
        print("No *_runs.json files found under *_results directories.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in paths:
        try:
            rows.extend(load_token_records(path))
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
            print(f"[WARN] skip {path}: {e}", file=sys.stderr)

    if args.mode:
        rows = [r for r in rows if r["mode"] == args.mode]
    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]
    if not rows:
        print("No records after filtering.", file=sys.stderr)
        sys.exit(1)

    table_model = args.model

    # group key for a question
    def qkey(r):
        return (r["mode"], r["subject"], r["dataset"], r["model"], r["extraction_model"], r["question"])

    by_question: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_question[qkey(r)].append(r)

    per_question = []
    for key, recs in sorted(by_question.items()):
        mode, subject, dataset, model, extraction_model, question = key
        prompt_tokens = recs[0]["prompt_tokens"]
        reasoning_vals = [r["reasoning_length"] for r in recs]
        total_vals = [r["total_tokens"] for r in recs]
        rl_mean, rl_std = _stats(reasoning_vals)
        tt_mean, tt_std = _stats(total_vals)
        per_question.append({
            "mode": mode,
            "subject": subject,
            "dataset": dataset,
            "model": model,
            "extraction_model": extraction_model or "",
            "question": question,
            "prompt_tokens": prompt_tokens,
            "reasoning_length_mean": round(rl_mean, 1),
            "reasoning_length_std": round(rl_std, 1),
            "total_tokens_mean": round(tt_mean, 1),
            "total_tokens_std": round(tt_std, 1),
            "n_runs": len({r["run_id"] for r in recs}),
        })

    print_master_table(per_question, table_model)

    if args.verbose:
        by_dataset: dict[tuple, list[dict]] = defaultdict(list)
        for r in per_question:
            by_dataset[(r["mode"], r["subject"], r["dataset"], r["model"], r["extraction_model"])].append(r)

        print("==== per-dataset summary ====")
        print(f"{'mode':<12} {'subject':<12} {'dataset':<22} {'model':<12} {'extractor':<12} "
              f"{'n_q':>4} {'prompt':>8} {'reason':>8} {'tokens':>8}")
        for key, qs in sorted(by_dataset.items()):
            mode, subject, dataset, model, extractor = key
            print(f"{mode:<12} {subject:<12} {dataset:<22} {model:<12} {extractor or '-':<12} "
                  f"{len(qs):>4} {round(mean(q['prompt_tokens'] for q in qs), 1):>8.1f} "
                  f"{round(mean(q['reasoning_length_mean'] for q in qs), 1):>8.1f} "
                  f"{round(mean(q['total_tokens_mean'] for q in qs), 1):>8.1f}")
        print()

    if args.output_csv:
        out = Path(args.output_csv)
        fields = list(per_question[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_question)
        print(f"\nWrote per-question CSV to {out}")


if __name__ == "__main__":
    main()
