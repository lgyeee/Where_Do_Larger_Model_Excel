"""Shared helpers for patching OpenRouter 10_runs.json result files."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

base = Path(__file__).resolve()
INTERVENTION_DIR = base.parents[1]
SRC_DIR = base.parents[2] / "src"

EXTRACTION_MODEL_BY_MODE = {
    "slm-guided": "gpt-oss-20b",
    "llm-guided": "gpt-oss-120b",
}

MODES = ("slm", "slm-normal", "slm-guided", "llm-guided")


def result_path(
    mode: str,
    subject: str,
    dataset: str,
    model_small: str,
    extraction_model: str | None = None,
    n_sample: int = 10,
) -> Path:
    root = INTERVENTION_DIR / f"{mode}_results" / subject / dataset / model_small
    if mode in ("slm-guided", "llm-guided"):
        if extraction_model is None:
            extraction_model = EXTRACTION_MODEL_BY_MODE[mode]
        root = root / extraction_model
    return root / f"{n_sample}_runs.json"


def parse_full_response(full_response: str) -> tuple[str, str]:
    marker = "</think>"
    if marker in full_response:
        thinking_part, content = full_response.split(marker, 1)
        reasoning = thinking_part.removeprefix("<think>\n")
        return reasoning, content.lstrip("\n")
    return "", full_response


def is_empty_failed_record(rec: dict) -> bool:
    """True when the API returned an empty/failed completion."""
    if rec.get("reasoning_length", 0) > 0:
        return False
    reasoning, content = parse_full_response(rec.get("full_response", ""))
    if reasoning.strip() or content.strip():
        return False
    return rec.get("prediction") is None


def needs_token_estimate(rec: dict) -> bool:
    """True when completion text exists but reasoning_length was not reported."""
    if rec.get("reasoning_length", 0) > 0:
        return False
    reasoning, content = parse_full_response(rec.get("full_response", ""))
    return bool(reasoning.strip() or content.strip())


def load_runs_by_rid(path: Path) -> tuple[dict[int, dict[str, dict]], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs_by_rid: dict[int, dict[str, dict]] = {}
    max_rid = -1
    for run in data.get("runs", []):
        rid = run.get("run_id")
        if not isinstance(rid, int):
            continue
        max_rid = max(max_rid, rid)
        bucket = runs_by_rid.setdefault(rid, {})
        for rec in run.get("records", []):
            q = rec.get("question")
            if q:
                bucket[q] = rec
    n_sample = max_rid + 1 if max_rid >= 0 else 0
    return runs_by_rid, n_sample


def build_summary(runs_by_rid: dict[int, dict[str, dict]], n_sample: int) -> dict:
    all_questions: set[str] = set()
    for rid in range(n_sample):
        all_questions.update(runs_by_rid.get(rid, {}).keys())

    hit_count_by_q = {
        q: sum(
            1 for rid in range(n_sample)
            if q in runs_by_rid.get(rid, {}) and runs_by_rid[rid][q].get("correct")
        )
        for q in all_questions
    }
    hit_rate_by_q = {
        q: (hit_count_by_q[q] / n_sample * 100.0) if n_sample > 0 else 0.0
        for q in all_questions
    }

    for rid in range(n_sample):
        for q, rec in runs_by_rid.get(rid, {}).items():
            rec["hit_rate"] = hit_rate_by_q[q]
            rec["hit_count"] = hit_count_by_q[q]

    sorted_qs = sorted(all_questions)
    summary: dict = {"runs": [], "aggregate": {}}
    accs: list[float] = []
    lengths: list[float] = []
    for rid in range(n_sample):
        run_records = [runs_by_rid[rid][q] for q in sorted_qs if q in runs_by_rid.get(rid, {})]
        if not run_records:
            continue
        acc = sum(1 for r in run_records if r.get("correct")) / len(run_records) * 100.0
        avg_len = sum(r.get("reasoning_length", 0) for r in run_records) / len(run_records)
        accs.append(acc)
        lengths.append(avg_len)
        summary["runs"].append({
            "run_id": rid,
            "accuracy": acc,
            "avg_length": avg_len,
            "records": run_records,
        })

    hit_rates = [hit_rate_by_q[q] for q in sorted_qs]
    summary["aggregate"]["mean_accuracy"] = float(np.mean(accs)) if accs else 0.0
    summary["aggregate"]["std_accuracy"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    summary["aggregate"]["mean_length"] = float(np.mean(lengths)) if lengths else 0.0
    summary["aggregate"]["std_length"] = float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0
    summary["aggregate"]["per_question_hit_rate_mean"] = float(np.mean(hit_rates)) if hit_rates else 0.0
    summary["aggregate"]["per_question_hit_rate_std"] = float(np.std(hit_rates, ddof=1)) if len(hit_rates) > 1 else 0.0
    return summary


def atomic_save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def backup_file(path: Path, backup_root: Path | None = None) -> Path:
    if backup_root is None:
        backup_root = INTERVENTION_DIR / "backups" / "10_runs" / time.strftime("%Y%m%d_%H%M%S")
    rel = path.relative_to(INTERVENTION_DIR)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst


def backup_all_mode_results(backup_root: Path | None = None) -> Path:
    if backup_root is None:
        backup_root = INTERVENTION_DIR / "backups" / "10_runs" / time.strftime("%Y%m%d_%H%M%S")
    copied = 0
    for mode in MODES:
        mode_dir = INTERVENTION_DIR / f"{mode}_results"
        if not mode_dir.is_dir():
            continue
        for path in sorted(mode_dir.rglob("10_runs.json")):
            backup_file(path, backup_root)
            copied += 1
    print(f"[BACKUP] Copied {copied} files to {backup_root}")
    return backup_root
