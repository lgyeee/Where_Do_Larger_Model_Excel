import os
import json
import numpy as np
import argparse
import sys
import time
import random
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

base = Path(__file__).resolve()
INTERVENTION_DIR = base.parents[1]
# experiments_intervention/OpenRouter/evaluate.py -> parents[2] is the repo root
SRC_DIR = base.parents[2] / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api_utils import OPENROUTER_MODEL_MAP
from utils import (
    DATASET_MAP,
    extract_answer,
    verify_answer,
    _extract_boxed,
    _literal_eval_safe,
    _eq_cruxeval,
    _normalize_gold,
)
from tqdm import tqdm
from api_utils import make_openrouter_messages, make_sampling_params


def main():
    # =============== Config ===============
    load_dotenv(SRC_DIR.parent / ".env")
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENROUTER_API_KEY in env/.env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="MATH-500")
    parser.add_argument("--model_large", choices=OPENROUTER_MODEL_MAP.keys(), default="gpt-oss-120b")
    parser.add_argument("--model_small", choices=OPENROUTER_MODEL_MAP.keys(), default="gpt-oss-20b")
    parser.add_argument("--extraction_model", choices=OPENROUTER_MODEL_MAP.keys(), default="gpt-oss-120b")
    parser.add_argument(
        "--advantage_extractor",
        default="gemini-3-pro",
        help="src/advantage_descriptions/{subject}/{dataset}/{advantage_extractor}/... (default: gemini-3-pro)",
    )
    parser.add_argument("--mode", required=True,
                        choices=["slm-normal", "slm-guided", "llm-guided"],
                        help="Reasoning mode: 'slm-normal' (standard CoT), 'slm-guided' (SLM constraint extraction), 'llm-guided' (LLM constraint extraction)")
    parser.add_argument("--reasoning_effort", required=True) # different parameters for different models # see extract_constraints.sh
    parser.add_argument("--n_sample", type=int, default=1)
    args = parser.parse_args()

    # 1) ──── Load dataset ───────────────────────────────
    dataset_name, split = DATASET_MAP[args.dataset]["args"]
    config_name = DATASET_MAP[args.dataset].get("config")
    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    question_key = DATASET_MAP[args.dataset]["question_key"]
    answer_key = DATASET_MAP[args.dataset]["answer_key"]
    subject = DATASET_MAP[args.dataset]["subject"]

    # ===== output path =========
    # slm-normal does not use an extraction model; guided modes keep it in the path.
    # ============================
    if args.mode == "slm-normal":
        output_path = (
            INTERVENTION_DIR
            / f"{args.mode}_results"
            / subject
            / args.dataset
            / args.model_small
            / f"{args.n_sample}_runs.json"
        )
    else:
        output_path = (
            INTERVENTION_DIR
            / f"{args.mode}_results"
            / subject
            / args.dataset
            / args.model_small
            / args.extraction_model
            / f"{args.n_sample}_runs.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: output matches the summary schema produced by
    # experiments/groq/parse_batch_reasoning.py (top-level keys: "runs", "aggregate").
    # We keep an in-memory per-run mapping: runs_by_rid[rid][question] = record.
    runs_by_rid: dict[int, dict[str, dict]] = {rid: {} for rid in range(args.n_sample)}
    if output_path.exists():
        loaded = None
        try:
            with output_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load existing {output_path} ({e}); starting fresh.")

        if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
            for run in loaded["runs"]:
                rid = run.get("run_id")
                if not isinstance(rid, int) or rid < 0 or rid >= args.n_sample:
                    continue
                for rec in run.get("records", []):
                    q = rec.get("question") if isinstance(rec, dict) else None
                    if q:
                        runs_by_rid[rid][q] = rec
            n_existing = sum(len(v) for v in runs_by_rid.values())
            print(f"[RESUME] Found {n_existing} completed (run_id, question) records in {output_path}")
        elif loaded is not None:
            print(f"[WARN] Existing file {output_path} is not in summary schema; ignoring and starting fresh.")

    def _build_summary() -> dict:
        """Compose the summary dict in the target schema (runs + aggregate)."""
        all_questions: set[str] = set()
        for rid in range(args.n_sample):
            all_questions.update(runs_by_rid[rid].keys())

        hit_count_by_q = {
            q: sum(
                1 for rid in range(args.n_sample)
                if q in runs_by_rid[rid] and runs_by_rid[rid][q].get("correct")
            )
            for q in all_questions
        }
        hit_rate_by_q = {
            q: (hit_count_by_q[q] / args.n_sample * 100.0) if args.n_sample > 0 else 0.0
            for q in all_questions
        }

        # Stamp hit_rate / hit_count back onto each record so they appear inline.
        for rid in range(args.n_sample):
            for q, rec in runs_by_rid[rid].items():
                rec["hit_rate"] = hit_rate_by_q[q]
                rec["hit_count"] = hit_count_by_q[q]

        sorted_qs = sorted(all_questions)
        summary: dict = {"runs": [], "aggregate": {}}
        accs: list[float] = []
        lengths: list[float] = []
        for rid in range(args.n_sample):
            run_records = [runs_by_rid[rid][q] for q in sorted_qs if q in runs_by_rid[rid]]
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
        summary["aggregate"]["std_accuracy"]  = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        summary["aggregate"]["mean_length"]   = float(np.mean(lengths)) if lengths else 0.0
        summary["aggregate"]["std_length"]    = float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0
        summary["aggregate"]["per_question_hit_rate_mean"] = float(np.mean(hit_rates)) if hit_rates else 0.0
        summary["aggregate"]["per_question_hit_rate_std"]  = float(np.std(hit_rates, ddof=1)) if len(hit_rates) > 1 else 0.0
        return summary

    def _save_records():
        """Atomically dump the summary (runs + aggregate) to output_path."""
        summary = _build_summary()
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, output_path)

    # ===== filter dataset =========
    if args.dataset == "CRUXEVAL-O":
        input_key = DATASET_MAP[args.dataset]["input_key"]

    if args.dataset == "CRUXEVAL-I":
        output_key = DATASET_MAP[args.dataset]["output_key"]

    if args.dataset == "OMNI-MATH":
        # keep examples whose difficulty >= 7 and gold answer length <= 20
        difficulty_key = DATASET_MAP[args.dataset]["difficulty_key"]
        ds = ds.filter(lambda ex: ex[difficulty_key] >= 7 and len(str(ex[answer_key])) <= 20)

    if args.dataset == "gpqa-physics" or args.dataset == "gpqa-chemistry":
        fk = DATASET_MAP[args.dataset]["filter_key"]
        fv = DATASET_MAP[args.dataset]["filter_value"]
        ds = ds.filter(lambda ex: str(ex.get(fk, "")).lower() == str(fv).lower())

    if args.dataset == "JEEBENCH-PHYSICS" or args.dataset == "JEEBENCH-CHEMISTRY" or args.dataset == "JEEBENCH-MATH":
        type_key = DATASET_MAP[args.dataset]["type_key"]
        fk = DATASET_MAP[args.dataset]["filter_key"]
        fv = DATASET_MAP[args.dataset]["filter_value"]
        ds = ds.filter(lambda ex: ex[fk] == fv)

    if args.dataset == "OlympiadBench-physics":
        unit_key = DATASET_MAP[args.dataset]["unit_key"]
        ds = ds.filter(lambda ex: ex["answer_type"] == "Numerical")
        
    # 1.5) ──── retained question list ───────────────────────────────────────
    # advantage_descriptions lives in <repo>/src/advantage_descriptions/
    constraint_file = (
        SRC_DIR / "advantage_descriptions" / subject / args.dataset / args.advantage_extractor
        / f"{args.model_large}_vs_{args.model_small}_analysis.json"
    )
    # Read from the file as JSON
    # First confirm JSON structure.
    # According to file_context_0, for example in advantage_descriptions/chemistry/gpqa-chemistry/gemini-3-pro/gpt-oss-120b_vs_gpt-oss-20b_analysis.json
    # the format is:
    # {
    #   "questions": [
    #     {
    #       "question": "...",
    #       ... other fields ...
    #     },
    #     ...
    #   ]
    # }
    with open(constraint_file, "r") as f:
        loaded_json = json.load(f)

    # retained_questions: a list of kept questions
    if "questions" not in loaded_json or not isinstance(loaded_json["questions"], list):
        raise ValueError(f"Expected key 'questions' mapped to a list in {constraint_file}")
    retained_questions = [qobj["question"] for qobj in loaded_json["questions"] if "question" in qobj]

    constraints_list = []
    if args.mode == "slm-guided":
        constraints_file = (
            INTERVENTION_DIR
            / "constraints"
            / subject
            / args.dataset
            / f"{args.model_large}_vs_{args.model_small}"
            / f"{args.model_small}_extracted_constraints.json"
        )
        with constraints_file.open("r", encoding="utf-8") as f:
            constraints_list = json.load(f)
    elif args.mode == "llm-guided":
        constraints_file = (
            INTERVENTION_DIR
            / "constraints"
            / subject
            / args.dataset
            / f"{args.model_large}_vs_{args.model_small}"
            / f"{args.model_large}_extracted_constraints.json"
        )
        with constraints_file.open("r", encoding="utf-8") as f:
            constraints_list = json.load(f)

    # 2) ──── Initialize API Client ───────────────────────────────
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # 3) ──── Build messages ───────────────────────────────
    messages_list = []
    for q in retained_questions:
        # Find question in ds
        ex = ds.filter(lambda ex: ex[question_key] == q).to_list()[0]
        gold = ex[answer_key]

        # Default full_question is the raw question text; per-dataset branches below
        # may override it with extra context (code blocks, MCQ hints, units, etc.).
        full_question = q
        if args.dataset == "CRUXEVAL-O":
            code = ex[question_key]
            code_input = ex[input_key]
            full_question = f"""
            What should the output of this code be so that the assertion is correct? Reason step by step before
            arriving at an answer. Finally, surround the answer, with no additional words, with [ANSWER]
            and [/ANSWER] tags.
            {code}
            assert f({code_input}) == ?"""

        elif args.dataset == "CRUXEVAL-I":
            code = ex[question_key]
            code_output = ex[output_key]
            full_question = f"""
            You will be given a function f and an output in the form f(??) == output. Your task is to find any
            input such that executing f on the input leads to the given output. There may be multiple
            answers, but only output one. First, think step by step. Then, surround ONLY the INPUT VALUE
            with [ANSWER] and [/ANSWER] tags (do NOT include a function call).
            {code}
            assert f(??) == {code_output}
            """

        elif args.dataset == "JEEBENCH-PHYSICS" or args.dataset == "JEEBENCH-CHEMISTRY" or args.dataset == "JEEBENCH-MATH":
            if ex[type_key] == "MCQ(multiple)":
                full_question = f"{q}\n\nPlease reason step by step, and put your answer choices in ONE \\boxed{{}}. For example, if the answer is X, Y, and Z, output \\boxed{{XYZ}}."
            elif ex[type_key] == "MCQ":
                full_question = f"{q}\n\nPlease reason step by step, and put your answer choice in \\boxed{{}}."
            elif ex[type_key] == "Integer" or ex[type_key] == "Numeric":
                full_question = f"{q}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."

        elif args.dataset == "OlympiadBench-physics":
            unit = ex.get(unit_key, "")
            if unit:
                full_question = (
                    f"{q}\n\n"
                    f"The final answer must be expressed in units of **{unit}**.\n"
                    "Please reason step by step, and put **only the answer** but not units in \\boxed{{}}"
                )
            else:
                full_question = f"{q}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."

        else:
            full_question = f"Problem: {q}\n\n"

        if args.mode == "slm-normal":
            prompt = f"""
            Please list all the explicit and derived implicit constraints for the problem. Reasoning using the constraints to check compliance at each step to prune unnecessary search space.
            Problem: {full_question}
            """
        elif args.mode == "slm-guided" or args.mode == "llm-guided":
            # Find constraints for this question.
            constraints_record = next((c for c in constraints_list if c["question"] == q), None)
            if constraints_record is None:
                raise ValueError(f"No constraints found for question: {q}")
            constraints = constraints_record["constraints"]
            prompt = f"""
            {full_question}
            Reason using explicit/implicit constraints, checking compliance at each step to prune unnecessary search space.

            Explicit constraints:
            {constraints.get("explicit", [])}

            Implicit constraints:
            {constraints.get("implicit", [])}
            """
        messages_list.append({
            "prompt": prompt,
            "question": q,
            "gold": gold,
        })
    
    # 6) ──── Send API requests one at a time ──────────────────────────
    # For each retained question, call OpenRouter Chat Completions API,
    # extract message.content / reasoning, and append each to the in-memory records list.
    # After every completed question we re-dump the full JSON list to output_path (atomic).
    # If a record with the same question text already exists in output, skip it.
    # Now, change: If failed, just retry forever until succeed.

    n_done = n_skipped = 0
    for rid in range(args.n_sample):
        print(f"\n=== Run {rid + 1}/{args.n_sample} ===")
        for item in tqdm(messages_list, desc=f"run {rid}"):
            q = item["question"]
            gold = item["gold"]

            if q in runs_by_rid[rid]:
                n_skipped += 1
                continue

            small_model_id = OPENROUTER_MODEL_MAP[args.model_small]["model_id"]
            max_tokens = OPENROUTER_MODEL_MAP[args.model_small]["max_tokens"]
            sampling_params = make_sampling_params(max_tokens, args.reasoning_effort, args.model_small)
            body = make_openrouter_messages(item["prompt"], small_model_id, sampling_params)

            content = ""
            reasoning = ""
            reasoning_length = 0
            succeeded = False
            attempt = 1
            # --- retry with Retry-After header or exponential backoff until succeed ---
            while not succeeded:
                try:
                    response = client.chat.completions.create(**body)
                    choice = response.choices[0].message
                    content = choice.content or ""
                    reasoning = getattr(choice, "reasoning", None) or ""
                    usage = getattr(response, "usage", None)
                    reasoning_length = getattr(usage, "completion_tokens", 0) if usage else 0
                    succeeded = True
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    if status is not None and 400 <= status < 500 and status not in (429, 498):
                        raise

                    headers = getattr(getattr(e, "response", None), "headers", None) or {}
                    retry_after = headers.get("retry-after") or headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = float(retry_after)
                        except (TypeError, ValueError):
                            delay = min(60, 2 ** min(attempt - 1, 6))
                    else:
                        delay = min(60, 2 ** min(attempt - 1, 6))

                    delay *= random.uniform(0.8, 1.2)  # jitter

                    print(
                        f"[ERROR] {args.dataset}-{q} run={rid}"
                        f" attempt {attempt}: {e}; retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    attempt += 1


            # ── Parse model output and verify against gold ────────────────
            full_response = f"<think>\n{reasoning}\n</think>\n{content}"

            pred_display = None
            gold_display = gold
            correct = False
            if succeeded:
                gold_for_check = gold
                if args.dataset == "OlympiadBench-physics":
                    gold_for_check = _normalize_gold(gold)

                if args.dataset in ("CRUXEVAL-O", "CRUXEVAL-I"):
                    pred = _extract_boxed(full_response)
                    gold_obj = _literal_eval_safe(gold_for_check)
                    pred_obj = _literal_eval_safe(pred)
                    try:
                        correct = _eq_cruxeval(pred_obj, gold_obj)
                    except Exception:
                        correct = False
                    pred_display = repr(pred_obj)
                    gold_display = repr(gold_obj)
                else:
                    pred = extract_answer(full_response)
                    try:
                        correct = verify_answer(gold_for_check, pred)
                    except Exception:
                        correct = False
                    pred_display = pred
                    gold_display = gold_for_check

            runs_by_rid[rid][q] = {
                "question": q,
                "full_response": full_response,
                "reasoning_length": reasoning_length,
                "prediction": pred_display,
                "gold": gold_display,
                "correct": correct,
            }
            n_done += 1
            _save_records()

    final = _build_summary()
    print(f"\nWrote results to {output_path}")
    print(f"  done={n_done}, skipped={n_skipped}")
    if final["runs"]:
        accs = [r["accuracy"] for r in final["runs"]]
        print(f"  Per-run accuracies: {[round(a, 2) for a in accs]}")
        print(f"  Mean ± std accuracy: {final['aggregate']['mean_accuracy']:.2f}% ± {final['aggregate']['std_accuracy']:.2f}%")
        print(f"  Mean ± std length:   {final['aggregate']['mean_length']:.1f} ± {final['aggregate']['std_length']:.1f} tokens")


if __name__ == "__main__":
    main()
