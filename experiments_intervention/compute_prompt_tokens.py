"""Reconstruct experiment input prompts and count their tokens.

Does NOT call any model for inference. For each dataset × mode it:
  1. Rebuilds the same prompts used at evaluation time
  2. Tokenizes them with the model's chat template
  3. Optionally merges with existing 10-run result files to get
     total_tokens = prompt_tokens + reasoning_length

Modes
-----
slm         : single-shot answer prompt (baseline)
slm-normal  : answer prompt wrapped with an in-prompt constraint hint
slm-guided  : stage1 extract constraints (small model) + stage2 reason with them
llm-guided  : same two stages, but stage1 uses --extraction_model (default: large)

Token aggregates (per benchmark, prompt_tokens.json → aggregate)
----------------------------------------------------------------
  num_questions, mean_prompt_tokens
  mean_length, mean_total_tokens, std_length, std_total_tokens
  mean_accuracy, std_accuracy              (0–1; analyze.py: mean ± std of per-run accuracy)
  mean_stage1_tokens, mean_stage2_tokens   (guided modes only)

Per-question fields include hit_rate (% correct across runs, same as evaluate.py).

Terminal summary: micro-average over all questions (not over #benchmarks);
for each of the 10 runs, pool all questions → one accuracy / token value,
then report mean ± std across runs (same convention as analyze.py);
saved to prompt_tokens/summary_{model}.json.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

from datasets import load_dataset


base = Path(__file__).resolve()
REPO_ROOT = base.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api_utils import OPENROUTER_MODEL_MAP  # type: ignore[import-not-found] # noqa: E402
from utils import DATASET_MAP, MODEL_MAP  # type: ignore[import-not-found] # noqa: E402


# Path roots (experiments_intervention/)
EXPERIMENTS_ROOT = base.parent
RESULTS_ROOT = EXPERIMENTS_ROOT          # {mode}_results/...
CONSTRAINTS_ROOT = EXPERIMENTS_ROOT / "constraints"
ADVANTAGE_ROOT = SRC_DIR / "advantage_descriptions"
OUTPUT_ROOT = EXPERIMENTS_ROOT / "prompt_tokens"


MODES = ("slm", "slm-normal", "slm-guided", "llm-guided")

# Default 10 benchmarks used in the intervention experiments.
DEFAULT_DATASETS = (
    "OMNI-MATH",
    "HHMT",
    "JEEBENCH-MATH",
    "gpqa-physics",
    "JEEBENCH-PHYSICS",
    "OlympiadBench-physics",
    "JEEBENCH-CHEMISTRY",
    "gpqa-chemistry",
    "CRUXEVAL-O",
    "CRUXEVAL-I",
)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def model_id_for(model_name: str) -> str:
    """Map short model name (e.g. qwen3-8b) → HuggingFace / OpenRouter model id."""
    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name]
    if model_name in OPENROUTER_MODEL_MAP:
        return OPENROUTER_MODEL_MAP[model_name]["model_id"]
    raise ValueError(f"Unknown model: {model_name}")


class TokenCounter:
    """Lazy-load and cache tokenizers; count chat-formatted prompt lengths."""

    def __init__(self):
        self._tokenizers = {}

    def tokenizer(self, model_name: str):
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        if model_name not in self._tokenizers:
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_id_for(model_name))
        return self._tokenizers[model_name]

    def count_chat_prompt(self, prompt: str, model_name: str) -> int:
        """Render as a user chat turn (with generation prompt) then encode."""
        tokenizer = self.tokenizer(model_name)
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Dataset loading (mirrors evaluation filters)
# ---------------------------------------------------------------------------

def load_filtered_dataset(dataset_name: str):
    """Load HF dataset and apply the same filters used in evaluation.py."""
    ds_name, split = DATASET_MAP[dataset_name]["args"]
    config_name = DATASET_MAP[dataset_name].get("config")
    ds = load_dataset(ds_name, config_name, split=split) if config_name else load_dataset(ds_name, split=split)

    answer_key = DATASET_MAP[dataset_name]["answer_key"]
    if dataset_name == "OMNI-MATH":
        difficulty_key = DATASET_MAP[dataset_name]["difficulty_key"]
        ds = ds.filter(lambda ex: ex[difficulty_key] >= 7 and len(str(ex[answer_key])) <= 20)

    if dataset_name in ("gpqa-physics", "gpqa-chemistry"):
        fk = DATASET_MAP[dataset_name]["filter_key"]
        fv = DATASET_MAP[dataset_name]["filter_value"]
        ds = ds.filter(lambda ex: str(ex[fk]).lower() == str(fv).lower())

    if dataset_name in ("JEEBENCH-PHYSICS", "JEEBENCH-CHEMISTRY", "JEEBENCH-MATH"):
        fk = DATASET_MAP[dataset_name]["filter_key"]
        fv = DATASET_MAP[dataset_name]["filter_value"]
        ds = ds.filter(lambda ex: ex[fk] == fv)

    if dataset_name in ("MMLU-Pro-math", "MMLU-Pro-physics", "MMLU-Pro-chemistry"):
        fk = DATASET_MAP[dataset_name]["filter_key"]
        fv = DATASET_MAP[dataset_name].get("filter_value", "math")
        ds = ds.filter(lambda ex: ex[fk] == fv)
        ds = ds.select(range(min(20, len(ds))))

    if dataset_name == "OlympiadBench-physics":
        ds = ds.filter(lambda ex: ex["answer_type"] == "Numerical")

    return ds


# ---------------------------------------------------------------------------
# Prompt builders (must stay in sync with evaluation.py)
# ---------------------------------------------------------------------------

def answer_full_question(dataset_name: str, ex: dict, baseline_slm: bool = False) -> str:
    """Full answer-prompt: question text + dataset-specific answer-format instructions.

    Used by slm / slm-normal / guided stage2.
    """
    question_key = DATASET_MAP[dataset_name]["question_key"]
    q = ex[question_key]

    if dataset_name == "CRUXEVAL-O":
        code_input = ex[DATASET_MAP[dataset_name]["input_key"]]
        return f"""
            What should the output of this code be so that the assertion is correct? Reason step by step before
            arriving at an answer. Finally, surround the answer, with no additional words, with [ANSWER]
            and [/ANSWER] tags.
            {q}
            assert f({code_input}) == ?"""

    if dataset_name == "CRUXEVAL-I":
        code_output = ex[DATASET_MAP[dataset_name]["output_key"]]
        return f"""
            You will be given a function f and an output in the form f(??) == output. Your task is to find any
            input such that executing f on the input leads to the given output. There may be multiple
            answers, but only output one. First, think step by step. Then, surround ONLY the INPUT VALUE
            with [ANSWER] and [/ANSWER] tags (do NOT include a function call).
            {q}
            assert f(??) == {code_output}
            """

    if dataset_name in ("JEEBENCH-PHYSICS", "JEEBENCH-CHEMISTRY", "JEEBENCH-MATH"):
        type_key = DATASET_MAP[dataset_name]["type_key"]
        if ex[type_key] == "MCQ(multiple)":
            return f"{q}\n\nPlease reason step by step, and put your answer choices in ONE \\boxed{{}}. For example, if the answer is X, Y, and Z, output \\boxed{{XYZ}}."
        if ex[type_key] == "MCQ":
            return f"{q}\n\nPlease reason step by step, and put your answer choice in \\boxed{{}}."
        if ex[type_key] in ("Integer", "Numeric"):
            return f"{q}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."
        return q

    if dataset_name == "OlympiadBench-physics":
        unit = ex.get(DATASET_MAP[dataset_name]["unit_key"], "")
        if unit:
            return (
                f"{q}\n\n"
                f"The final answer must be expressed in units of **{unit}**.\n"
                "Please reason step by step, and put **only the answer** but not units in \\boxed{{}}"
            )
        return f"{q}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."

    # Generic math/science templates (slight wording difference for baseline slm).
    if baseline_slm:
        return f"Problem: {q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return f"Problem: {q}\n\n Please reason step by step, and put your answer in \\boxed{{}}."


def constraint_full_question(dataset_name: str, ex: dict) -> str:
    """Shorter question stem for stage1 constraint extraction (no answer-format boilerplate)."""
    question_key = DATASET_MAP[dataset_name]["question_key"]
    q = ex[question_key]

    if dataset_name == "CRUXEVAL-O":
        code_input = ex[DATASET_MAP[dataset_name]["input_key"]]
        return f"""
            What should the output of this code be so that the assertion is correct? 
            {q}
            assert f({code_input}) == ?"""

    if dataset_name == "CRUXEVAL-I":
        code_output = ex[DATASET_MAP[dataset_name]["output_key"]]
        return f"""
            You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. 
            {q}
            assert f(??) == {code_output}
            """

    if dataset_name in ("JEEBENCH-PHYSICS", "JEEBENCH-CHEMISTRY", "JEEBENCH-MATH"):
        type_key = DATASET_MAP[dataset_name]["type_key"]
        if ex[type_key] == "MCQ(multiple)":
            return f"{q}\n\n This is a Multiple-selection question."
        if ex[type_key] == "MCQ":
            return f"{q}\n\n This is a Single-choice question."
        if ex[type_key] in ("Integer", "Numeric"):
            return f"{q}\n\n"
        return q

    if dataset_name == "OlympiadBench-physics":
        unit = ex.get(DATASET_MAP[dataset_name]["unit_key"], "")
        if unit:
            return f"{q}\n\nThe final answer must be expressed in units of **{unit}**.\n"
        return f"{q}\n\n"

    return q


def slm_normal_prompt(full_question: str) -> str:
    """Single-shot prompt that asks the model to list constraints while solving."""
    return f"""
            Please list all the explicit and derived implicit constraints for the problem. Reasoning using the constraints to check compliance at each step to prune unnecessary search space.
            Problem: {full_question}
            """


def extraction_prompt(full_question: str) -> str:
    """Stage1 prompt: extract explicit/implicit constraints as JSON (do not solve)."""
    return f"""
        You are an expert mathematical and logical strategist. Your objective is to extract and reformulate constraints to prune the search space for a downstream reasoning system.
        CRITICAL ANTI-LEAKAGE RULES:
        1. DO NOT solve the problem. DO NOT generate step-by-step solutions, intermediate execution traces, or the final answer.
        2. You may compute global structural numbers (e.g., total initial area, parity, conservation bounds) based on the initial state. You MUST NOT compute local state numbers that depend on hypothetical intermediate steps.

        For the given problem:
        1. Extract explicit constraints directly mentioned.
        2. Derive implicit conditions and hidden invariants (e.g., via invariant parameterization, symmetry, conservation laws) that must logically hold.
        
        Problem: {full_question}

        Output STRICTLY in the following JSON format:
        {{
          "explicit": [
              {{"constraint": "brief constraint description"}},
              {{"constraint": "brief constraint description"}},
              ...
          ],
          "implicit": [
              {{"constraint": "brief constraint description", "rationale": "1-2 sentences explaining how this constraint is derived."}},
              {{"constraint": "brief constraint description", "rationale": "... explanation of how this constraint is derived ..."}},
              ...
          ]
        }}"""


def guided_prompt(full_question: str, constraints: dict) -> str:
    """Stage2 prompt: answer the question while using pre-extracted constraints.

    Mirrors evaluation.py: constraints are interpolated as Python lists.
    """
    return f"""
            {full_question}
            Reason using explicit/implicit constraints, checking compliance at each step to prune unnecessary search space.

            Explicit constraints:
            {constraints.get("explicit", [])}

            Implicit constraints:
            {constraints.get("implicit", [])}
            """


# ---------------------------------------------------------------------------
# I/O: retained questions, pre-extracted constraints, 10-run results
# ---------------------------------------------------------------------------

def load_retained_questions(
    subject: str,
    dataset_name: str,
    model_large: str,
    model_small: str,
    advantage_extractor: str = "gemini-3-pro",
) -> list[str]:
    """Questions kept after large-vs-small advantage filtering."""
    path = (
        ADVANTAGE_ROOT
        / subject
        / dataset_name
        / advantage_extractor
        / f"{model_large}_vs_{model_small}_analysis.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["question"] for item in data["questions"]]


def load_constraints(
    subject: str,
    dataset_name: str,
    model_large: str,
    model_small: str,
    extraction_model: str,
) -> dict:
    """Load pre-extracted constraints keyed by question text."""
    path = (
        CONSTRAINTS_ROOT
        / subject
        / dataset_name
        / f"{model_large}_vs_{model_small}"
        / f"{extraction_model}_extracted_constraints.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {item["question"]: item["constraints"] for item in records}


def summarize(records: list[dict]) -> dict:
    """Prompt-only aggregate: num_questions, mean_prompt_tokens, optional stage means."""
    totals = [r["prompt_tokens"] for r in records]
    out = {
        "num_questions": len(records),
        "mean_prompt_tokens": float(mean(totals)),
    }
    stage_summary_names = {
        "stage1_extract_constraints": "stage1",
        "stage2_reason_with_constraints": "stage2",
    }
    for stage, stage_name in stage_summary_names.items():
        vals = [r["stages"][stage]["prompt_tokens"] for r in records if stage in r.get("stages", {})]
        if vals:
            out[f"mean_{stage_name}_tokens"] = float(mean(vals))
    return out


def output_path(model: str, subject: str, dataset_name: str, mode: str, extraction_model: str | None) -> Path:
    """Where to write prompt_tokens.json under prompt_tokens/."""
    path = OUTPUT_ROOT / mode / subject / dataset_name / model
    if mode in ("slm-guided", "llm-guided"):
        path = path / extraction_model
    return path / "prompt_tokens.json"


def load_ten_runs(
    subject: str,
    dataset_name: str,
    mode: str,
    model: str,
    extraction_model: str | None,
) -> dict:
    path = RESULTS_ROOT / f"{mode}_results" / subject / dataset_name / model
    if mode in ("slm-guided", "llm-guided"):
        path = path / extraction_model
    with open(path / "10_runs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_stdev(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def compute_run_stats(prompt_records: list[dict], result_data: dict) -> tuple[dict, list[dict]]:
    """Merge 10-run outputs: token stats + accuracy (analyze.py run-mean; evaluate.py hit_rate).

    Returns
    -------
    aggregate : per-benchmark summary written into prompt_tokens.json
    per_run   : one dict per run, for cross-benchmark micro pooling
                {n_correct, n_attempts, sum_total_tokens, n_token_samples}
    """
    prompt_by_question = {rec["question"]: rec["prompt_tokens"] for rec in prompt_records}
    questions = set(prompt_by_question)

    lengths_by_q: dict[str, list[float]] = {}
    totals_by_q: dict[str, list[float]] = {}
    hit_count_by_q = {q: 0 for q in questions}
    run_accs: list[float] = []
    per_run: list[dict] = []
    n_runs = len(result_data["runs"])

    for run in result_data["runs"]:
        n_correct = 0
        n_attempts = 0
        sum_total_tokens = 0.0
        n_token_samples = 0

        for rec in run["records"]:
            question = rec["question"]
            if question not in questions:
                continue
            n_attempts += 1
            if rec["correct"]:
                n_correct += 1
                hit_count_by_q[question] += 1
            reasoning_length = rec["reasoning_length"]
            if reasoning_length <= 0:
                continue
            total = float(prompt_by_question[question] + reasoning_length)
            lengths_by_q.setdefault(question, []).append(float(reasoning_length))
            totals_by_q.setdefault(question, []).append(total)
            sum_total_tokens += total
            n_token_samples += 1

        if n_attempts:
            run_accs.append(n_correct / n_attempts)
        per_run.append({
            "n_correct": n_correct,
            "n_attempts": n_attempts,
            "sum_total_tokens": sum_total_tokens,
            "n_token_samples": n_token_samples,
        })

    if not lengths_by_q:
        raise ValueError("No valid runs with reasoning_length > 0")

    hit_rate_by_q = {q: hit_count_by_q[q] / n_runs * 100.0 for q in questions}
    mean_length_by_q = {q: mean(vals) for q, vals in lengths_by_q.items()}
    mean_total_by_q = {q: mean(vals) for q, vals in totals_by_q.items()}
    std_length_within_q = {q: _safe_stdev(vals) for q, vals in lengths_by_q.items()}
    std_total_within_q = {q: _safe_stdev(vals) for q, vals in totals_by_q.items()}

    for rec in prompt_records:
        q = rec["question"]
        rec["hit_rate"] = hit_rate_by_q[q]
        if q not in mean_length_by_q:
            continue
        rec["mean_reasoning_length"] = mean_length_by_q[q]
        rec["mean_total_tokens"] = mean_total_by_q[q]
        rec["std_reasoning_length_within_runs"] = std_length_within_q[q]
        rec["std_total_tokens_within_runs"] = std_total_within_q[q]
        rec["num_valid_runs"] = len(lengths_by_q[q])

    per_q_mean_length = list(mean_length_by_q.values())
    per_q_mean_total = list(mean_total_by_q.values())

    aggregate = {
        # Per-benchmark: mean ± std over runs for accuracy; over questions for tokens.
        "mean_accuracy": float(mean(run_accs)),
        "std_accuracy": _safe_stdev(run_accs),
        "mean_length": float(mean(per_q_mean_length)),
        "mean_total_tokens": float(mean(per_q_mean_total)),
        "std_length": _safe_stdev(per_q_mean_length),
        "std_total_tokens": _safe_stdev(per_q_mean_total),
        "num_questions": len(questions),
    }
    return aggregate, per_run


def _accumulate_per_run(pool: list[dict], per_run: list[dict]) -> None:
    """Add one dataset's per-run counts into the cross-benchmark pool (aligned by run index)."""
    while len(pool) < len(per_run):
        pool.append({
            "n_correct": 0,
            "n_attempts": 0,
            "sum_total_tokens": 0.0,
            "n_token_samples": 0,
        })
    for i, r in enumerate(per_run):
        pool[i]["n_correct"] += r["n_correct"]
        pool[i]["n_attempts"] += r["n_attempts"]
        pool[i]["sum_total_tokens"] += r["sum_total_tokens"]
        pool[i]["n_token_samples"] += r["n_token_samples"]


# ---------------------------------------------------------------------------
# Per-mode record building & writing
# ---------------------------------------------------------------------------

def build_mode_records(args, counter: TokenCounter, dataset_name: str, mode: str) -> tuple[list[dict], str | None]:
    """Build one prompt-token record per question for the given mode.

    Returns (records, extraction_model). extraction_model is set only for
    guided modes (used in the output path).
    """
    subject = DATASET_MAP[dataset_name]["subject"]
    question_key = DATASET_MAP[dataset_name]["question_key"]
    ds = load_filtered_dataset(dataset_name)
    ds_by_question = {ex[question_key]: ex for ex in ds}

    # Question set: all filtered examples, or retained advantage questions only.
    if args.question_source == "dataset":
        questions = list(ds_by_question)
    else:
        questions = load_retained_questions(
            subject,
            dataset_name,
            args.model_large,
            args.model_small,
            args.advantage_extractor,
        )
        questions = [q for q in questions if q in ds_by_question]

    # Guided modes need pre-extracted constraints; which extractor differs.
    constraints_by_question = {}
    extraction_model = None
    if mode == "slm-guided":
        extraction_model = args.model_small
        constraints_by_question = load_constraints(
            subject,
            dataset_name,
            args.model_large,
            args.model_small,
            extraction_model,
        )
    elif mode == "llm-guided":
        extraction_model = args.extraction_model
        constraints_by_question = load_constraints(
            subject,
            dataset_name,
            args.model_large,
            args.model_small,
            extraction_model,
        )

    records = []
    for question in questions:
        ex = ds_by_question[question]
        stages = {}

        if mode == "slm":
            # Baseline: one answer prompt → tokens counted with --model.
            prompt = answer_full_question(dataset_name, ex, baseline_slm=True)
            tokens = counter.count_chat_prompt(prompt, args.model)
            stages["slm"] = {"prompt_tokens": tokens}

        elif mode == "slm-normal":
            # Same as baseline but with an in-prompt "list constraints" instruction.
            prompt = slm_normal_prompt(answer_full_question(dataset_name, ex))
            tokens = counter.count_chat_prompt(prompt, args.model)
            stages["slm_normal"] = {"prompt_tokens": tokens}

        else:
            # Guided: stage1 (extract) + stage2 (reason with constraints).
            # prompt_tokens for the question = stage1 + stage2.
            constraints = constraints_by_question.get(question)
            if constraints is None:
                raise ValueError(f"No constraints found for dataset={dataset_name}, mode={mode}, question={question!r}")

            stage1_prompt = extraction_prompt(constraint_full_question(dataset_name, ex))
            stage1_tokens = counter.count_chat_prompt(stage1_prompt, extraction_model)
            stage2_prompt = guided_prompt(answer_full_question(dataset_name, ex), constraints)
            stage2_tokens = counter.count_chat_prompt(stage2_prompt, args.model)

            stages["stage1_extract_constraints"] = {
                "prompt_tokens": stage1_tokens,
            }
            stages["stage2_reason_with_constraints"] = {
                "prompt_tokens": stage2_tokens,
            }
            tokens = stage1_tokens + stage2_tokens

        record = {
            "question": question,
            "prompt_tokens": tokens,
            "stages": stages,
        }
        records.append(record)

    return records, extraction_model


def write_mode_file(
    args, dataset_name: str, mode: str, records: list[dict], extraction_model: str | None
) -> tuple[dict, list[dict]]:
    """Write prompt_tokens.json; return (aggregate, per_run) from 10_runs.json."""
    subject = DATASET_MAP[dataset_name]["subject"]
    path = output_path(args.model, subject, dataset_name, mode, extraction_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = summarize(records)
    result_data = load_ten_runs(subject, dataset_name, mode, args.model, extraction_model)
    run_stats, per_run = compute_run_stats(records, result_data)
    # summarize already has num_questions / mean_prompt_tokens; run_stats overwrites
    # num_questions with the same value and adds accuracy / token aggregates.
    aggregate.update(run_stats)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"questions": records, "aggregate": aggregate}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")
    return aggregate, per_run


def build_mode_summary(
    run_pool_by_mode: dict[str, list[dict]],
    num_questions_by_mode: dict[str, int],
) -> dict:
    """Micro-average: pool all questions into each run, then mean ± std over runs.

    For run i (across all datasets):
      accuracy_i = total_correct_i / total_attempts_i
      tokens_i   = mean of (prompt + reasoning) over valid samples in run i

    Then mean_accuracy / std_accuracy are mean ± stdev over the 10 run values
    (same convention as analyze.py, but questions are pooled across benchmarks).
    """
    micro: dict[str, dict] = {}
    for mode in MODES:
        pool = run_pool_by_mode.get(mode, [])
        if not pool:
            continue
        run_accs = [
            r["n_correct"] / r["n_attempts"]
            for r in pool
            if r["n_attempts"] > 0
        ]
        run_toks = [
            r["sum_total_tokens"] / r["n_token_samples"]
            for r in pool
            if r["n_token_samples"] > 0
        ]
        if not run_accs or not run_toks:
            continue
        micro[mode] = {
            "mean_accuracy": float(mean(run_accs)),
            "std_accuracy": _safe_stdev(run_accs),
            "mean_total_tokens": float(mean(run_toks)),
            "std_total_tokens": _safe_stdev(run_toks),
            "num_questions": num_questions_by_mode.get(mode, 0),
            "num_runs": len(run_accs),
        }
    return micro


def summary_table(micro: dict) -> list[dict]:
    return [
        {
            "mode": mode,
            "mean_total_tokens": micro[mode]["mean_total_tokens"],
            "std_total_tokens": micro[mode]["std_total_tokens"],
            "mean_accuracy": micro[mode]["mean_accuracy"],
            "std_accuracy": micro[mode]["std_accuracy"],
            "num_questions": micro[mode]["num_questions"],
            "num_runs": micro[mode]["num_runs"],
        }
        for mode in MODES
        if mode in micro
    ]


def print_mode_summary(micro: dict):
    """Print cross-benchmark micro mean ± std over runs (questions pooled)."""
    if not micro:
        return

    print()
    print(
        f"{'mode':<14} {'mean_total_tokens':>28} {'mean_accuracy':>22} "
        f"{'n_q':>8} {'n_runs':>8}"
    )
    for mode in MODES:
        if mode not in micro:
            continue
        row = micro[mode]
        toks = f"{row['mean_total_tokens']:.2f} ± {row['std_total_tokens']:.2f}"
        acc = f"{row['mean_accuracy']:.3f} ± {row['std_accuracy']:.3f}"
        print(
            f"{mode:<14} {toks:>28} {acc:>22} "
            f"{row['num_questions']:>8} {row['num_runs']:>8}"
        )


def save_mode_summary(path: Path, micro: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_table(micro), f, ensure_ascii=False, indent=2)
    print(f"Wrote summary {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute prompt token counts for slm/slm-normal/slm-guided/llm-guided prompts."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_MAP.keys(),
        action="append",
        help="Dataset to process. Can be repeated. Defaults to the 10 experiment datasets.",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--model", default=None, help="Current reasoning model. Defaults to --model_small.")
    parser.add_argument("--model_large", choices=MODEL_MAP.keys(), default="qwen3-32b")
    parser.add_argument("--model_small", choices=MODEL_MAP.keys(), default="qwen3-8b")
    parser.add_argument("--extraction_model", choices=MODEL_MAP.keys(), default=None)
    parser.add_argument(
        "--advantage_extractor",
        default="gemini-3-pro",
        help="src/advantage_descriptions/{subject}/{dataset}/{advantage_extractor}/... (default: gemini-3-pro)",
    )
    parser.add_argument(
        "--question-source",
        choices=("advantage", "dataset"),
        default="advantage",
        help="Use retained advantage questions, or all filtered dataset questions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = args.model or args.model_small
    args.extraction_model = args.extraction_model or args.model_large

    datasets = args.dataset or list(DEFAULT_DATASETS)
    counter = TokenCounter()
    # Per mode: accumulate (run_idx → counts) across datasets for micro over runs.
    run_pool_by_mode: dict[str, list[dict]] = {mode: [] for mode in MODES}
    num_questions_by_mode: dict[str, int] = {mode: 0 for mode in MODES}

    for dataset_name in datasets:
        for mode in args.modes:
            try:
                records, extraction_model = build_mode_records(args, counter, dataset_name, mode)
                aggregate, per_run = write_mode_file(
                    args, dataset_name, mode, records, extraction_model
                )
                _accumulate_per_run(run_pool_by_mode[mode], per_run)
                num_questions_by_mode[mode] += aggregate["num_questions"]
            except (FileNotFoundError, ValueError) as exc:
                print(f"[SKIP] dataset={dataset_name} mode={mode}: {exc}")

    micro = build_mode_summary(run_pool_by_mode, num_questions_by_mode)
    print_mode_summary(micro)
    save_mode_summary(OUTPUT_ROOT / f"summary_{args.model}.json", micro)


if __name__ == "__main__":
    main()
