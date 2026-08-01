import os
import json
import re
import argparse
import sys
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

base = Path(__file__).resolve()
INTERVENTION_DIR = base.parent
REPO_ROOT = base.parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api_utils import OPENROUTER_MODEL_MAP
from utils import DATASET_MAP, MODEL_MAP
from tqdm import tqdm
from api_utils import make_openrouter_messages, make_sampling_params


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_model_json(text: str):
    """
    Try to recover a JSON object from a model response. Returns parsed object
    on success, or None on failure (caller should keep the raw text instead).
    Handles: empty/whitespace-only strings, direct JSON, ```json ... ``` fences.
    """
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None


def is_clean_constraints(parsed) -> bool:
    """Validate the minimal constraints schema expected by evaluation.py."""
    if not isinstance(parsed, dict):
        return False
    if not isinstance(parsed.get("explicit"), list):
        return False
    if not isinstance(parsed.get("implicit"), list):
        return False
    return True


def main():
    # =============== Config ===============
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="MATH-500")
    parser.add_argument("--model_large", choices=MODEL_MAP.keys(), default="gpt-oss-120b")
    parser.add_argument("--model_small", choices=MODEL_MAP.keys(), default="gpt-oss-20b")
    parser.add_argument(
        "--extraction_model",
        choices=OPENROUTER_MODEL_MAP.keys(),
        default="gpt-oss-120b",
    )
    parser.add_argument(
        "--advantage_extractor",
        default="gemini-3-pro",
        help="src/advantage_descriptions/{subject}/{dataset}/{advantage_extractor}/... (default: gemini-3-pro)",
    )
    parser.add_argument("--reasoning_effort", required=True) # different parameters for different models # see extract_constraints.sh
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENROUTER_API_KEY in env/.env")

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
    # For clarity, use: constraints/{subject}/{args.dataset}/{args.model_large}_vs_{args.model_small}/{args.extraction_model}_extracted_constraints.json
    # ============================
    output_path = (
        INTERVENTION_DIR
        / "constraints"
        / subject
        / args.dataset
        / f"{args.model_large}_vs_{args.model_small}"
        / f"{args.extraction_model}_extracted_constraints.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing JSON list (if any) and skip questions already saved
    records: list[dict] = []
    done_questions: set[str] = set()
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                records = loaded
                done_questions = {r["question"] for r in records if isinstance(r, dict) and r.get("question")}
        except Exception as e:
            print(f"[WARN] Failed to load existing {output_path} ({e}); starting fresh.")
            records = []
            done_questions = set()
        print(f"[RESUME] Found {len(done_questions)} completed records in {output_path}")

    def _save_records():
        """Atomically dump the in-memory list to output_path."""
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
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
    if not constraint_file.exists():
        print(f"[SKIP] No advantage questions file found: {constraint_file}")
        return
    with open(constraint_file, "r") as f:
        loaded_json = json.load(f)

    # retained_questions: a list of kept questions
    if "questions" not in loaded_json or not isinstance(loaded_json["questions"], list):
        raise ValueError(f"Expected key 'questions' mapped to a list in {constraint_file}")
    retained_questions = [qobj["question"] for qobj in loaded_json["questions"] if "question" in qobj]

    # 2) ──── Load model config  ───────────────────────────────
    extraction_model_id = OPENROUTER_MODEL_MAP[args.extraction_model]["model_id"]
    max_tokens = OPENROUTER_MODEL_MAP[args.extraction_model]["max_tokens"]

    # 3) ──── Initialize OpenRouter client ───────────────────────────────
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # 4) ──── Prepare Sampling Parameters ───────────────────────────────
    sampling_params = make_sampling_params(max_tokens, args.reasoning_effort, args.extraction_model)

    # 5) ──── Build messages ───────────────────────────────
    messages_list = []
    for q in retained_questions:
        # Find question in ds
        ex = ds.filter(lambda ex: ex[question_key] == q).to_list()[0]
        # Default full_question is the raw question text; per-dataset branches below
        # may override it with extra context (code blocks, MCQ hints, units, etc.).
        full_question = q
        if args.dataset == "CRUXEVAL-O":
            code = ex[question_key]
            code_input = ex[input_key]
            full_question = f"""
            What should the output of this code be so that the assertion is correct? 
            {code}
            assert f({code_input}) == ?"""

        elif args.dataset == "CRUXEVAL-I":
            code = ex[question_key]
            code_output = ex[output_key]
            full_question = f"""
            You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. 
            {code}
            assert f(??) == {code_output}
            """

        elif args.dataset == "JEEBENCH-PHYSICS" or args.dataset == "JEEBENCH-CHEMISTRY" or args.dataset == "JEEBENCH-MATH":

            if ex[type_key] == "MCQ(multiple)":
                full_question = f"{q}\n\n This is a Multiple-selection question."
            
            elif ex[type_key] == "MCQ":
                full_question = f"{q}\n\n This is a Single-choice question."
            
            elif ex[type_key] == "Integer" or ex[type_key] == "Numeric":
                full_question = f"{q}\n\n"

        elif args.dataset == "OlympiadBench-physics":
            unit = ex.get(unit_key, "")
            if unit:
                full_question = (
                    f"{q}\n\n"
                    f"The final answer must be expressed in units of **{unit}**.\n"
                )
            else:
                full_question = f"{q}\n\n"
        
        prompt = f"""
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

        messages_list.append({
            "prompt": prompt,
            "question": q,
            "gold": ex[answer_key],
        })
    
    # 6) ──── Send API requests one at a time via OpenRouter ─────────────────
    # After every completed question we re-dump the full JSON list to output_path (atomic).
    # If a record with the same question text already exists in output, skip it.
    # If failed, retry up to 10 times before giving up.

    n_done = n_skipped = n_failed = 0
    for item in tqdm(messages_list):
        q = item["question"]

        if q in done_questions:
            n_skipped += 1
            continue

        body = make_openrouter_messages(item["prompt"], extraction_model_id, sampling_params)

        max_attempts = 10
        content = ""
        reasoning = ""
        completion_tokens = 0
        prompt_tokens = 0
        parsed = None
        succeeded = False
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(**body)
                choice = response.choices[0].message
                content = choice.content or ""
                reasoning = getattr(choice, "reasoning", None) or ""
                usage = getattr(response, "usage", None)
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0

                parsed = parse_model_json(content)
                if is_clean_constraints(parsed):
                    succeeded = True
                    break

                print(f"[WARN] {args.dataset}-{q} attempt {attempt}: invalid constraints JSON/schema, retrying")
            except Exception as e:
                print(f"[ERROR] {args.dataset}-{q} attempt {attempt}: {e}")

            if attempt == max_attempts:
                n_failed += 1

        if not succeeded:
            parsed = {"explicit": [], "implicit": []}
            print(f"[WARN] {args.dataset}: failed to get clean constraints after {max_attempts} attempts; keeping last raw text")

        records.append({
            "question": q,
            "constraints": parsed,
            "content": content + "<reasoning>\n" + reasoning + "\n</reasoning>",
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "constraints_parse_failed": not succeeded,
        })
        done_questions.add(q)
        n_done += 1
        _save_records()

    print(f"Wrote {n_done} new records to {output_path} (skipped={n_skipped}, failed={n_failed})")


main()
