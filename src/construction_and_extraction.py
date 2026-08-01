import os
import sys
import re
import json
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv
import requests
from utils import DATASET_MAP, MODEL_MAP
from api_utils import OPENROUTER_MODEL_MAP
from tqdm import tqdm
import random

# Prefer repo-root .env, then cwd override.
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
ADVANTAGE_ROOT = _SRC_DIR / "advantage_descriptions"
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = os.environ.get("OPENROUTER_URL")

# HF datasets repo id for published reasoning traces → copied into eval_outputs/.
HF_TRACES_REPO = "lgyeee/where-larger-models-excel-reasoning-traces"


def build_first_run_hit_rate_map(data: dict) -> dict:
    runs = data.get("runs") or []
    if not runs:
        return {}
    out = {}
    for rec in runs[0].get("records", []):
        q = rec.get("question")
        if q is not None and rec.get("hit_rate") is not None:
            out[q] = rec.get("hit_rate")
    return out


def filter_by_gap_qtext(llm_data, slm_data, gap_value: float) -> list[str]:
    """
    Return list of question texts where (llm_hit_rate - slm_hit_rate) / 100 >= gap_value.
    Uses only the first run's records and matches by question text.
    Does not recompute hit rate; relies on stored hit_rate in the records.
    """
    llm_map = build_first_run_hit_rate_map(llm_data)
    slm_map = build_first_run_hit_rate_map(slm_data)
    if not llm_map or not slm_map:
        return []

    gap_questions = []
    for q, llm_pr in llm_map.items():
        if q not in slm_map:
            continue
        slm_pr = slm_map[q]
        gap = (llm_pr - slm_pr) / 100.0
        if gap >= gap_value:
            gap_questions.append(q)

    return gap_questions


def build_question_stats(llm_data, slm_data) -> dict:
    """
    Build a dictionary keyed by question text with:
    {
       "<question text>": {
        "gold": <gold>,
        "llm_hit_rate": <first LLM run's hit_rate>,
        "slm_hit_rate": <first SLM run's hit_rate>,
        "llm_correct_runs": [run_id...],
        "llm_wrong_runs":   [run_id...],
        "slm_correct_runs": [run_id...],
        "slm_wrong_runs":   [run_id...],
        },
        ...
    }
    """
    llm_runs = llm_data.get("runs") or []
    slm_runs = slm_data.get("runs") or []

    llm_pr_map = build_first_run_hit_rate_map(llm_data)
    slm_pr_map = build_first_run_hit_rate_map(slm_data)

    stats = {}

    for run in llm_runs:
        rid = run.get("run_id")
        for rec in run.get("records", []):
            q = rec.get("question")
            if q is None:
                continue 
            entry = stats.setdefault(q, {
                "gold": rec.get("gold"),
                "llm_hit_rate": llm_pr_map.get(q),
                "slm_hit_rate": slm_pr_map.get(q),
                "llm_correct_runs": [],
                "llm_wrong_runs": [],
                "slm_correct_runs": [],
                "slm_wrong_runs": [],
            })
            if rec.get("correct"):
                entry["llm_correct_runs"].append(rid)
            else:
                entry["llm_wrong_runs"].append(rid)

    for run in slm_runs:
        rid = run.get("run_id")
        for rec in run.get("records", []):
            q = rec.get("question")
            if q is None:
                continue
            entry = stats.setdefault(q, {
                "gold": rec.get("gold"),
                "llm_hit_rate": llm_pr_map.get(q),
                "slm_hit_rate": slm_pr_map.get(q),
                "llm_correct_runs": [],
                "llm_wrong_runs": [],
                "slm_correct_runs": [],
                "slm_wrong_runs": [],
            })
            if rec.get("correct"):
                entry["slm_correct_runs"].append(rid)
            else:
                entry["slm_wrong_runs"].append(rid)

    return stats

def find_records_by_run_id(runs, run_id) -> list[dict]:
    for run in runs:
        if run.get("run_id") == run_id:
            return run.get("records", [])
    return []

def find_record_by_question(records, question) -> dict:
    for rec in records:
        if rec.get("question") == question:
            return rec
    return None


def compute_valid_g(q_stat: dict, n_sample: int) -> tuple[float | None, float | None, list, list, int, str | None]:
    llm_pr = q_stat.get("llm_hit_rate")
    slm_pr = q_stat.get("slm_hit_rate")
    if llm_pr is None or slm_pr is None:
        return llm_pr, slm_pr, [], [], 0, "missing hit_rate for question, skip"
    g = int(round((llm_pr - slm_pr) * n_sample / 100.0))
    if g <= 0:
        return llm_pr, slm_pr, [], [], 0, f"non-positive gap-derived G={g}, skip"
    llm_pool = q_stat.get("llm_correct_runs") or []
    slm_pool = q_stat.get("slm_wrong_runs") or []
    g = min(g, len(llm_pool), len(slm_pool))
    if g <= 0:
        return llm_pr, slm_pr, llm_pool, slm_pool, 0, "insufficient pool size for question, skip"
    return llm_pr, slm_pr, llm_pool, slm_pool, g, None


def parse_json_content(content: str):
    """Parse model output that may be wrapped in ```json ... ``` fences."""
    text = content.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def main():
    # 1) parse args & env check
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="MATH-500")
    parser.add_argument("--model_large", choices=MODEL_MAP.keys(), default="gpt-oss-120b")
    parser.add_argument("--model_small", choices=MODEL_MAP.keys(), default="gpt-oss-20b")
    parser.add_argument("--advantage_extractor", choices=OPENROUTER_MODEL_MAP.keys(), default="gemini-3-pro")
    parser.add_argument("--n-sample", type=int, default=2, help="which runs file to read (e.g., 2_runs.json)")
    parser.add_argument("--gap-value", type=float, default=0.6, help="gap ratio between LLM and SLM pass rates (e.g., 0.6 = 60%%)")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for LLM/SLM run-pair sampling (same gap+seed → same pairs across extractors).",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help=(
            "Download reasoning traces from Hugging Face into eval_outputs/ before running "
            f"(repo: {HF_TRACES_REPO})."
        ),
    )
    args = parser.parse_args()

    if not OPENROUTER_API_KEY or not OPENROUTER_URL:
        raise SystemExit("Please set OPENROUTER_API_KEY and OPENROUTER_URL in env/.env")
    random.seed(args.seed)
    extractor_model_id = OPENROUTER_MODEL_MAP[args.advantage_extractor]["model_id"]

    def build_output_payload(questions_map: dict) -> dict:
        """Include sampling metadata so released artifacts stay reproducible."""
        return {
            "meta": {
                "seed": args.seed,
                "gap_value": args.gap_value,
                "n_sample": args.n_sample,
                "dataset": args.dataset,
                "model_large": args.model_large,
                "model_small": args.model_small,
                "advantage_extractor": args.advantage_extractor,
                "advantage_extractor_model_id": extractor_model_id,
            },
            "questions": list(questions_map.values()),
        }

    Path("eval_outputs").mkdir(parents=True, exist_ok=True)
    if args.from_hf:
        from huggingface_hub import snapshot_download

        snapshot_root = Path(snapshot_download(repo_id=HF_TRACES_REPO, repo_type="dataset"))
        shutil.copytree(snapshot_root, "eval_outputs", dirs_exist_ok=True)

    # ===============================================
    # 1.1) locate input files
    # ===============================================
    subject = DATASET_MAP[args.dataset]["subject"]
    llm_file = Path(f"eval_outputs/{subject}/{args.dataset}/{args.model_large}/{args.n_sample}_runs.json")
    slm_file = Path(f"eval_outputs/{subject}/{args.dataset}/{args.model_small}/{args.n_sample}_runs.json")
    if not llm_file.exists() or not slm_file.exists():
        raise FileNotFoundError(f"LLM or SLM file not found: {llm_file} or {slm_file}")

    # ===============================================
    # 1.2) load data
    # ===============================================
    with llm_file.open("r", encoding="utf-8") as f:
        llm_data = json.load(f)
    with slm_file.open("r", encoding="utf-8") as f:
        slm_data = json.load(f)
        
    # ===============================================
    # 1.3) locate output file
    #     src/advantage_descriptions/{subject}/{dataset}/{extraction_model}/...
    # ===============================================
    output_file = (
        ADVANTAGE_ROOT
        / subject
        / args.dataset
        / args.advantage_extractor
        / f"{args.model_large}_vs_{args.model_small}_analysis.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    
    # ===============================================
    # 2.1) filter questions by pass rate thresholds
    # ===============================================
    gap_questions = filter_by_gap_qtext(
        llm_data,
        slm_data,
        gap_value=args.gap_value,
    )
    if not gap_questions:
        tqdm.write("[WARN] No questions passed the gap filter; nothing to analyze.")
        return  

    # ===============================================
    # 2.2) build question stats
    # ===============================================
    question_stats = build_question_stats(llm_data, slm_data)

    # ===============================================
    # 3.0) construct tqdm progress bar
    # ===============================================
    questions_map = {} 
    total_calls = 0 # total number of API calls
    for q in gap_questions:
        q_stat = question_stats.get(q) or {}
        _, _, _, _, g, _ = compute_valid_g(q_stat, args.n_sample)
        total_calls += g

    # [DISPLAY] the progress bar
    pbar = tqdm(
    total=total_calls, # total number of API calls
    desc=f"{args.dataset} API calls", # dataset name
    dynamic_ncols=True, 
    disable=False,
    file=sys.stdout
    )
    
    # ===============================================
    # 3.1) select pairs for analysis by gap value by random sampling
    # ===============================================
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    for q in gap_questions:
        q_stat = question_stats.get(q) or {}
        llm_pr, slm_pr, llm_pool, slm_pool, g, err_msg = compute_valid_g(q_stat, args.n_sample)
        if err_msg:
            tqdm.write(f"[WARN] {err_msg}")
            continue

        # random sample the llm-slm pairs for the question
        llm_selected_runs = random.sample(llm_pool, g)
        slm_selected_runs = random.sample(slm_pool, g)
        
        # [SETUP] the question entry
        q_entry = questions_map.setdefault(q, {
            "question": q,
            "gold": q_stat.get("gold"),
            "llm_hit_rate": llm_pr,
            "slm_hit_rate": slm_pr,
            "gap_g": g,
            "analysis": []
        })  
        
        # [MAIN LOOP] send the selected pairs to analysis API
        for i in range(g):
            llm_resp_id = llm_selected_runs[i]
            slm_resp_id = slm_selected_runs[i]
            llm_recs = find_records_by_run_id(llm_data["runs"], llm_resp_id)
            slm_recs = find_records_by_run_id(slm_data["runs"], slm_resp_id)
            
            llm_rec = find_record_by_question(llm_recs, q)
            slm_rec = find_record_by_question(slm_recs, q)
            # [GUARD CLAUSE] if the record is not found, skip the question
            if not llm_rec or not slm_rec:
                tqdm.write(f"[ERROR] no record found for question {q}")
                continue
            
            llm_reasoning = llm_rec.get("full_response", "")
            slm_reasoning = slm_rec.get("full_response", "")
            
            # [GUARD CLAUSE] if the reasoning is not found, skip the question
            if not llm_reasoning or not slm_reasoning:
                tqdm.write(f"[ERROR] no reasoning found for question {q}")
                continue
            
            prompt = f"""You are an LLM error analysis expert. Model_A is correct; Model_B is incorrect.

TASK: 
Compare their reasoning and do the following:
1. Identify the FIRST Newman's Error Analysis stage where Model_B fails.
Choose ONE: Reading | Comprehension | Transformation | Process Skills | Encoding
2. Briefly describe the specific reasoning failure at that stage.
3. From this failure, extract 2-5 advantage objects explaining why Model_A succeeds.

!!! RULES FOR 'advantage' FIELD !!!
1. ABSTRACTION: Generalize to universal reasoning skills (no problem-specific variables).
2. FORMAT: Start EXACTLY with an action verb (e.g., "Identifies", "Applies").
3. FORBIDDEN: NEVER use "Model_A", "Model_B", "The model", or "Correctly".
4. LENGTH: Maximum 10 words.

Problem: {q}
Model_A reasoning: {llm_reasoning}
Model_B reasoning: {slm_reasoning}

Output Format:
Each object MUST follow this exact schema:
** ONLY OUTPUT LIST OF OBJECTS, NO OTHER TEXT. **
[
  {{
    "type": "failure", 
    "failure_stage": "<ONE OF: Reading | Comprehension | Transformation | Process Skills | Encoding>",
    "failure_description": "<Brief description of Model_B's FIRST failure>"
  }},
  {{
    "type": "advantage",
    "advantage": "<Action verb + abstract skill (max 10 words)>",
    "evidence": "<Specific text contrast proving the advantage>"
  }},
  {{
    "type": "advantage",
    "advantage": "<another Action verb + abstract skill (max 10 words)>",
    "evidence": "<another specific text contrast proving the advantage>"
  }},...
]
"""
            messages = [{"role": "user", "content": prompt}]
            payload = {
                    "model": extractor_model_id,
                    "messages": messages,
                    "temperature": 0,
            }
            
            content, resp_json = "", {}
            for _ in range(5):
                try:
                    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60); resp.raise_for_status(); resp_json = resp.json(); content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", ""); break
                except Exception as e:
                    resp_json = {"error": str(e)}
            if content:
                tqdm.write(f"[INFO] dataset={args.dataset} llm_run_id={llm_resp_id} slm_run_id={slm_resp_id} q_len={len(q or '')} content_len={len(content)}")
            else:
                tqdm.write(
                    f"[ERROR] dataset={args.dataset} llm_run_id={llm_resp_id} slm_run_id={slm_resp_id} q=...: "
                    f"{resp_json.get('error', 'empty content after retries')}"
                )
            pbar.update(1)
            
            analysis_list = []
            """ analysis_list is a list of dict"""
            # ================================
            # Extract analysis from the response
            # ================================
            if content:
                try:
                    analysis_list = parse_json_content(content)
                except Exception as e:
                    tqdm.write(f"[ERROR] parse JSON failed: {e}")
                    tqdm.write(f"[DEBUG] content_repr[:300]={content[:300]!r}")

            q_entry["analysis"].append({
                "llm_run_id": llm_resp_id,
                "slm_run_id": slm_resp_id,
                "analysis": analysis_list
            })
            # flush after each question update
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(build_output_payload(questions_map), f, ensure_ascii=False, indent=2)
            tqdm.write(f"[INFO] flushed question {q} to {output_file}")

    # ===============================================
    # 4) save results
    # ===============================================
    analysis_data = build_output_payload(questions_map)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    tqdm.write(f"Saved analysis outputs to {output_file}")


if __name__ == "__main__":
    main()
