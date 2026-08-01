import os
import json
import numpy as np
import argparse
import hashlib
import sys
from pathlib import Path
from datasets import load_dataset
from groq import Groq
from dotenv import load_dotenv

base = Path(__file__).resolve()
for p in (base.parents[2] / "src", base.parents[3]):
    if p.exists():
        sys.path.insert(0, str(p))
        break

from api_utils import GROQ_MODEL_MAP
from utils import DATASET_MAP, MODEL_MAP, extract_answer, verify_answer
from tqdm import tqdm
from api_utils import make_groq_messages, make_sampling_params


def main():
    # =============== Config ===============
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="MATH-500")
    parser.add_argument("--model", choices=GROQ_MODEL_MAP.keys(), default="gpt-oss-20b")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="high")
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

    output_path = f"batch_files/{subject}/{args.dataset}/{args.model}/{args.n_sample}_runs.jsonl"
    if os.path.exists(output_path):
        print(f"[SKIP] Already exists: {output_path}")
        return

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

    if args.dataset == "MMLU-Pro-math" or args.dataset == "MMLU-Pro-physics" or args.dataset == "MMLU-Pro-chemistry":
        fk = DATASET_MAP[args.dataset]["filter_key"]
        fv = DATASET_MAP[args.dataset]["filter_value"]
        ds = ds.filter(lambda ex: ex[fk] == fv)
        ds = ds.select(range(20))
        options_key = DATASET_MAP[args.dataset]["options_key"]

    if args.dataset == "OlympiadBench-physics":
        unit_key = DATASET_MAP[args.dataset]["unit_key"]
        ds = ds.filter(lambda ex: ex["answer_type"] == "Numerical")

    # 2) ──── Load model config  ───────────────────────────────
    model_id = MODEL_MAP[args.model]
    max_tokens = GROQ_MODEL_MAP[args.model]["max_tokens"]

    # 3) ──── Initialize API Client ───────────────────────────────
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # 4) ──── Prepare Sampling Parameters ───────────────────────────────
    sampling_params = make_sampling_params(max_tokens, args.reasoning_effort, args.model)

    # 5) ──── Build messages ───────────────────────────────
    messages_list = []
    for ex in ds:
        q = ex[question_key]
        gold = ex[answer_key]
        if args.dataset == "CRUXEVAL-O":
            code = ex[question_key]
            code_input = ex[input_key]
            prompt = f"""
            What should the output of this code be so that the assertion is correct? Reason step by step before
            arriving at an answer. Finally, surround the answer, with no additional words, with [ANSWER]
            and [/ANSWER] tags.
            {code}
            assert f({code_input}) == ?"""

        elif args.dataset == "CRUXEVAL-I":
            code = ex[question_key]
            code_output = ex[output_key]
            prompt = f"""
            You will be given a function f and an output in the form f(??) == output. Your task is to find any
            input such that executing f on the input leads to the given output. There may be multiple
            answers, but only output one. First, think step by step. Then, surround ONLY the INPUT VALUE
            with [ANSWER] and [/ANSWER] tags (do NOT include a function call).
            {code}
            assert f(??) == {code_output}
            """

        elif args.dataset == "JEEBENCH-PHYSICS" or args.dataset == "JEEBENCH-CHEMISTRY" or args.dataset == "JEEBENCH-MATH":
            if ex[type_key] == "MCQ(multiple)":
                prompt = f"{q}\n\nPlease reason step by step, and put your answer choices in ONE \\boxed{{}}. For example, if the answer is X, Y, and Z, output \\boxed{{XYZ}}."
            elif ex[type_key] == "MCQ":
                prompt = f"{q}\n\nPlease reason step by step, and put your answer choice in \\boxed{{}}."
            elif ex[type_key] == "Integer" or ex[type_key] == "Numeric":
                prompt = f"{q}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."

        elif args.dataset == "OlympiadBench-physics":
            unit = ex.get(unit_key, "")
            if unit:
                prompt = (
                    f"{q}\n\n"
                    f"The final answer must be expressed in units of **{unit}**.\n"
                    "Please reason step by step, and put **only the answer** but not units in \\boxed{{}}"
                )
            else:
                prompt = f"{q}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."

        else:
            prompt = f"Problem: {q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        messages_list.append({
            "messages": make_groq_messages(prompt, model_id, sampling_params),
            "question": q,
            "gold": gold,
        })
    # 7) ──── Generate batch file ───────────────────────────────
    request_objects = []
    for item in tqdm(messages_list):
        messages = item["messages"]
        q = item["question"]
        for rid in range(args.n_sample):
            q_hash = hashlib.sha256(q.encode()).hexdigest()[:8]
            custom_id = f"{args.dataset}-{q_hash}-run-{rid}"
            # Construct the Request Object
            request_object = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": messages,
            }
            request_objects.append(request_object)
    # 8) ──── Save batch file ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for request_object in request_objects:
            f.write(json.dumps(request_object) + "\n")
    print(f"Wrote {args.n_sample}_runs.jsonl batch file to {output_path}")


main()
