import os
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import GenerationConfig, AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

base = Path(__file__).resolve()
INTERVENTION_DIR = base.parents[1]
SRC_DIR = base.parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import verify_answer, extract_answer, DATASET_MAP, MODEL_MAP


def apply_chat(prompt: str, tokenizer):
    """
    Wraps a user prompt in the vLLM chat template.
    """
    conversations = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True
    )

def make_params(n: int, budget: int, cfg) -> SamplingParams:
    """
    Build SamplingParams from model config and given budget.
    """
    kw = {"n": n, "max_tokens": budget}
    if hasattr(cfg, "temperature") and cfg.temperature is not None:
        kw["temperature"] = cfg.temperature
    if hasattr(cfg, "top_k") and cfg.top_k is not None:
        kw["top_k"] = cfg.top_k
    if hasattr(cfg, "top_p") and cfg.top_p is not None:
        kw["top_p"] = cfg.top_p
    return SamplingParams(**kw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="MATH-500")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="qwen3-8b")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0, help="if >0, cap dataset to first N examples before sharding")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0, help="this process’s shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="total number of parallel shards")
    parser.add_argument("--mode", choices=["slm-normal", "slm-guided", "llm-guided"], default="slm-normal")
    parser.add_argument("--model_large", choices=MODEL_MAP.keys(), default="qwen3-32b")
    parser.add_argument("--model_small", choices=MODEL_MAP.keys(), default="qwen3-8b")
    parser.add_argument("--extraction_model", choices=MODEL_MAP.keys(), default="qwen3-8b")
    parser.add_argument(
        "--advantage_extractor",
        default="gemini-3-pro",
        help="src/advantage_descriptions/{subject}/{dataset}/{advantage_extractor}/... (default: gemini-3-pro)",
    )
    args = parser.parse_args()

    # ─── 1) Load dataset ─────────────────────────────────────────────────────
    dataset_name, split = DATASET_MAP[args.dataset]["args"]
    config_name = DATASET_MAP[args.dataset].get("config")
    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    question_key = DATASET_MAP[args.dataset]["question_key"]
    answer_key   = DATASET_MAP[args.dataset]["answer_key"]
    subject = DATASET_MAP[args.dataset]["subject"]

    if args.dataset == "AIME2024":
        override_28 = r"""Torus $T$ is the surface produced by revolving a circle with radius $3$ around an axis in the plane of the circle that is a distance $6$ from the center of the circle (so like a donut). Let $S$ be a sphere with a radius $11$. When $T$ rests on the inside of $S$, it is internally tangent to $S$ along a circle with radius $r_i$, and when $T$ rests on the outside of $S$, it is externally tangent to $S$ along a circle with radius $r_o$. The difference $r_i-r_o$ can be written as $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$. 
[asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); draw((-1.2,0.1)..(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11)..(0.8,-0.03)..(1.2,0.1)); draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12)..(1,0.04)); draw((0,2.4)--(0,-0.15)); draw((0,-0.15)--(0,-1.75), dashed); draw((0,-1.75)--(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); draw((2.03,-0.02)--(2.9,-0.4)); [/asy]"""
    
        # Override only the example at index 28
        ds = ds.map(
            lambda example, idx: {"problem": override_28} if idx == 28 else example,
            with_indices=True
        )
    if args.dataset == "MMLU-Pro-math":
        ds = ds.filter(lambda ex: ex["category"] == "math")
        options_key = DATASET_MAP[args.dataset]["options_key"]
    
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
        ds = ds.filter(lambda ex: ex[fk] == fv)
        
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
        
    if args.dataset == "MATH-L5":
        # run 100 example first
        ds = ds.select(range(1))
    
    if args.dataset == "OlympiadBench-physics":
        unit_key = DATASET_MAP[args.dataset]["unit_key"]

    # ─── 1.5) Shard‐slice ─────────────────────────────────────────────────────
    if args.num_shards > 1:
        import math
        total = len(ds)
        per_shard = math.ceil(total / args.num_shards)
        start = args.shard_id * per_shard
        end = min(start + per_shard, total)
        ds = ds.select(range(start, end))
        print(f"[shard {args.shard_id+1}/{args.num_shards}] examples {start}…{end-1}")

    # ─── 2) Load model config and tokenizer ─────────────────────────────────────
    model_id  = MODEL_MAP[args.model]
    max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
    if args.model == "deepseek-qwen3-8b":
        cfg = GenerationConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
    else:
        cfg = GenerationConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ─── 3) Initialize vLLM ─────────────────────────────────────────────────────
    llm = LLM(
        model=model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_pos,
        dtype=torch.bfloat16
    )

    # ─── 4) Prepare sampling parameters ─────────────────────────────────────
    sampling_params = make_params(args.n_sample, max_pos - 1024, cfg)
    
    

    # ─── 5) Build prompts ─────────────────────────────────────────────────────
    # 5.1) ──── retained question list ───────────────────────────────────────
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
    ds_by_question = {ex[question_key]: ex for ex in ds}
    retained_questions = [q for q in retained_questions if q in ds_by_question]
    
    # 5.2) ──── build prompts ───────────────────────────────────────
    constraints_list = []
    constraints_root = base.parents[1] / "constraints"
    if args.mode == "slm-guided":
        constraints_file = constraints_root / subject / args.dataset / f"{args.model_large}_vs_{args.model_small}" / f"{args.model_small}_extracted_constraints.json"
        with open(constraints_file, "r") as f:
            constraints_list = json.load(f)
    elif args.mode == "llm-guided":
        constraints_file = constraints_root / subject / args.dataset / f"{args.model_large}_vs_{args.model_small}" / f"{args.model_large}_extracted_constraints.json"
        with open(constraints_file, "r") as f:
            constraints_list = json.load(f)
            
            
    prompts = []
    eval_items = []
    for q in retained_questions:
        # Find question in ds
        ex = ds_by_question[q]
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
            full_question = f"Problem: {q}\n\n. Please reason step by step, and put your answer in \\boxed{{}}."
            
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
        chat_prompt = apply_chat(prompt, tokenizer)
        prompt_tokens = len(tokenizer.encode(chat_prompt, add_special_tokens=False))
        prompts.append(chat_prompt)
        eval_items.append({
            "question": q,
            "gold": gold,
            "prompt_tokens": prompt_tokens,
        })
        print(prompt)

    # ─── 6) Generate ─────────────────────────────────────────────────────
    results = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # ─── 7) Collect runs and compute stats ─────────────────────────────────────
    runs = {rid: [] for rid in range(args.n_sample)}
    def _extract_boxed(text: str) -> str:
        """
        For CRUXEVAL: prefer [ANSWER]...[/ANSWER], else last \boxed{...}, else last non-empty line.
        """
        import re
        answers = re.findall(r"\[ANSWER\](.*?)\[/ANSWER\]", text, flags=re.S | re.I)
        if answers:
            return answers[-1].strip()
        boxes = re.findall(r"\\boxed\{(.*?)\}", text, flags=re.S)
        if boxes:
            return boxes[-1].strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text.strip()

    def _literal_eval_safe(val: str):
        import ast
        try:
            parsed = ast.literal_eval(val)
            return parsed
        except Exception:
            return val.strip()

    def _eq_cruxeval(pred_obj, gold_obj):
        """
        CRUXEVAL: allow string answers to match after strip(); otherwise fall back to ==.
        """
        if isinstance(pred_obj, str) and isinstance(gold_obj, str):
            return pred_obj.strip() == gold_obj.strip()
        return pred_obj == gold_obj

    def _normalize_gold(val):
        """
        OlympiadBench: Flatten gold into a list of string answers; handles list-wrapped strings.
        """
        if val is None:
            return None
        if isinstance(val, list):
            return val[0]
        

    for idx, gen in enumerate(results):
        item = eval_items[idx]
        gold = item["gold"]
        if args.dataset == "OlympiadBench-physics":
            gold = _normalize_gold(gold)
        for rid, out in enumerate(gen.outputs):
            correct = False
            text = out.text.strip()
            # prediction extraction
            if args.dataset == "CRUXEVAL-O" or args.dataset == "CRUXEVAL-I":
                gold_obj = _literal_eval_safe(gold)
                pred = _extract_boxed(text)
                pred_obj= _literal_eval_safe(pred)
                try:
                    correct = _eq_cruxeval(pred_obj, gold_obj)
                except Exception:
                    pass
            else:
                pred = extract_answer(text)
                try:
                    correct = verify_answer(gold, pred)
                except:
                    pass
            # reasoning length (entire response) in tokens
            reasoning_length = len(tokenizer.encode(text, add_special_tokens=False))

            runs[rid].append({
                "question":         item["question"],
                "full_response":    text,
                "prompt_tokens":    item["prompt_tokens"],
                "reasoning_length": reasoning_length,
                "prediction":       repr(pred_obj) if args.dataset == "CRUXEVAL-O" or args.dataset == "CRUXEVAL-I" else pred,
                "gold":             repr(gold_obj) if args.dataset == "CRUXEVAL-O" or args.dataset == "CRUXEVAL-I" else gold,
                "correct":          correct
            })
    
    # ─── 7.5) per-question pass rate across n_sample ─────────────────────────────────────
    num_q = len(eval_items)
    for qi in range(num_q):
        pass_count = sum(1 for rid in runs if runs[rid][qi].get("correct"))
        pass_rate = pass_count / args.n_sample * 100.0
        for rid in runs:
            runs[rid][qi]["pass_rate"] = pass_rate

    # ─── 8) Save per‐shard JSON ─────────────────────────────────────────────────────
    if args.mode == "slm-normal":
        output_dir = (
            INTERVENTION_DIR / f"{args.mode}_results" / subject / args.dataset / args.model
        )
    elif args.mode == "slm-guided" or args.mode == "llm-guided":
        output_dir = (
            INTERVENTION_DIR
            / f"{args.mode}_results"
            / subject
            / args.dataset
            / args.model
            / args.extraction_model
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.n_sample}_runs_shard{args.shard_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"runs":[{"run_id":rid,"records":recs} for rid,recs in runs.items()]}, f, indent=4)
    print(f"Wrote shard {args.shard_id} results to {output_path}")


if __name__ == "__main__":
    main()