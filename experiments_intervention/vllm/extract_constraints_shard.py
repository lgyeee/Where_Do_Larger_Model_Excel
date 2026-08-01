import os
import json
import re
import argparse
import sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import GenerationConfig, AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

base = Path(__file__).resolve()
SRC_DIR = base.parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import DATASET_MAP, MODEL_MAP


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINKING_RE = re.compile(r".*?", re.DOTALL | re.IGNORECASE)


def _try_json_load(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def parse_model_json(text: str):
    """Try to recover a JSON object from a model response."""
    if not text or not text.strip():
        return None

    candidates = [text.strip()]
    stripped = _THINKING_RE.sub("", text).strip()
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    for candidate in candidates:
        parsed = _try_json_load(candidate)
        if parsed is not None:
            return parsed

    for candidate in candidates:
        m = _JSON_FENCE_RE.search(candidate)
        if m:
            parsed = _try_json_load(m.group(1).strip())
            if parsed is not None:
                return parsed

    for candidate in candidates:
        key_idx = candidate.rfind('"explicit"')
        if key_idx == -1:
            continue
        start = candidate.rfind("{", 0, key_idx)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == "{":
                depth += 1
            elif candidate[i] == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try_json_load(candidate[start:i + 1])
                    if parsed is not None:
                        return parsed
                    break

    return None


def is_clean_constraints(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False
    if not isinstance(parsed.get("explicit"), list):
        return False
    if not isinstance(parsed.get("implicit"), list):
        return False
    return True


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
    parser.add_argument("--extraction_model", choices=MODEL_MAP.keys(), default="qwen3-8b")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0, help="if >0, cap dataset to first N examples before sharding")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0, help="this process’s shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="total number of parallel shards")
    parser.add_argument("--model_large", choices=MODEL_MAP.keys(), default="qwen3-32b")
    parser.add_argument("--model_small", choices=MODEL_MAP.keys(), default="qwen3-8b")
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
        ds = ds.filter(lambda ex: ex["answer_type"] == "Numerical")

    # ─── 1.5) retained question list from advantage_descriptions ─────────────
    constraint_file = (
        SRC_DIR / "advantage_descriptions" / subject / args.dataset / args.advantage_extractor
        / f"{args.model_large}_vs_{args.model_small}_analysis.json"
    )
    if not constraint_file.exists():
        print(f"[SKIP] No advantage questions file found: {constraint_file}")
        return
    with open(constraint_file, "r") as f:
        loaded_json = json.load(f)
    if "questions" not in loaded_json or not isinstance(loaded_json["questions"], list):
        raise ValueError(f"Expected key 'questions' mapped to a list in {constraint_file}")
    retained_questions = [qobj["question"] for qobj in loaded_json["questions"] if "question" in qobj]

    ds_by_question = {ex[question_key]: ex for ex in ds}
    retained_questions = [q for q in retained_questions if q in ds_by_question]

    # ─── 1.6) Shard‐slice retained questions ─────────────────────────────────
    if args.num_shards > 1:
        import math
        total = len(retained_questions)
        per_shard = math.ceil(total / args.num_shards)
        start = args.shard_id * per_shard
        end = min(start + per_shard, total)
        retained_questions = retained_questions[start:end]
        print(f"[shard {args.shard_id+1}/{args.num_shards}] questions {start}…{end-1}")

    if not retained_questions:
        print(f"[SKIP] No retained questions for shard {args.shard_id}")
        return

    # ─── 2) Load model config and tokenizer ─────────────────────────────────────
    model_id = MODEL_MAP[args.extraction_model]
    max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
    if args.extraction_model == "deepseek-qwen3-8b":
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
    prompts = []
    eval_items = []
    for q in retained_questions:
        ex = ds_by_question[q]

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
        !ONLY OUTPUT JSON!:
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
        chat_prompt = apply_chat(prompt, tokenizer)
        prompt_tokens = len(tokenizer.encode(chat_prompt, add_special_tokens=False))
        prompts.append(chat_prompt)
        eval_items.append({
            "question": q,
            "prompt_tokens": prompt_tokens,
        })

    # ─── 6) Generate ─────────────────────────────────────────────────────
    results = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # ─── 7) Collect extracted constraints ─────────────────────────────────────
    records = []
    for idx, gen in enumerate(results):
        item = eval_items[idx]
        for out in gen.outputs:
            text = out.text.strip()
            completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            parsed = parse_model_json(text)
            succeeded = is_clean_constraints(parsed)
            if not succeeded:
                parsed = {"explicit": [], "implicit": []}
            records.append({
                "question": item["question"],
                "constraints": parsed,
                "content": text,
                "completion_tokens": completion_tokens,
                "prompt_tokens": item["prompt_tokens"],
                "constraints_parse_failed": not succeeded,
            })

    # ─── 8) Save per‐shard JSON ─────────────────────────────────────────────────────
    output_dir = (
        base.parents[1] / "constraints" / subject / args.dataset
        / f"{args.model_large}_vs_{args.model_small}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{args.extraction_model}_extracted_constraints_shard{args.shard_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()