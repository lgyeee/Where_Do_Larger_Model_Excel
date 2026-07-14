import os
import json
import re
import math
import numpy as np
# from math_grader import strip_string
from math_verify import parse, verify, LatexExtractionConfig

DATASET_MAP = {
    "MATH-500": {
        "subject": "math",
        "args": ("HuggingFaceH4/MATH-500", "test"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "AIME2024": {
        "subject": "math",
        "args": ("HuggingFaceH4/aime_2024", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "gpqa": {
        "subject": "mixed",
        "args": ("hendrydong/gpqa_diamond_mc", "test"),
        "question_key": "problem",
        "answer_key": "solution"
    },
    "gpqa-chemistry": {
        "subject": "chemistry",
        "args": ("hendrydong/gpqa_diamond_mc", "test"),
        "question_key": "problem",
        "answer_key": "solution",
        "filter_key": "domain",
        "filter_value": "Chemistry",
    },
    "gpqa-physics": {
        "subject": "physics",
        "args": ("hendrydong/gpqa_diamond_mc", "test"),
        "question_key": "problem",
        "answer_key": "solution",
        "filter_key": "domain",
        "filter_value": "Physics",
    },
    "gsm8k": {
        "subject": "math",
        "args": ("skrishna/gsm8k_only_answer", "test"),
        "question_key": "text",
        "answer_key": "label"
    },
    "openr1-math": {
        "subject": "math",
        "args": ("open-r1/OpenR1-Math-220k", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "AIME2025": {
        "subject": "math",
        "args": ("yentinglin/aime_2025", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "MMLU-Pro-math": {
        "subject": "math",
        "filter_key": "category",
        "args": ("TIGER-Lab/MMLU-Pro", "test"),
        "options_key": "options",
        "question_key": "question",
        "answer_key": "answer"
    },
    "MMLU-Pro-physics": {
        "subject": "physics",
        "filter_key": "category",
        "filter_value": "physics",
        "args": ("TIGER-Lab/MMLU-Pro", "test"),
        "options_key": "options",
        "question_key": "question",
        "answer_key": "answer"
    },
    "MMLU-Pro-chemistry": {
        "subject": "chemistry",
        "filter_key": "category",
        "filter_value": "chemistry",
        "args": ("TIGER-Lab/MMLU-Pro", "test"),
        "options_key": "options",
        "question_key": "question",
        "answer_key": "answer"
    },
    "CRUXEVAL-O":{
        "subject": "programming",
        "args": ("cruxeval-org/cruxeval", "test"),
        "input_key": "input",
        "question_key": "code",
        "answer_key": "output"
    },
    "MATH-L5": {
        "subject": "math",
        "args": ("AI-MO/aimo-validation-math-level-5", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "HHMT":{
        "subject": "math",
        "args": ("MathArena/hmmt_nov_2025", "train"),
        "question_key": "problem",
        "answer_key": "answer"
    },
    "OMNI-MATH":{
        "subject": "math",
        "args": ("KbsdJames/Omni-MATH", "test"),
        "question_key": "problem",
        "answer_key": "answer",
        "difficulty_key": "difficulty"
    },
    "CRUXEVAL-I":{
        "subject": "programming",
        "args": ("cruxeval-org/cruxeval", "test"),
        "output_key": "output",
        "question_key": "code",
        "answer_key": "input"
    },
    "JEEBENCH-PHYSICS":{
        "subject": "physics",
        "filter_key": "subject",
        "filter_value": "phy",
        "args": ("daman1209arora/jeebench", "test"),
        "question_key": "question",
        "answer_key": "gold",
        "type_key": "type"
    },
    "JEEBENCH-CHEMISTRY":{
        "subject": "chemistry",
        "filter_key": "subject",
        "filter_value": "chem",
        "args": ("daman1209arora/jeebench", "test"),
        "question_key": "question",
        "answer_key": "gold",
        "type_key": "type"
    },
    "JEEBENCH-MATH":{
        "subject": "math",
        "filter_key": "subject",
        "filter_value": "math",
        "args": ("daman1209arora/jeebench", "test"),
        "question_key": "question",
        "answer_key": "gold",
        "type_key": "type"
    },
    "OlympiadBench-physics":{
        "subject": "physics",
        "args": ("Hothan/OlympiadBench", "train"),
        "config": "OE_TO_physics_en_COMP",
        "question_key": "question",
        "answer_key": "final_answer",
        "unit_key": "unit"
    }
}

MODEL_MAP   = {
    "deepseek-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-llama3-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "qwq-32b": "Qwen/QwQ-32B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "deepseek-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "phi4-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
    "nemotron-7b": "nvidia/OpenMath-Nemotron-7B",
    "olmo3-7b-think": "allenai/Olmo-3-7B-Think",
    "olmo3-32b-think": "allenai/Olmo-3-32B-Think",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemma4-12b": "google/gemma-4-12B-it"
}

def verify_answer(pred: str, ref: str) -> bool:

    # ── patterns & threshold ─────────────────────────────────────────────────
    BASE_N_RE    = re.compile(r"^\(?([0-9A-Za-z]+)\)?_\{(\d+)\}$")
    EXP_RE       = re.compile(r"\^\{(\d+)\}")
    MAX_SAFE_EXP = 10_000

    def _strip_tex_delims(s: str) -> str:
        """
        Remove outer $...$, $$...$$, \\(...\\), \\[...\\] if present.
        """
        if s is None:
            return ""
        t = s.strip()
        while t.startswith("$") and t.endswith("$") and len(t) >= 2:
            t = t[1:-1].strip()
        if t.startswith("\\(") and t.endswith("\\)"):
            t = t[2:-2].strip()
        if t.startswith("\\[") and t.endswith("\\]"):
            t = t[2:-2].strip()
        return t

    # ── normalize inputs ─────────────────────────────────────────────────────
    if pred is None or ref is None:
        return False
    p = _strip_tex_delims(pred)
    r = _strip_tex_delims(ref)

    # ── 1) base-N literal in prediction ─────────────────────────────────────
    m = BASE_N_RE.match(p)
    if m:
        return m.group(1) == r

    # ── 2) base-N literal in reference ──────────────────────────────────────
    m = BASE_N_RE.match(r)
    if m:
        return m.group(1) == p

    # ── 3) huge-exponent guard ───────────────────────────────────────────────
    exps = [int(e) for e in EXP_RE.findall(p)]
    if exps and max(exps) > MAX_SAFE_EXP:
        return p.replace(" ", "") == r.replace(" ", "")

    # ── 4) fallback to math_verify ──────────────────────────────────────────
    wrap = lambda s: f"\\({s}\\)"
    cfg  = LatexExtractionConfig()
    try:
        g_node = parse(wrap(r), extraction_config=[cfg])
        p_node = parse(wrap(p), extraction_config=[cfg])
        return verify(g_node, p_node, float_rounding=2)
    except Exception:
        return False

def extract_answer(text):
    if text is None:
        return None
    # Step 1: Remove everything that is not a number, letter, ".", or "-"
    # text = re.sub(r'[^0-9a-zA-Z{}\\.\-]', '', text)
    # Try extracting from 'boxed' first
    boxed_matches = extract_boxed(text)
    if boxed_matches:
        extracted_answer = boxed_matches[-1][1:-1]
        return extracted_answer

    # Fallback: extract any numbers
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if not numbers:
        return None

    try:
        extracted_number = float(numbers[-1])
        # Guard against infinity
        if math.isinf(extracted_number):
            return None
        
        return numbers[-1]
    except (ValueError, OverflowError):
        return None

def extract_boxed(text):
    pattern = re.compile(r'boxed\{')
    matches = []
    stack = []
    
    i = 0
    while i < len(text):
        match = pattern.search(text, i)
        if not match:
            break
        
        start = match.end() - 1  # Position at the first `{`
        stack.append(start)
        i = start + 1
        count = 1  # To track `{}` pairs
        
        while i < len(text) and stack:
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:  # Found a matching closing `}`
                    start = stack.pop()
                    matches.append(text[start:i+1])
                    break
            i += 1
    
    return matches

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