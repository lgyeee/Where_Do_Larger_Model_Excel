"""Prompt construction helpers mirroring OpenRouter/evaluate.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

base = Path(__file__).resolve()
INTERVENTION_DIR = base.parents[1]
SRC_DIR = base.parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import DATASET_MAP  # noqa: E402


def load_filtered_dataset(dataset_name: str):
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
        ds = ds.filter(lambda ex: str(ex.get(fk, "")).lower() == str(fv).lower())

    if dataset_name in ("JEEBENCH-PHYSICS", "JEEBENCH-CHEMISTRY", "JEEBENCH-MATH"):
        fk = DATASET_MAP[dataset_name]["filter_key"]
        fv = DATASET_MAP[dataset_name]["filter_value"]
        ds = ds.filter(lambda ex: ex[fk] == fv)

    if dataset_name == "OlympiadBench-physics":
        ds = ds.filter(lambda ex: ex["answer_type"] == "Numerical")

    return ds


def load_constraints(mode: str, subject: str, dataset: str, model_large: str, model_small: str) -> list[dict]:
    if mode == "slm-guided":
        constraints_file = (
            INTERVENTION_DIR
            / "constraints"
            / subject
            / dataset
            / f"{model_large}_vs_{model_small}"
            / f"{model_small}_extracted_constraints.json"
        )
    elif mode == "llm-guided":
        constraints_file = (
            INTERVENTION_DIR
            / "constraints"
            / subject
            / dataset
            / f"{model_large}_vs_{model_small}"
            / f"{model_large}_extracted_constraints.json"
        )
    else:
        return []

    with open(constraints_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_item(
    dataset_name: str,
    mode: str,
    question: str,
    model_large: str,
    model_small: str,
) -> dict:
    ds = load_filtered_dataset(dataset_name)
    question_key = DATASET_MAP[dataset_name]["question_key"]
    answer_key = DATASET_MAP[dataset_name]["answer_key"]
    subject = DATASET_MAP[dataset_name]["subject"]

    matches = ds.filter(lambda ex: ex[question_key] == question).to_list()
    if not matches:
        raise ValueError(f"Question not found in filtered dataset {dataset_name}")
    ex = matches[0]
    gold = ex[answer_key]

    full_question = question
    if dataset_name == "CRUXEVAL-O":
        input_key = DATASET_MAP[dataset_name]["input_key"]
        code_input = ex[input_key]
        full_question = f"""
            What should the output of this code be so that the assertion is correct? Reason step by step before
            arriving at an answer. Finally, surround the answer, with no additional words, with [ANSWER]
            and [/ANSWER] tags.
            {question}
            assert f({code_input}) == ?"""

    elif dataset_name == "CRUXEVAL-I":
        output_key = DATASET_MAP[dataset_name]["output_key"]
        code_output = ex[output_key]
        full_question = f"""
            You will be given a function f and an output in the form f(??) == output. Your task is to find any
            input such that executing f on the input leads to the given output. There may be multiple
            answers, but only output one. First, think step by step. Then, surround ONLY the INPUT VALUE
            with [ANSWER] and [/ANSWER] tags (do NOT include a function call).
            {question}
            assert f(??) == {code_output}
            """

    elif dataset_name in ("JEEBENCH-PHYSICS", "JEEBENCH-CHEMISTRY", "JEEBENCH-MATH"):
        type_key = DATASET_MAP[dataset_name]["type_key"]
        if ex[type_key] == "MCQ(multiple)":
            full_question = f"{question}\n\nPlease reason step by step, and put your answer choices in ONE \\boxed{{}}. For example, if the answer is X, Y, and Z, output \\boxed{{XYZ}}."
        elif ex[type_key] == "MCQ":
            full_question = f"{question}\n\nPlease reason step by step, and put your answer choice in \\boxed{{}}."
        elif ex[type_key] in ("Integer", "Numeric"):
            full_question = f"{question}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."

    elif dataset_name == "OlympiadBench-physics":
        unit_key = DATASET_MAP[dataset_name]["unit_key"]
        unit = ex.get(unit_key, "")
        if unit:
            full_question = (
                f"{question}\n\n"
                f"The final answer must be expressed in units of **{unit}**.\n"
                "Please reason step by step, and put **only the answer** but not units in \\boxed{{}}"
            )
        else:
            full_question = f"{question}\n\nPlease reason step by step, and put your answer in \\boxed{{}}."

    else:
        full_question = f"Problem: {question}\n\n"

    if mode == "slm-normal":
        prompt = f"""
            Please list all the explicit and derived implicit constraints for the problem. Reasoning using the constraints to check compliance at each step to prune unnecessary search space.
            Problem: {full_question}
            """
    elif mode in ("slm-guided", "llm-guided"):
        constraints_list = load_constraints(mode, subject, dataset_name, model_large, model_small)
        constraints_record = next((c for c in constraints_list if c["question"] == question), None)
        if constraints_record is None:
            raise ValueError(f"No constraints found for question in {dataset_name} mode={mode}")
        constraints = constraints_record["constraints"]
        prompt = f"""
            {full_question}
            Reason using explicit/implicit constraints, checking compliance at each step to prune unnecessary search space.

            Explicit constraints:
            {constraints.get("explicit", [])}

            Implicit constraints:
            {constraints.get("implicit", [])}
            """
    else:
        raise ValueError(f"Unsupported mode for prompt building: {mode}")

    return {"prompt": prompt, "question": question, "gold": gold}
