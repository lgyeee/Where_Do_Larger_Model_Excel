# Usage: python analyze.py
import json
import pandas as pd
from pathlib import Path

MODES = ['slm', 'slm-normal', 'slm-guided', 'llm-guided']
BASE_DIR = Path(__file__).resolve().parent 
MODEL_FAMILIES = {
    'gpt-oss': ['gpt-oss-20b'],
    'qwen3': ['qwen3-8b'],
}
EXTRACTION_MODELS = {
    ('gpt-oss-20b', 'slm-guided'): 'gpt-oss-20b',
    ('gpt-oss-20b', 'llm-guided'): 'gpt-oss-120b',
    ('qwen3-8b', 'slm-guided'): 'qwen3-8b',
    ('qwen3-8b', 'llm-guided'): 'qwen3-32b',
}

# Utility function to load runs
def load_runs(path: str) -> pd.DataFrame:
    """
    Load a 10_runs.json file and return a DataFrame with columns:
      - run_id
      - correct (bool)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for run in data.get('runs', []):
        rid = run.get('run_id')
        for rec in run.get('records', []):
            rows.append({
                'run_id': rid,
                'correct': rec.get('correct', False),
            })
    return pd.DataFrame(rows)

# Specify datasets and models
datasets = ['OMNI-MATH', 'HHMT', 'JEEBENCH-MATH', 'gpqa-physics', 'JEEBENCH-PHYSICS', 'OlympiadBench-physics', 'JEEBENCH-CHEMISTRY', 'gpqa-chemistry', 'CRUXEVAL-O','CRUXEVAL-I']
models = [model for family_models in MODEL_FAMILIES.values() for model in family_models]
EVAL_DIR_MAP = {
    'slm': "slm_results",
    'slm-normal': "slm-normal_results",
    'slm-guided': "slm-guided_results",
    'llm-guided': "llm-guided_results",
}


def get_subject(ds):
    if ds in ['OMNI-MATH', 'HHMT', 'JEEBENCH-MATH']:
        return 'math'
    elif ds in ['gpqa-chemistry', 'JEEBENCH-CHEMISTRY']:
        return 'chemistry'
    elif ds in ['gpqa-physics', 'JEEBENCH-PHYSICS', 'OlympiadBench-physics']:
        return 'physics'
    elif ds in ['CRUXEVAL-O','CRUXEVAL-I']:
        return 'programming'
    else:
        return 'unknown'

def get_result_path(mode, subject, dataset, model):
    base = BASE_DIR / EVAL_DIR_MAP[mode] / subject / dataset / model
    if mode == 'slm-normal':
        return base / "10_runs.json"

    extractor = EXTRACTION_MODELS.get((model, mode))
    if extractor:
        extractor_path = base / extractor / "10_runs.json"
        if extractor_path.is_file():
            return extractor_path

    matches = sorted(base.glob("*/10_runs.json"))
    return matches[0] if matches else base / "10_runs.json"

# 1) Evaluation accuracy from results
eval_rows = []
for ds in datasets:
    subject = get_subject(ds)
    for model in models:
        for mode in MODES:
            path = get_result_path(mode, subject, ds, model)
            if path.is_file():
                try:
                    df = load_runs(path)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[WARN] skip invalid JSON: {path} ({e})")
                    continue
                if df.empty:
                    continue
                acc = df.groupby('run_id')['correct'].mean()
                eval_rows.append({
                    'model': model,
                    'mode': mode,
                    'dataset': ds,
                    'mean_accuracy': acc.mean(),
                    'accuracy': f"{acc.mean():.3f} ± {acc.std(ddof=1):.3f}",
                })

df_eval_raw = pd.DataFrame(eval_rows)

subjects = {
    'math': ['OMNI-MATH', 'HHMT', 'JEEBENCH-MATH'],
    'physics': ['gpqa-physics', 'JEEBENCH-PHYSICS', 'OlympiadBench-physics'],
    'chemistry': ['JEEBENCH-CHEMISTRY', 'gpqa-chemistry'],
    'programming': ['CRUXEVAL-O', 'CRUXEVAL-I'],
}

def print_accuracy_table(table_id, family, family_models):
    print(f"Table {table_id}: {family}")
    print("==== accuracy ======")
    for subject, cols in subjects.items():
        print(subject)
        subject_cols = [c for c in cols if c in datasets]
        if df_eval_raw.empty:
            df_subject = pd.DataFrame(index=MODES, columns=subject_cols)
        else:
            df_subject = (
                df_eval_raw[df_eval_raw['model'].isin(family_models)]
                .pivot_table(
                    index='mode',
                    columns='dataset',
                    values='accuracy',
                    aggfunc='first',
                )
                .reindex(index=MODES, columns=subject_cols)
            )
        print(df_subject.fillna('NA').to_string())
        print()


def print_overall_accuracy_table(table_id, family, family_models):
    print(f"Table {table_id}: {family}")
    print("==== macro-average accuracy (mean over datasets) ======")
    if df_eval_raw.empty:
        df_overall = pd.DataFrame(index=MODES, columns=['macro_accuracy'])
    else:
        macro = (
            df_eval_raw[df_eval_raw['model'].isin(family_models)]
            .groupby('mode')['mean_accuracy']
            .mean()
            .reindex(MODES)
        )
        df_overall = pd.DataFrame({
            'macro_accuracy': macro.apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else 'NA',
            )
        })
    print(df_overall.fillna('NA').to_string())
    print()


for table_id, (family, family_models) in enumerate(MODEL_FAMILIES.items(), start=1):
    print_accuracy_table(table_id, family, family_models)
    print_overall_accuracy_table(f"{table_id}-overall", family, family_models)
