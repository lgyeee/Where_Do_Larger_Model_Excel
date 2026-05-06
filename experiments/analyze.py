import os
import json
import pandas as pd
from pathlib import Path


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
models   = ['qwen3-8b', 'qwen3-32b', 'gpt-oss-20b', 'gpt-oss-120b']
EVAL_DIR = Path(__file__).resolve().parents[1] / "src" / "eval_outputs"


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
    
# 1) Evaluation accuracy from eval_outputs
eval_rows = []
for ds in datasets:
    # ===== Get subject =====
    subject = get_subject(ds)
    # =======================
    for model in models:
        path = EVAL_DIR / subject / ds / model / "10_runs.json"
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
                'model':   model,
                'dataset': ds,
                'accuracy': f"{acc.mean():.3f} ± {acc.std(ddof=1):.3f}"
            })

df_eval_raw = pd.DataFrame(eval_rows)
if df_eval_raw.empty:
    df_eval = pd.DataFrame(index=models, columns=datasets)
else:
    df_eval = df_eval_raw.pivot(index='model', columns='dataset', values='accuracy')

# Print by subject
subjects = {
    'math': ['OMNI-MATH', 'HHMT', 'JEEBENCH-MATH'],
    'physics': ['gpqa-physics', 'JEEBENCH-PHYSICS', 'OlympiadBench-physics'],
    'chemistry': ['JEEBENCH-CHEMISTRY', 'gpqa-chemistry'],
    'programming': ['CRUXEVAL-O', 'CRUXEVAL-I'],
}
for subject, cols in subjects.items():
    print(f"=== eval_accuracy ({subject}) ===")
    print(df_eval[[c for c in cols if c in df_eval.columns]], "\n")
