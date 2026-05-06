import os
import json
import argparse
import sys
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

base = Path(__file__).resolve()
for p in (base.parents[2] / "src", base.parents[3]):
    if p.exists():
        sys.path.insert(0, str(p))
        break

from utils import DATASET_MAP

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def already_uploaded(dataset: str, model: str, n_sample: int, path: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if (
                obj.get("dataset") == dataset
                and obj.get("model") == model
                and int(obj.get("n_sample", -1)) == n_sample
            ):
                return True
    return False

def main():
    # =============== Config ===============
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_sample", type=int, required=True)
    args = parser.parse_args()

    input_ids_file = "batch_files/input_file_id.jsonl"
    if already_uploaded(args.dataset, args.model, args.n_sample, input_ids_file):
        print(f"[skip] already uploaded: dataset={args.dataset} model={args.model} n_sample={args.n_sample}")
        return

    file_path = f"batch_files/{DATASET_MAP[args.dataset]['subject']}/{args.dataset}/{args.model}/{args.n_sample}_runs.jsonl"
    response = client.files.create(file=open(file_path, "rb"), purpose="batch")
    # =============== Save the input_file_id from response ==============
    input_file_id = response.id
    record = {
        "subject": DATASET_MAP[args.dataset]["subject"],
        "model": args.model,
        "dataset": args.dataset,
        "n_sample": args.n_sample,
        "input_file_id": input_file_id,
    }
    os.makedirs("batch_files", exist_ok=True)
    with open(input_ids_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Input file id: {input_file_id}")
    print(f"Appended to batch_files/input_file_id.jsonl")
if __name__ == "__main__":
    main()