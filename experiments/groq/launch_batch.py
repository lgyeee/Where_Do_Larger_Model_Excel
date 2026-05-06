import os
import json
import argparse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def main():
    # =============== Config ===============
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_sample", type=int, required=True)
    args = parser.parse_args()

    # =============== Read pending input files ==============
    with open("batch_files/input_file_id.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # =============== Load existing batch jobs to skip duplicates ==============
    existing_keys = set()
    existing_path = "batch_files/batch_job_ids.jsonl"
    if os.path.exists(existing_path):
        with open(existing_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = (obj.get("dataset"), obj.get("model"), int(obj.get("n_sample", -1)), obj.get("input_file_id"))
                existing_keys.add(key)

    # =============== Collect batch job entries ==============
    records = []

    for line in lines:
        # only handle requested dataset/model/n_sample
        if line.get("dataset") != args.dataset or line.get("model") != args.model or int(line.get("n_sample", -1)) != args.n_sample:
            continue
        input_file_id = line["input_file_id"]
        key = (line["dataset"], line["model"], int(line["n_sample"]), input_file_id)
        if key in existing_keys:
            print(f"[skip] already has batch job for {key}")
            continue
        response = client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=input_file_id,
        )
        subject = line["subject"]
        model = line["model"]
        dataset = line["dataset"]
        n_sample = line["n_sample"]
        print(f"Created batch job for {subject} {model} {dataset} {n_sample} with input file id {input_file_id}")
        records.append({
            "batch_job_id": response.id,
            "subject": subject,
            "model": model,
            "dataset": dataset,
            "n_sample": n_sample,
            "input_file_id": input_file_id,
        })
    os.makedirs("batch_files", exist_ok=True)
    with open("batch_files/batch_job_ids.jsonl", "a", encoding="utf-8") as f:
        for data in records:
            f.write(json.dumps(data) + "\n")
    print("Saved to batch_files/batch_job_ids.jsonl")


if __name__ == "__main__":
    main()
