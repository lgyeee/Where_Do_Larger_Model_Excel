import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# read from batch_files/batch_job_ids.jsonl and retrieve the batch results by reading output_file_id
with open("batch_files/batch_job_ids.jsonl", "r") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        batch_job_id = record["batch_job_id"]
        response = client.batches.retrieve(batch_job_id)
        #==========================================
        # Check if the batch has error
        #==========================================
        if response.error_file_id:
            err_path = f"batch_errors/{record['subject']}/{record['dataset']}/{record['model']}/{record['n_sample']}_runs.jsonl"
            error_content = client.files.content(response.error_file_id)
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            error_content.write_to_file(err_path)
            print(f"[SAVE] Batch error to: {err_path}")
            
        # ==========================================
        # Check if the batch results already downloaded 
        #==========================================
        out_path = f"batch_results/{record['subject']}/{record['dataset']}/{record['model']}/{record['n_sample']}_runs.jsonl"
        if os.path.exists(out_path):
            print(f"[SKIP] Already exists: {out_path}")
            continue            
        #==========================================
        # Download the batch results 
        #==========================================
        if response.output_file_id:
            output_file_id = response.output_file_id
            output_content = client.files.content(output_file_id)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            output_content.write_to_file(out_path)
            print(f"[SAVE] Batch results to: {out_path}")
        else:
            print(f"[SKIP] No output yet for batch {batch_job_id} ({record['dataset']})")
            continue
    