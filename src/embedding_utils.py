# embedding_utils.py
# Embed a text using OpenAI's embedding model
import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from utils import DATASET_MAP

def embed_text(text, embed_model="text-embedding-3-large") -> np.array:
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        resp = client.embeddings.create(model=embed_model, input=text)
        return resp.data[0].embedding
    except Exception as e:
        print(f"[WARN] embedding failed: {e}")
        return None
    
def load_embedding_file(datasets, model_large, model_small) -> dict:
    embedding_data_by_ds = {}
    for ds in datasets:
        path = f"embedding_results/{DATASET_MAP[ds]['subject']}/{ds}/{model_large}_vs_{model_small}_embeddings.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embedding_data_by_ds[ds] = data
    return embedding_data_by_ds

