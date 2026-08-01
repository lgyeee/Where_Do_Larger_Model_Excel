#!/usr/bin/env python
# gather_constraints_shard.py
#
# Merge per-shard constraint extraction outputs into a single JSON list.

import json
import argparse
import sys
from pathlib import Path

base = Path(__file__).resolve()
src_dir = base.parents[2] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils import DATASET_MAP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--extraction_model", required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--model_large", default="qwen3-32b")
    p.add_argument("--model_small", default="qwen3-8b")
    args = p.parse_args()

    subject = DATASET_MAP[args.dataset]["subject"]
    out_dir = (
        base.parents[1] / "constraints" / subject / args.dataset
        / f"{args.model_large}_vs_{args.model_small}"
    )

    merged = []
    for sid in range(args.num_shards):
        path = out_dir / f"{args.extraction_model}_extracted_constraints_shard{sid}.json"
        if not path.exists():
            print(f"[WARN] missing shard {sid}: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)
        if isinstance(shard, list):
            merged.extend(shard)

    out_path = out_dir / f"{args.extraction_model}_extracted_constraints.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(merged)} records to {out_path}")


if __name__ == "__main__":
    main()
