# Where Do Larger Models Excel?

This is the official repo for the paper: [Where Do Larger Model Excel?]()

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Inference and Evaluation](#inference-and-evaluation)
- [Analyze Larger Model Advantages](#analyze-larger-model-advantages)
- [Citation](#citation)

## Introduction

In this work, we propose a framework to compare reasoning traces from larger and smaller models within the same model family. Our goal is to identify and organize the advantages that larger models exhibit naturally via a semantic clustering pipeline on questions where they consistently outperform smaller models. <img src="figs/where_larger_model_excel_framework.png" alt="framework" width="800" />

## Getting Started

For reproducibility, we recommend Python `3.10`.

Install the required dependencies:

```bash
pip install -r requirements.txt
```






## Inference and Evaluation

We evaluate two model families:

- `qwen3-8b` vs. `qwen3-32b`
- `gpt-oss-20b` vs. `gpt-oss-120b`

### Qwen3 (local vLLM)

To run evaluations on all datasets and models:

```bash
cd experiments/vllm
bash evaluate_shards.sh
```

This calls `evaluate_reasoning_models_shard.py` and `gather_results.py`.
If you do not have 4 GPUs, override GPU count when running:

```bash
TOTAL_GPUS=4 bash evaluate_shards.sh
```


### GPT-OSS (Groq API batch)

```bash
cd experiments/groq
bash submit_batch.sh
bash parse_batch.sh
```

For GPT-OSS models, we use API-based inference.
Please set `GROQ_API_KEY` in your `.env` file before running.

`submit_batch.sh` creates/uploads/launches batch jobs.  
`parse_batch.sh` retrieves finished jobs and parsed outputs. 

> **Output location:**  
> Infernece results (JSON) for **both Qwen3 and GPT-OSS models** are written to:  
> ```
> src/eval_outputs/...
> ```

> **Note:** Re-running the same configuration will overwrite the corresponding output JSON file.


#### Datasets:

We evaluate models on benchmarks from four domains:

- Math: `HHMT`, `JEEBENCH`, `OMNI-MATH` 
- Physics: `JEEBENCH`, `OlympiadBench`, `GPQA` 
- Chemistry: `JEEBENCH`, `GPQA`
- Programming: `CRUXEVAL-I`, `CRUXEVAL-O`

After evaluation, run:

```bash
python3 analyze.py
```
This script will create a summary of all benchmark accuracy.



## Analyze Larger model advantages

Our framework contains three stages: 
- stage1: Constructing the Analysis Question Set
- stage2: Advantage Extraction
- stage3: Semantic Clustering

All scripts and Python entrypoints live under `src/` and rely on paths relative to that directory. Run them **after**:
```bash
cd src
```

(Keep `.env`, `eval_outputs/`, `advantage_descriptions/`, `clustering_results/`, `heatmaps_results/`, etc. under `src/` as the code expects, unless you symlink or change the code.)

### Stage 1 and 2: Constructing the Analysis Question Set & Advantage Extraction

We first identify questions where the larger model consistently outperforms the smaller model
and use `gemini-3-pro` as an advantage extractor.   
Please configure the `OPENROUTER_API_KEY` in your `.env` file (for steps under `src/`, placing `.env` in `src/` is simplest).

```bash
bash construction_and_extraction.sh
```

Optional env vars: `GAP_VALUE`, `SUBJECTS`, `MODEL_PAIRS`, `ADVANTAGE_EXTRACTOR`.

The `GAP_VALUE` variable controls the minimum pass-rate gap ratio used to construct the analysis question set (default is `0.6` = `60%`).
If you cannot run evaluation locally, set `FROM_HF=1` when running `construction_and_extraction.sh` to download raw CoT traces from Hugging Face into `eval_outputs/` (repo: `lgyeee/where-larger-models-excel-reasoning-traces`). By default the script uses local `eval_outputs/` only (`FROM_HF=0`).

### Stage 3: Semantic Clustering

After extracting advantages, we use a structured clustering pipeline to organize these findings semantically.

#### 3.1 Cluster Candidate Metrics Exploration

To run the full candidate clustering pipeline, just execute:

```bash
bash sweep_dim_k.sh
```

This script mainly uses `adv_cluster.py` and will:  
1. Generate text embeddings (using OpenAI `text-embedding-3-large` — set `OPENAI_API_KEY` in your `.env`)
2. Remove duplicates
3. Run clustering for multiple combinations of PCA dimensions (`pca_dim`) and cluster numbers (`k`)

**Main parameters and configuration:**
- **Model pairs:** By default, both `qwen3-32b:qwen3-8b` and `gpt-oss-120b:gpt-oss-20b` are evaluated.  
  - Use `MODEL_PAIRS_OVERRIDE="large:small ..."` to specify pairs, or 
  - Use `MODEL_LARGE_OVERRIDE` / `MODEL_SMALL_OVERRIDE` for a single pair.
- **Datasets:** By default, all relevant datasets are included.  
  - For Qwen3, HHMT is omitted because, in our runs, there were no gap-filtered samples for that split (so nothing to cluster). For GPT-OSS, HHMT is included as usual.
- **Embeddings caching:** The first PCA dimension for each pair computes and caches embeddings; subsequent dimensions re-use cached results.  
  - To force using cached embeddings for all, run:
    ```bash
    USE_CACHED_EMBEDDINGS=1 bash sweep_dim_k.sh
    ```

**Results:**  
After completion, summary tables by model pair report mean DBI and mean silhouette for each `pca_dim`. Per-dimension k-sweep metrics and aggregation artifacts are saved under `clustering_results/`; `adv_cluster.py` also writes `*_dim_selection.csv` (one per model pair) when the k-sweep path runs.

*Example summary (Qwen3 pair):*
```bash
pca_dim  | mean_dbi     | mean_silhouette   | n_k 
---------------------------------------------------------
256      | 3.747092     | 0.055199          | 14  
128      | 3.460551     | 0.063713          | 14  
64       | 2.978327     | 0.080299          | 14  
16       | 1.998086     | 0.146378          | 14  
8        | 1.563650     | 0.207454          | 14  
4        | 1.156095     | 0.294988          | 14  
```



#### 3.2 Running Fixed-k Clustering

To select the most suitable clustering settings, we look for PCA dimensions (`pca_dim`) with relatively low mean DBI values, as this indicates better clustering performance. 
In practice, we found that using `pca_dim = 4` and `pca_dim = 8` yields good results. For each chosen dimension, we then try several values of `k` (number of clusters), with the exact combinations depending on the model pair being analyzed:

- **Qwen3** (`qwen3-32b` vs `qwen3-8b`):  
  - For `pca_dim = 4`: try `k = 3, 4, 5, 6`
  - For `pca_dim = 8`: try `k = 6, 7, 8, 9`
- **GPT-OSS** (`gpt-oss-120b` vs `gpt-oss-20b`):  
  - For `pca_dim = 4`: try `k = 2, 3, 4, 5, 6`
  - For `pca_dim = 8`: try `k = 2, 3, 12, 13, 14`

Adjust the (dim, k) pairs in `cluster_candidates.sh` if needed (defaults given above), then run:

```bash
bash cluster_candidates.sh
```

#### 3.3 Assign Cluster Tags and Definitions

To assign tags and definitions to each cluster, run:

```bash
bash create_cluster_tag.sh
```

- **Requires:** `openai/gpt-5.2` (default) and your API key as `OPENROUTER_API_KEY` in `.env`.
- Uses `summarizing.py` to generate tags and definitions.
- Output is now saved in `clustering_tags/<model_family>/<pcadim>/<model_large>_vs_<model_small>_k<cluster_k>.json`.

#### 3.4 Review & Select the Final Cluster Solution

Evaluate and score each cluster candidate using a reviewer model by running:

```bash
bash run_select_cluster_candidates.sh
```

This calls `select_cluster_candidates.py`, which uses a reviewer model (default: `openai/gpt-5.2`, configurable via `REVIEW_MODEL`) to assess and rank each candidate cluster-tag taxonomy generated by `summarizing.py`. Outputs are saved to `select_results/*_select_k_candidate_review.json`.

#### 3.5 Draw Final Cluster HeatMap

```bash
bash draw_heatmaps.sh
```

By default this runs heatmaps for both model pairs in `draw_heatmaps.sh`. Each pair writes CSV tables and three PNGs (counts, row-normalized proportions, column-normalized proportions) under `heatmaps_results/`; the paper figures use selected views from these outputs.


## Citation

If you find this repository useful, please cite our work:

```bibtex
@inproceedings{your-citation-key,
  title = {Where Do Larger Reasoning Models Excel?},
  booktitle = {Proceedings of ...},
  year = {2026}
}
```

