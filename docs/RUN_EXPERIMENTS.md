# LAR-RAG Experiments - Quick Run Guide

Covers all three datasets (BGL, HDFS, Thunderbird). All configs converge on the same
underlying engine, `src.evaluation.run_experiments_v2`, so results are directly comparable
across datasets — only the YAML config and, for HDFS/Thunderbird, the pipeline wrapper differ.

## Prerequisites
- Python environment activated: `.venv\Scripts\Activate.ps1`
- Data processed for the dataset(s) you're running (parsed logs, templates, embeddings, FAISS/BM25/occurrence store)
- **Thunderbird only:** the `D:\Paper\LAR-RAG\` external drive must be connected — `parsed.jsonl`, `template_map.jsonl`, and `occurrences.duckdb` live there (too large for the repo's `C:` tree). See `config/thunderbird_full_run.yaml` / `config/thunderbird_ablations.yaml` for the exact paths.

## BGL

### 1. Full Run (All Baselines + LAR-RAG)
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config config\bgl_full_run.yaml
```
**Output:** `results/bgl_full_run/`

### 2. Ablation Study (LAR-RAG Variants)
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config config\bgl_ablations.yaml
```
**Output:** `results/bgl_ablations/`

Evaluates:
- `lar_rag_full` (α=1.0, β=0.25, γ=0.25)
- `lar_rag_no_time` (α=1.0, β=0.0, γ=0.25)
- `lar_rag_no_topo` (α=1.0, β=0.25, γ=0.0)

### 3. Parameter Sweep (16 Configurations)
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config config\bgl_sweep.yaml
```
**Output:** `results/bgl_sweep/`

Grid: β ∈ {0.0, 0.25, 0.5, 1.0} × γ ∈ {0.0, 0.25, 0.5, 1.0}

## HDFS

### 1. Full Run (All Baselines + LAR-RAG)
```powershell
.venv\Scripts\python.exe scripts\run_hdfs_pipeline.py --config config\hdfs_full_run.yaml
```
**Output:** `results/hdfs_full_run/`

### 2. Ablation Study (LAR-RAG Variants)
```powershell
.venv\Scripts\python.exe scripts\run_hdfs_pipeline.py --config config\hdfs_ablations.yaml
```
**Output:** `results/hdfs_ablations/`

`run_hdfs_pipeline.py` re-checks data-prep steps (parse/templates/embed/index/ground truth) before
evaluating and skips any that already exist, so re-running is cheap once the pipeline has been built once.

## Thunderbird

### 1. Full Run (All Baselines + LAR-RAG)
```powershell
.venv\Scripts\python.exe scripts\run_thunderbird_pipeline.py
```
**Output:** `results/thunderbird_full_run/`. Config is hardcoded to `config/thunderbird_full_run.yaml`
inside the script (unlike `run_hdfs_pipeline.py`, it doesn't expose a `--config` flag yet).

### 2. Ablation Study (LAR-RAG Variants)
Data prep for Thunderbird is already built, so ablations are run by calling the evaluation engine
directly instead of going through the pipeline wrapper:
```powershell
.venv\Scripts\python.exe -m src.evaluation.run_experiments_v2 --config config\thunderbird_ablations.yaml
```
**Output:** `results/thunderbird_ablations/`

Evaluates the same variants as BGL/HDFS ablations. Expect noticeably higher latency than
BGL/HDFS — Thunderbird's node-cooccurrence topology graph is denser, so `lar_rag_full`/`_no_time`
P95 latency runs into the tens of seconds per query (see `results/thunderbird_full_run/results.json`
for the full-run baseline: ~49s P95 for `lar_rag_full`).

## Results
Each run generates, under its `output_dir`:
- `results.json` - Aggregate metrics
- `results.csv` - Summary table
- `per_query_*.jsonl` - Per-query results
- `run_meta.json` - Configuration metadata

`scripts/compute_tighter_cutoffs.py` derives nDCG@3/5 and Recall@3/5 from the `per_query_*.jsonl`
files for every dataset's main-run and ablation directory without re-running anything.

## Force Rebuild
To regenerate ground truth or indices:
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config <config> --force-rebuild        # BGL
.venv\Scripts\python.exe scripts\run_hdfs_pipeline.py --config <config> --force-rebuild        # HDFS
.venv\Scripts\python.exe scripts\run_thunderbird_pipeline.py --force-rebuild                    # Thunderbird
```

## Typical Runtime
- BGL full run: ~20 minutes
- BGL ablations: ~25 minutes
- Sweep (baselines only): ~7 minutes
