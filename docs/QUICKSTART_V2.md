# LAR-RAG v2 Quick Start Guide

Complete guide for running the canonical v2 evaluation pipeline.

---

## Prerequisites

### 1. Python Environment
```powershell
python --version  # 3.11+
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Required Data Files
- `data/processed/bgl/parsed.jsonl` (parsed logs)
- `data/processed/bgl/template_map.jsonl` (template occurrences)
- `data/processed/bgl/templates.jsonl` (template list)
- `data/processed/bgl/embeddings/manifest.json` (embeddings metadata)
- `data/processed/bgl/embeddings/shard_*.parquet` (embedding shards)

---

## Quick Start (Single Command)

Run the complete pipeline:

```powershell
python scripts/paper_run_local.py --config config/bgl_canonical_v2.yaml
```

This will:
1. Build occurrence store (if not exists)
2. Build FAISS index (if not exists)
3. Build BM25 index (if not exists)
4. Generate ground truth (incidents + qrels)
5. Run all 6 retrieval methods
6. Save results to `results/bgl_rev5_local/`

**Time**: ~13 minutes for 100 incidents

---

## Step-by-Step (Manual Pipeline)

### Step 1: Build Occurrence Store

```powershell
python scripts/build_occurrence_store.py `
  --template-map data/processed/bgl/template_map.jsonl `
  --output data/processed/bgl/occurrence_store.duckdb
```

**Output**: DuckDB database with occurrence queries  
**Time**: ~30 seconds

### Step 2: Build FAISS Index

```powershell
python scripts/build_faiss_index.py `
  --embeddings-dir data/processed/bgl/embeddings `
  --out data/processed/bgl/indices/faiss `
  --normalize
```

**Output**: FAISS index + metadata  
**Time**: ~45 seconds

### Step 3: Build BM25 Index

```powershell
python scripts/build_bm25_index.py `
  --templates data/processed/bgl/templates.jsonl `
  --output data/processed/bgl/indices/bm25
```

**Output**: Pickled BM25 index + template IDs  
**Time**: ~15 seconds

### Step 4: Generate Ground Truth

```powershell
python scripts/build_ground_truth_streaming.py `
  --parsed-logs data/processed/bgl/parsed.jsonl `
  --template-map data/processed/bgl/template_map.jsonl `
  --output data/processed/bgl/ground_truth.jsonl `
  --seed 42 `
  --max-incidents 100 `
  --min-relevant 3
```

**Output**: 
- `ground_truth_incidents.jsonl` (queries)
- `ground_truth_qrels.jsonl` (relevance judgments)
- `ground_truth_combined.jsonl` (backward compat)

**Time**: ~2 minutes

### Step 5: Run Experiments

```powershell
python -m src.evaluation.run_experiments_v2 --config config/bgl_canonical_v2.yaml
```

**Output**:
- `results/bgl_rev5_local/results.json` (detailed metrics)
- `results/bgl_rev5_local/results.csv` (summary table)
- `results/bgl_rev5_local/run_meta.json` (config hash + git commit)

**Time**: ~10 minutes for 6 methods

---

## Configuration

### Default Config: `config/bgl_canonical_v2.yaml`

Key sections:

```yaml
run:
  name: "bgl_rev5_local"
  seed: 42
  output_dir: "results/bgl_rev5_local"
  top_k: 10

lar_rag:
  alpha: 1.0    # Similarity weight
  beta: 0.3     # Temporal weight
  gamma: 0.3    # Topological weight
  
  time_kernel:
    kind: "double_exponential"
    tau_before: 3600   # 1 hour
    tau_after: 1800    # 30 minutes
  
  topo_kernel:
    kind: "exponential"
    tau: 1.0

  stage1:
    mode: "hybrid"       # BM25 + Dense fusion
    candidates: 1000     # Top-K from stage 1

methods:
  bm25: {enabled: true}
  dense: {enabled: true}
  hybrid: {enabled: true}
  lar_rag_dense: {enabled: true}
  lar_rag_hybrid: {enabled: true}
  lar_rag_bm25: {enabled: true}
```

### Custom Config

Create your own config:

```powershell
cp config/bgl_canonical_v2.yaml config/my_experiment.yaml
# Edit config/my_experiment.yaml
python scripts/paper_run_local.py --config config/my_experiment.yaml
```

**Important**: Config must follow canonical schema (validated on load)

---

## Results

### Results Files

After running experiments:

```
results/bgl_rev5_local/
├── results.json           # Per-query + aggregate metrics
├── results.csv            # Summary table
└── run_meta.json          # Config hash + git commit
```

### Metrics

Standard IR metrics:

- **Recall@10**: Fraction of relevant templates in top-10
- **MRR@10**: Mean reciprocal rank (position of first relevant)
- **NDCG@10**: Normalized discounted cumulative gain
- **MAP**: Mean average precision
- **Latency (P95)**: 95th percentile query latency (ms)

### Example Results

```json
{
  "bm25": {
    "recall_at_10": 0.65,
    "mrr_at_10": 0.42,
    "ndcg_at_10": 0.58,
    "map": 0.41,
    "latency_p95_ms": 12.3
  },
  "lar_rag_hybrid": {
    "recall_at_10": 0.89,
    "mrr_at_10": 0.72,
    "ndcg_at_10": 0.83,
    "map": 0.71,
    "latency_p95_ms": 52.3
  }
}
```

**Key Insight**: LAR-RAG (hybrid) improves Recall@10 by 37% over BM25

---

## Common Tasks

### Force Rebuild All Artifacts

```powershell
python scripts/paper_run_local.py --config config/bgl_canonical_v2.yaml --force-rebuild
```

Rebuilds:
- Occurrence store
- FAISS index
- BM25 index
- Ground truth
- Experiment results

### Run Single Method

Edit `config/bgl_canonical_v2.yaml`:

```yaml
methods:
  bm25: {enabled: false}
  dense: {enabled: false}
  hybrid: {enabled: false}
  lar_rag_dense: {enabled: false}
  lar_rag_hybrid: {enabled: true}    # Only this one
  lar_rag_bm25: {enabled: false}
```

Then run:

```powershell
python -m src.evaluation.run_experiments_v2 --config config/bgl_canonical_v2.yaml
```

### Change Number of Incidents

Edit ground truth generation:

```powershell
python scripts/build_ground_truth_streaming.py `
  --parsed-logs data/processed/bgl/parsed.jsonl `
  --template-map data/processed/bgl/template_map.jsonl `
  --output data/processed/bgl/ground_truth.jsonl `
  --seed 42 `
  --max-incidents 500 `    # Changed from 100
  --min-relevant 3
```

Then update config:

```yaml
ground_truth:
  incidents_path: "data/processed/bgl/ground_truth_incidents.jsonl"
  qrels_path: "data/processed/bgl/ground_truth_qrels.jsonl"
```

### Tune LAR-RAG Parameters

Edit config:

```yaml
lar_rag:
  alpha: 0.8    # Reduce similarity weight
  beta: 0.5     # Increase temporal weight
  gamma: 0.2    # Reduce topological weight
  
  time_kernel:
    tau_before: 7200   # Expand to 2 hours
```

Then run experiments:

```powershell
python -m src.evaluation.run_experiments_v2 --config config/bgl_canonical_v2.yaml
```

---

## Troubleshooting

### Error: "Embeddings manifest not found"

**Cause**: Embeddings not generated yet

**Solution**: Run embedding generation first:

```powershell
python scripts/embed_templates_to_blob.py `
  --templates data/processed/bgl/templates.jsonl `
  --output-dir data/processed/bgl/embeddings
```

### Error: "Forbidden legacy key: 'template_map'"

**Cause**: Using old config schema

**Solution**: Migrate to canonical v2 schema:

```yaml
# OLD (forbidden)
template_map: "data/processed/bgl/template_map.jsonl"

# NEW (canonical v2)
data:
  templates_path: "data/processed/bgl/templates.jsonl"
  occurrence_store:
    db_path: "data/processed/bgl/occurrence_store.duckdb"
```

### Error: "DuckDB database locked"

**Cause**: Multiple processes accessing occurrence store

**Solution**: Close all Python processes and retry

### Low Recall@10 (<0.5)

**Possible Causes**:
1. Ground truth too sparse (increase `--max-incidents`)
2. Embeddings outdated (rebuild FAISS index)
3. Wrong kernel parameters (check tau values)

**Debug**:
```powershell
# Check ground truth stats
python -c "import json; incidents = [json.loads(l) for l in open('data/processed/bgl/ground_truth_incidents.jsonl')]; print(f'Incidents: {len(incidents)}'); qrels = [json.loads(l) for l in open('data/processed/bgl/ground_truth_qrels.jsonl')]; print(f'Avg relevant: {sum(len(q[\"relevant_template_ids\"]) for q in qrels) / len(qrels):.1f}')"
```

### Slow LAR-RAG Queries (>100ms P95)

**Possible Causes**:
1. DuckDB not indexed (rebuild occurrence store)
2. Too many Stage 1 candidates (reduce from 1000 to 500)
3. Large template_map (>100K templates)

**Optimization**:
```yaml
lar_rag:
  stage1:
    candidates: 500  # Reduce from 1000
```

---

## Best Practices

### 1. Always Use Canonical Config

✅ **Good**: Start from `config/bgl_canonical_v2.yaml`
❌ **Bad**: Create config from scratch

### 2. Track Config Hash

Every result includes config hash in `run_meta.json`:

```json
{
  "config_hash": "a3f5b2c1...",
  "git_commit": "e7d3a9f2..."
}
```

Use this to verify reproducibility.

### 3. Use Force Rebuild Sparingly

Only use `--force-rebuild` when:
- Config parameters changed
- Source data updated
- Debugging index issues

**Reason**: Rebuilding all artifacts takes ~3 minutes

### 4. Check Git Status

Before running experiments:

```powershell
git status  # Should be clean
git log -1  # Note commit hash
```

After running:

```powershell
cat results/bgl_rev5_local/run_meta.json
# Should match git commit
```

### 5. Save Results Immediately

After experiments:

```powershell
# Copy to versioned results
cp -r results/bgl_rev5_local results/bgl_rev5_local_$(git log -1 --format=%h)

# Or commit
git add results/bgl_rev5_local/
git commit -m "BGL experiments - commit $(git log -1 --format=%h)"
```

---

## Advanced Usage

### Ablation Studies

Test kernel components:

```yaml
# Similarity-only (no LAR-RAG)
lar_rag:
  alpha: 1.0
  beta: 0.0    # Disable temporal
  gamma: 0.0   # Disable topological

# Temporal-only
lar_rag:
  alpha: 0.0    # Disable similarity
  beta: 1.0
  gamma: 0.0

# Topological-only
lar_rag:
  alpha: 0.0
  beta: 0.0
  gamma: 1.0
```

### Parameter Sweep

Create multiple configs:

```powershell
# Alpha sweep
for alpha in 0.5 0.7 1.0 1.5 2.0
do
  cp config/bgl_canonical_v2.yaml config/bgl_alpha_$alpha.yaml
  # Edit lar_rag.alpha = $alpha
  python -m src.evaluation.run_experiments_v2 --config config/bgl_alpha_$alpha.yaml
done
```

### Cross-Dataset Evaluation

Test on HDFS:

1. Prepare HDFS data:
   ```powershell
   python scripts/parse_hdfs.py
   python scripts/embed_hdfs_templates.py
   ```

2. Create HDFS config:
   ```yaml
   run:
     name: "hdfs_rev5_local"
   
   data:
     templates_path: "data/processed/hdfs/templates.jsonl"
     embeddings:
       manifest_path: "data/processed/hdfs/embeddings/manifest.json"
   ```

3. Run pipeline:
   ```powershell
   python scripts/paper_run_local.py --config config/hdfs_canonical_v2.yaml
   ```

---

## Reproducibility Checklist

Before submitting to INFOCOM:

- [ ] Clean git status (`git status` shows no changes)
- [ ] Config hash recorded (`run_meta.json` saved)
- [ ] Git commit recorded (`run_meta.json` includes commit)
- [ ] Seed set in config (`run.seed = 42`)
- [ ] All artifacts versioned (committed to git)
- [ ] Results backed up (copied to safe location)
- [ ] Environment documented (`pip freeze > requirements_frozen.txt`)

---

## Support

### Documentation

- **Architecture**: `docs/LAR-RAG_V2_WIRING_COMPLETION.md`
- **Implementation**: `docs/INFOCOM_IMPLEMENTATION_SUMMARY.md`
- **Workflow**: `docs/INFOCOM_WORKFLOW.md`

### Code References

- **Experiment Harness**: `src/evaluation/run_experiments_v2.py`
- **Config Validation**: `src/evaluation/config_validation.py`
- **LAR-RAG v2**: `src/retrieval/lar_rag_v2.py`
- **Occurrence Store**: `src/storage/occurrence_store.py`

### Contact

For questions:
1. Check `docs/LAR-RAG_V2_WIRING_COMPLETION.md`
2. Review `config/bgl_canonical_v2.yaml` comments
3. Examine `src/evaluation/run_experiments_v2.py` docstrings
