# LAR-RAG Experiment Execution Guide
## Quick Reference for INFOCOM 2026 Runs

**Status:** Ready for execution  
**Last Updated:** December 23, 2025

---

## Prerequisites

### Required Artifacts
Before running experiments, ensure these files exist:
- [ ] `data/processed/bgl/parsed.jsonl` - Parsed BGL logs
- [ ] `data/processed/bgl/templates.jsonl` - Log template corpus
- [ ] `data/processed/bgl/template_map.jsonl` - Template metadata

### Python Environment
```bash
# Activate virtual environment
.venv\Scripts\activate

# Verify dependencies installed
pip list | findstr "pyarrow duckdb rank-bm25 numpy faiss"
```

---

## Phase 1: Smoke Run (MANDATORY)

### Purpose
Validate end-to-end pipeline with 10 incidents before full runs.

### Command
```bash
python scripts/paper_run_local.py --config config/bgl_smoke_run.yaml
```

### Expected Duration
~5-10 minutes (includes building indices, ground truth, and running all methods)

### Validation Checklist
After completion, verify:
- [ ] No network calls occurred (check logs for Azure/OpenAI warnings)
- [ ] All methods returned non-empty rankings (BM25, Dense, Hybrid, LAR-RAG)
- [ ] LAR-RAG output includes `similarity`, `time_score`, `topology_score` components
- [ ] Metrics files written to `results/smoke_run/`:
  - [ ] `results.json` - Aggregate metrics
  - [ ] `results_per_query.jsonl` - Per-incident details
  - [ ] `run_metadata.json` - Configuration snapshot
- [ ] Latency p50/p95 are reasonable (no extreme spikes)

### Troubleshooting
If smoke run fails:
1. Check `data/processed/bgl/` directory has required files
2. Verify Python environment has all dependencies
3. Inspect error logs for missing artifacts
4. Use `--force-rebuild` to regenerate indices if needed

---

## Phase 2: Baseline-Only Full Run

### Purpose
Establish reference performance table.

### Configuration
Edit `config/bgl_canonical_v2.yaml`:
```yaml
methods:
  bm25_baseline:
    enabled: true
  dense_baseline:
    enabled: true
  hybrid_baseline:
    enabled: true
  lar_rag_full:
    enabled: false  # DISABLE for baseline-only run
  lar_rag_no_time:
    enabled: false
  lar_rag_no_topo:
    enabled: false
```

### Command
```bash
python scripts/paper_run_local.py --config config/bgl_canonical_v2.yaml
```

### Expected Duration
~30-60 minutes (depends on dataset size and hardware)

### Validation
- [ ] Hybrid Recall@10 ≥ max(BM25 Recall@10, Dense Recall@10)
- [ ] Results saved to `results/experiments/results.json`

---

## Phase 3: LAR-RAG Main Run + Ablations

### Purpose
Demonstrate value of time and topology components.

### Configuration
Re-enable all LAR-RAG variants in `config/bgl_canonical_v2.yaml`:
```yaml
methods:
  bm25_baseline:
    enabled: true
  dense_baseline:
    enabled: true
  hybrid_baseline:
    enabled: true
  lar_rag_full:
    enabled: true  # ENABLE
  lar_rag_no_time:
    enabled: true  # ENABLE
  lar_rag_no_topo:
    enabled: true  # ENABLE
```

### Command
```bash
python scripts/paper_run_local.py --config config/bgl_canonical_v2.yaml
```

### Expected Duration
~1-2 hours (full dataset with all ablations)

### Validation
- [ ] LAR-RAG Full ≥ Hybrid (Recall@10 or MRR@10)
- [ ] Ablations show: Dense < +Time < Full OR Dense < +Topo < Full
- [ ] Latency overhead bounded (LAR-RAG p95 < 2× Hybrid p95)

---

## Configurations

### Smoke Run
- **File:** `config/bgl_smoke_run.yaml`
- **Incidents:** 10
- **Output:** `results/smoke_run/`
- **Purpose:** Quick validation

### Full Run
- **File:** `config/bgl_canonical_v2.yaml`
- **Incidents:** 100 (or full test set)
- **Output:** `results/experiments/`
- **Purpose:** INFOCOM paper results

---

## Output Structure

```
results/
├── smoke_run/                    # Phase 1 output
│   ├── results.json
│   ├── results_per_query.jsonl
│   └── run_metadata.json
└── experiments/                  # Phase 2 & 3 output
    ├── results.json
    ├── results_per_query.jsonl
    └── run_metadata.json
```

### Results JSON Schema
```json
{
  "method_name": {
    "num_queries": 100,
    "recall_at_1": 0.45,
    "recall_at_5": 0.72,
    "recall_at_10": 0.85,
    "mrr_at_10": 0.62,
    "ndcg_at_10": 0.78,
    "map": 0.71,
    "latency_mean_ms": 12.3,
    "latency_p50_ms": 10.5,
    "latency_p95_ms": 25.1,
    "latency_p99_ms": 35.2
  }
}
```

---

## Command Reference

### Full Pipeline
```bash
# Smoke run (10 incidents)
python scripts/paper_run_local.py --config config/bgl_smoke_run.yaml

# Full run with all methods
python scripts/paper_run_local.py --config config/bgl_canonical_v2.yaml

# Force rebuild all artifacts
python scripts/paper_run_local.py --config config/bgl_canonical_v2.yaml --force-rebuild
```

### Individual Steps
```bash
# Build occurrence store only
python scripts/build_occurrence_store.py \
  --template-map data/processed/bgl/template_map.jsonl \
  --output data/processed/bgl/occurrences.duckdb

# Build FAISS index only
python scripts/build_faiss_index.py \
  --embeddings-dir data/processed/bgl/embeddings \
  --out data/index/bgl_faiss \
  --normalize

# Build BM25 index only
python scripts/build_bm25_index.py \
  --templates data/processed/bgl/templates.jsonl \
  --output data/processed/bgl/bm25_index

# Build ground truth only
python scripts/build_ground_truth_streaming.py \
  --parsed-logs data/processed/bgl/parsed.jsonl \
  --template-map data/processed/bgl/template_map.jsonl \
  --output data/evaluation/bgl_ground_truth.jsonl \
  --seed 42 \
  --max-incidents 10
```

---

## Guardrails

### DO NOT
- ✗ Change scoring functions during experiments
- ✗ Tune on test split
- ✗ Rerun with different seeds without documenting
- ✗ Modify ground truth after first run

### DO
- ✓ Run smoke test first
- ✓ Validate each phase before proceeding
- ✓ Save all results with clear filenames
- ✓ Document any deviations from protocol

---

## Troubleshooting

### "Module not found" errors
```bash
# Ensure virtual environment active
.venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "File not found" errors
```bash
# Check required files exist
ls data/processed/bgl/parsed.jsonl
ls data/processed/bgl/templates.jsonl
ls data/processed/bgl/template_map.jsonl

# If missing, run preprocessing pipeline first
```

### "Out of memory" errors
- Reduce `max_cached_shards` in config (default: 16)
- Use smaller `max_test_incidents` for testing
- Close other applications

### Slow performance
- Ensure indices are built (not rebuilding each time)
- Check SSD vs HDD storage (indices should be on SSD)
- Verify `local_only: true` (no network overhead)

---

## Next Steps After Execution

1. **Inspect Results**
   - Open `results/experiments/results.json`
   - Verify LAR-RAG ≥ Hybrid
   - Check ablation monotonicity

2. **Generate Paper Tables**
   - Use results to populate INFOCOM paper tables
   - Include confidence intervals if multiple runs

3. **Archive Results**
   - Git commit summaries to `results/summaries/`
   - Tag release: `git tag v1.0-experiments`

---

**Ready to execute Phase 1 smoke run!**
