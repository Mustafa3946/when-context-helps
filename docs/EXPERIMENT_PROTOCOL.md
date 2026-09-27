# LAR-RAG Experiment Protocol
## IEEE INFOCOM 2026 Submission

**Status:** FROZEN (do not modify after first commit)  
**Date:** December 23, 2025  
**Version:** 1.0

---

## 1. Dataset

### Source
- **Dataset:** Blue Gene/L (BGL) Supercomputer Logs
- **Source:** [LogHub Repository](https://github.com/logpai/loghub)
- **Size:** 4,747,963 log entries
- **Time span:** Continuous operational logs

### Preprocessing
- **Parser:** Drain algorithm (default LogPai configuration)
- **Templates extracted:** ~1,169,037 unique log templates
- **Template map:** `data/processed/bgl/template_map.jsonl`
- **Parsed logs:** `data/processed/bgl/parsed.jsonl`

### Artifact Paths (IMMUTABLE)
```
data/processed/bgl/
  ├── parsed.jsonl              # Parsed log entries with template_id
  ├── templates.jsonl           # Log template corpus
  ├── template_map.jsonl        # Template metadata
  └── embeddings/
      ├── manifest.json         # Embedding shard manifest
      └── shards/               # Parquet-sharded template embeddings
          └── shard_*.parquet
```

---

## 2. Incident Definition

### Incident Construction
An **incident** is defined as:
- A randomly sampled template ID (query template)
- Time window: uniformly sampled from log timeline
- Relevant templates: templates co-occurring within ±1 hour window
- Minimum relevant templates: 3 (ensures non-trivial retrieval)

### Ground Truth Generation
- **Method:** Streaming builder with DuckDB backend
- **Seed:** 42 (for reproducibility)
- **Max incidents:** Configurable (100 for full run, 10 for smoke)
- **Output:**
  - `data/evaluation/bgl_incidents.jsonl` - Incident metadata
  - `data/evaluation/bgl_qrels.jsonl` - Relevance judgments

---

## 3. Train/Test Split

### Method
**Time-based split** (chronological)
- Ensures realistic temporal evaluation
- Prevents data leakage from future to past
- Reflects production deployment scenario

### Configuration
- **Train fraction:** 0.7 (70% earliest incidents)
- **Test fraction:** 0.3 (30% latest incidents)
- **Rationale:** Train period establishes occurrence patterns, test period validates generalization

### Smoke Run
- **Train fraction:** 0.0 (no training split)
- **Test incidents:** 10 (all incidents used for quick validation)

---

## 4. Metrics

### Retrieval Quality
1. **Recall@K** (K ∈ {1, 5, 10, 20})
   - Primary metric: Recall@10
   - Measures coverage of relevant templates in top-K

2. **Mean Reciprocal Rank (MRR@10)**
   - Focuses on rank of first relevant result
   - Penalizes late discovery

3. **Normalized Discounted Cumulative Gain (nDCG@10)**
   - Graded relevance metric
   - Rewards higher-ranked relevant results

4. **Mean Average Precision (MAP)**
   - Precision-recall curve summary
   - Standard IR benchmark

### Performance
1. **Latency p50** (median query latency)
   - Typical-case performance

2. **Latency p95** (95th percentile)
   - Worst-case tolerance
   - Critical for production readiness

### Reporting
All metrics computed per-query and aggregated (mean across test set).

---

## 5. Baselines

### BM25 Baseline
- **Method:** BM25Okapi (rank-bm25 library)
- **Index:** Offline pickled index
- **Query:** Incident template text
- **Purpose:** Lexical matching baseline

### Dense Baseline
- **Method:** Dense embedding retrieval (FAISS)
- **Embeddings:** Pre-computed template embeddings (sharded Parquet)
- **Index:** FAISS L2 index with normalization
- **Query:** Incident template embedding
- **Purpose:** Semantic matching baseline

### Hybrid Baseline
- **Method:** BM25 + Dense fusion
- **Stage 1:** BM25 retrieves top-1000 candidates
- **Stage 2:** Dense reranks candidates
- **Fusion:** Linear combination (α_BM25=0.3, α_Dense=0.7)
- **Purpose:** State-of-the-art retrieval baseline

---

## 6. LAR-RAG Configuration

### Default Parameters
```yaml
alpha: 1.0   # Semantic similarity weight
beta: 0.3    # Temporal proximity weight
gamma: 0.3   # Topology similarity weight
```

### Time Kernel (Temporal Scoring)
- **Type:** Asymmetric exponential decay
- **τ_before:** 3600 seconds (1 hour before incident)
- **τ_after:** 7200 seconds (2 hours after incident)
- **Rationale:** Root causes precede symptoms, effects follow

### Topology Kernel (Co-occurrence Scoring)
- **Type:** Exponential decay
- **τ:** 2.0 (decay parameter)
- **Occurrence store:** DuckDB-backed (disk-resident)
- **Rationale:** Frequently co-occurring templates are causally related

### Stage-1 Retrieval
- **Mode:** Hybrid (BM25 + Dense)
- **Candidates:** 1000 (before LAR-RAG reranking)

---

## 7. Ablation Studies

### Ablation Configurations
1. **Dense only** (α=1, β=0, γ=0)
   - Baseline: semantic similarity only

2. **+Time only** (α=1, β>0, γ=0)
   - Semantic + temporal scoring

3. **+Topology only** (α=1, β=0, γ>0)
   - Semantic + co-occurrence scoring

4. **Full LAR-RAG** (α=1, β>0, γ>0)
   - All components active

### Expected Behavior
- Each component (time, topology) should improve metrics over Dense
- Full LAR-RAG should be best or tie with best ablation
- Monotonic improvement validates component contributions

---

## 8. Reproducibility Constraints

### Immutable Artifacts
Once experiments begin, the following MUST NOT change:
- ✓ Dataset (BGL parsed logs, templates, embeddings)
- ✓ Ground truth (incidents, qrels)
- ✓ Train/test split (seed=42, time-based, 70/30)
- ✓ Metric definitions
- ✓ Baseline configurations

### Allowed Variations (documented)
- LAR-RAG hyperparameters (α, β, γ, τ values)
- Top-K values for reporting
- Additional ablations (if time permits)

### Forbidden During Experiments
- ✗ Re-parsing logs
- ✗ Changing incident definition
- ✗ Re-splitting train/test with different seed
- ✗ Modifying scoring functions
- ✗ Tuning on test set

---

## 9. Offline Operation

### Local-Only Mode
- **Flag:** `local_only: true` (enforced in all configs)
- **Validation:** RuntimeError if Azure/OpenAI modules loaded
- **Rationale:** Reproducibility, no API dependencies, defensible latency

### Artifacts Required
- ✓ Pre-computed template embeddings (Parquet shards)
- ✓ FAISS index (offline)
- ✓ BM25 index (pickled)
- ✓ Occurrence store (DuckDB)
- ✓ Ground truth (incidents + qrels)

---

## 10. Execution Phases

### Phase 1: Smoke Run (MANDATORY)
- **Purpose:** Validate pipeline end-to-end
- **Incidents:** 10
- **Methods:** All (BM25, Dense, Hybrid, LAR-RAG)
- **Validation:**
  - No network calls
  - Non-empty rankings
  - LAR-RAG components present
  - Metrics files written
  - Latency stable

### Phase 2: Baseline-Only Full Run
- **Incidents:** 100 (or full test set)
- **Methods:** BM25, Dense, Hybrid only
- **Purpose:** Reference table

### Phase 3: LAR-RAG Main Run + Ablations
- **Incidents:** Same as Phase 2
- **Methods:** All ablations (Dense, +Time, +Topo, Full)
- **Purpose:** Demonstrate component value

---

## 11. Result Inspection Checklist

Before writing:
- [ ] LAR-RAG ≥ Hybrid in Recall@10 or MRR
- [ ] Time/topology ablations show monotonic improvement
- [ ] Latency overhead is bounded (<2x Hybrid)
- [ ] No method returns empty rankings
- [ ] Metrics are stable across runs (same seed)

---

## 12. Version Control

### Git Commit Protocol
- Commit this protocol before first experiment
- Tag experiment runs: `v1.0-smoke`, `v1.0-baselines`, `v1.0-full`
- Results stored in `results/` directory (gitignored)
- Summaries committed to `results/summaries/`

---

## 13. Contact

**Maintainer:** LAR-RAG Research Team  
**Conference:** IEEE INFOCOM 2026  
**Revision:** rev11 (verification complete)

**FROZEN - DO NOT MODIFY AFTER COMMIT**
