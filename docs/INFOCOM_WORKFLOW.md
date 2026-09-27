# INFOCOM-Compliant LAR-RAG Workflow

This README describes the INFOCOM-ready evaluation pipeline with streaming architecture and disk-backed queries.

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have:
# - data/processed/bgl/parsed.jsonl (4.4M logs)
# - data/processed/bgl/templates.jsonl (1.169M templates)
# - data/processed/bgl/template_map.jsonl (4.4M mappings)
# - data/processed/bgl/template_embeddings/ (Parquet shards + manifest.json)
```

### Step 1: Build Occurrence Store

Convert `template_map.jsonl` to disk-backed DuckDB database:

```bash
python scripts/build_occurrence_store.py \
  --template-map data/processed/bgl/template_map.jsonl \
  --output data/processed/bgl/occurrences.duckdb
```

**Expected Output**:
- ✅ 4.4M occurrences inserted
- ✅ Indexes created (template_id, timestamp, node_id)
- Database size: ~200MB

**Performance**: ~30 seconds on modern hardware

### Step 2: Generate Ground Truth

Create ground truth using streaming (no full log loading):

```bash
python scripts/build_ground_truth_streaming.py \
  --parsed-logs data/processed/bgl/parsed.jsonl \
  --template-map data/processed/bgl/template_map.jsonl \
  --output data/evaluation/bgl_ground_truth.jsonl \
  --seed 42 \
  --max-incidents 100 \
  --min-relevant 3 \
  --time-window 1.0
```

**Expected Output**:
- ✅ ~87 valid ground truth entries
- Average 12 relevant templates per incident
- Output: `data/evaluation/bgl_ground_truth.jsonl`

**Performance**: ~2-3 minutes (streaming, low memory)

### Step 3: Build FAISS Index

If not already built:

```bash
python scripts/build_faiss_index.py \
  --embeddings-dir data/processed/bgl/template_embeddings \
  --out data/index/bgl_faiss \
  --normalize
```

**Expected Output**:
- ✅ FAISS index built (1.169M templates)
- Output: `data/index/bgl_faiss/`

### Step 4: Validate Configuration

Ensure configuration is INFOCOM-compliant:

```bash
python -m src.retrieval.factory --config config/bgl_eval_v2.yaml
```

**Expected Output**:
```
✅ ALL METHODS VALIDATED - INFOCOM COMPLIANT
```

### Step 5: Run Evaluation

Execute all retrieval methods:

```bash
python scripts/run_bgl_eval.py \
  --config config/bgl_eval_v2.yaml \
  --output results/bgl_eval_v2
```

**Expected Metrics**:
- Hit Rate @1, @5, @10
- MRR (Mean Reciprocal Rank)
- mAP (Mean Average Precision)
- Latency per query

---

## Architecture Overview

### Memory-Safe Design

**Old (Memory-Intensive)**:
```
Template Map (4.4M entries) → RAM (2.2GB)
LAR-RAG Ranker: Preload all mappings → 2.2GB RAM
```

**New (Streaming + Lazy)**:
```
Template Map → DuckDB (200MB disk)
LAR-RAG Ranker: Query on-demand → 50KB RAM per query
```

**Memory Reduction**: 44× less RAM (2.2GB → 50MB)

### Component Hierarchy

```
┌─────────────────────────────────────────────────┐
│ Experiment Runner (run_bgl_eval.py)            │
└─────────────────────────────────────────────────┘
              │
              ├─► RetrieverFactory (validates config)
              │
              ├─► BM25Retriever (lexical baseline)
              │
              ├─► DenseRetriever (FAISS semantic)
              │
              ├─► HybridRetriever (BM25 + Dense)
              │
              └─► LARRAGRankerV2 (lazy loading)
                      │
                      ├─► HybridRetriever (Stage 1)
                      ├─► OccurrenceStore (disk queries)
                      └─► SystemTopologyGraph (graph distances)
```

---

## Configuration File

`config/bgl_eval_v2.yaml`:

```yaml
data:
  templates: data/processed/bgl/templates.jsonl
  occurrence_store: data/processed/bgl/occurrences.duckdb
  faiss_index: data/index/bgl_faiss
  ground_truth: data/evaluation/bgl_ground_truth.jsonl

methods:
  bm25_baseline:
    type: bm25
  
  dense_baseline:
    type: dense
    index_dir: data/index/bgl_faiss
  
  hybrid_baseline:
    type: hybrid
    index_dir: data/index/bgl_faiss
  
  lar_rag_full:
    type: lar_rag
    occurrence_store_path: data/processed/bgl/occurrences.duckdb
    alpha: 0.4  # Semantic
    beta: 0.3   # Temporal
    gamma: 0.3  # Topology
  
  lar_rag_no_time:
    type: lar_rag
    alpha: 0.7
    beta: 0.0   # Ablation: No temporal
    gamma: 0.3
  
  lar_rag_no_topo:
    type: lar_rag
    alpha: 0.7
    beta: 0.3
    gamma: 0.0  # Ablation: No topology

evaluation:
  metrics:
    - hit_rate@1
    - hit_rate@5
    - hit_rate@10
    - mrr
    - map
```

---

## Key Design Decisions

### 1. Streaming Ground Truth

**Why?**
- 4.4M logs cannot fit in RAM on all machines
- Reviewers expect memory-efficient implementation

**How?**
- Pass 1: Stream logs → identify incidents (O(incidents) memory)
- Pass 2: For each incident → stream logs → find relevant (O(1) memory)

**Rationale**: "Streaming architecture ensures scalability to larger log corpora."

### 2. DuckDB Occurrence Store

**Why?**
- Lazy loading requires fast queries
- SQLite too slow for analytical queries
- Parquet requires loading entire partitions

**How?**
- DuckDB: Embedded SQL + columnar performance
- Indexes on (template_id, timestamp, node_id)

**Rationale**: "DuckDB provides O(log n) indexed queries with minimal overhead."

### 3. Lazy LAR-RAG Ranker

**Why?**
- Preloading 4.4M mappings wastes memory
- Stage 1 filters to ~200 candidates

**How?**
- Stage 1: Hybrid retrieval → 200 candidates
- Stage 2: Query OccurrenceStore → 200 queries
- Total memory: O(200) instead of O(4.4M)

**Rationale**: "Lazy loading reduces memory 44× with negligible latency impact."

### 4. Explicit Failure Markers

**Why?**
- Time/node proximity not defensible to reviewers
- Need explicit ground truth rationale

**How?**
- Regex patterns for GPU errors, ECC errors, kernel panics
- Time used ONLY as candidate filter

**Rationale**: "Relevance determined by explicit failure signatures, not heuristics."

---

## Verification

Run verification script to test all components:

```bash
python scripts/verify_infocom_implementation.py
```

**Tests**:
1. ✅ Occurrence store build and queries
2. ✅ Streaming ground truth generation
3. ✅ Configuration validation (rejects Azure Search, JSONL embeddings)
4. ✅ Lazy ranker initialization

---

## Troubleshooting

### "Occurrence store not found"

**Solution**:
```bash
python scripts/build_occurrence_store.py \
  --template-map data/processed/bgl/template_map.jsonl \
  --output data/processed/bgl/occurrences.duckdb
```

### "No ground truth found"

**Solution**:
```bash
python scripts/build_ground_truth_streaming.py \
  --parsed-logs data/processed/bgl/parsed.jsonl \
  --template-map data/processed/bgl/template_map.jsonl \
  --output data/evaluation/bgl_ground_truth.jsonl
```

### "FAISS index not found"

**Solution**:
```bash
python scripts/build_faiss_index.py \
  --embeddings-dir data/processed/bgl/template_embeddings \
  --out data/index/bgl_faiss \
  --normalize
```

### "INFOCOM VIOLATION: JSONL embeddings not allowed"

**Cause**: Configuration points to `.jsonl` embedding file

**Solution**: Update config to use Parquet embeddings or built FAISS index:
```yaml
index_dir: data/index/bgl_faiss  # ✅ Correct
# NOT: embeddings_path: data/embeddings/templates.jsonl  # ❌ Wrong
```

---

## File Checklist

Before running evaluation:

- [ ] `data/processed/bgl/parsed.jsonl` (4.4M logs)
- [ ] `data/processed/bgl/templates.jsonl` (1.169M templates)
- [ ] `data/processed/bgl/template_map.jsonl` (4.4M mappings)
- [ ] `data/processed/bgl/template_embeddings/` (Parquet shards)
- [ ] `data/processed/bgl/occurrences.duckdb` (built)
- [ ] `data/evaluation/bgl_ground_truth.jsonl` (built)
- [ ] `data/index/bgl_faiss/` (built)
- [ ] `config/bgl_eval_v2.yaml` (validated)

---

## Performance Expectations

**Ground Truth Generation**:
- Time: ~2-3 minutes
- Memory: ~50MB peak
- Output: ~87 incidents

**Occurrence Store Build**:
- Time: ~30 seconds
- Memory: ~200MB during build
- Output: ~200MB database

**FAISS Index Build**:
- Time: ~5 minutes
- Memory: ~2GB during build
- Output: ~500MB index

**Evaluation (6 methods, 87 queries)**:
- Time: ~10-15 minutes
- Memory: ~500MB peak
- Output: Metrics table + per-query results

---

## Citation

If using this implementation in your research:

```bibtex
@inproceedings{larrag2026,
  title={Latency-Aware Retrieval for Root-Cause Analysis in Large-Scale Networked Systems},
  author={...},
  booktitle={IEEE INFOCOM 2026},
  year={2026}
}
```

---

## Support

For issues or questions:
1. Check `docs/INFOCOM_IMPLEMENTATION_SUMMARY.md` for detailed documentation
2. Run `python scripts/verify_infocom_implementation.py` to test components
3. Review error messages (designed to be self-explanatory)

---

**Status**: INFOCOM-ready ✅
