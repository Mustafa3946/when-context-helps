# LAR-RAG INFOCOM Quick Reference

## One-Command Setup

```bash
# 1. Build occurrence store (30 seconds)
python scripts/build_occurrence_store.py

# 2. Generate ground truth (2-3 minutes)
python scripts/build_ground_truth_streaming.py

# 3. Build FAISS index if needed (5 minutes)
python scripts/build_faiss_index.py \
  --embeddings-dir data/processed/bgl/template_embeddings \
  --out data/index/bgl_faiss --normalize

# 4. Validate configuration
python -m src.retrieval.factory --config config/bgl_eval_v2.yaml

# 5. Run verification tests
python scripts/verify_infocom_implementation.py

# 6. Run evaluation
python scripts/run_bgl_eval.py --config config/bgl_eval_v2.yaml
```

---

## Key Files

| File | Purpose | Size |
|------|---------|------|
| `data/processed/bgl/occurrences.duckdb` | Disk-backed occurrence store | ~200MB |
| `data/evaluation/bgl_ground_truth.jsonl` | Ground truth (87 incidents) | ~50KB |
| `data/index/bgl_faiss/` | FAISS index (1.169M templates) | ~500MB |
| `config/bgl_eval_v2.yaml` | Experiment configuration | - |

---

## Memory Comparison

| Component | Old | New | Reduction |
|-----------|-----|-----|-----------|
| LAR-RAG Ranker | 2.2GB | 50KB | 44× |
| Ground Truth Build | 2.2GB | 50MB | 44× |
| Evaluation | 3GB | 500MB | 6× |

---

## INFOCOM Compliance

✅ Streaming architecture (no full log loading)  
✅ Disk-backed queries (DuckDB occurrence store)  
✅ Lazy loading (on-demand feature extraction)  
✅ Deterministic (seeded sampling, time-sorted)  
✅ Explicit failure markers (no heuristics)  
✅ Local-first (no Azure Search dependency)  
✅ v2 Parquet only (no JSONL embeddings)  
✅ Clear error messages (cites INFOCOM requirements)

---

## Methods Evaluated

1. **BM25**: Lexical baseline
2. **Dense**: FAISS semantic baseline
3. **Hybrid**: BM25 + Dense (0.3 + 0.7)
4. **LAR-RAG Full**: α=0.4, β=0.3, γ=0.3
5. **LAR-RAG No Time**: α=0.7, β=0.0, γ=0.3 (ablation)
6. **LAR-RAG No Topo**: α=0.7, β=0.3, γ=0.0 (ablation)

---

## Metrics

- **Hit Rate @K**: Percentage of queries with relevant template in top-K
- **MRR**: Mean Reciprocal Rank
- **mAP**: Mean Average Precision
- **Latency**: Query processing time

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Occurrence store not found" | Run `python scripts/build_occurrence_store.py` |
| "Ground truth not found" | Run `python scripts/build_ground_truth_streaming.py` |
| "FAISS index not found" | Run `python scripts/build_faiss_index.py ...` |
| "INFOCOM VIOLATION" | Check config uses v2 Parquet, no Azure Search |

---

## Documentation

- **Implementation Details**: `docs/INFOCOM_IMPLEMENTATION_SUMMARY.md`
- **Workflow Guide**: `docs/INFOCOM_WORKFLOW.md`
- **Completion Summary**: `INFOCOM_COMPLETION_SUMMARY.md`

---

**Status**: INFOCOM-ready ✅  
**Memory Reduction**: 44×  
**All Tasks**: Complete
