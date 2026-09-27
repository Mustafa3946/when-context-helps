# When Context Helps

Code and evaluation harness for **"When Context Helps: An Empirical Study of Temporal and Topological Context in System Log Retrieval"** — an empirical study of LAR-RAG, a two-stage retrieval framework that augments hybrid sparse–dense log-template retrieval with temporal decay and topology proximity re-ranking, evaluated on HDFS, BGL, and Thunderbird.

The paper investigates whether contextual (temporal/topological) re-ranking improves log-based root cause analysis evidence retrieval, and under what structural conditions it does or doesn't.

## Repository structure

```
src/            Retrieval, evaluation, indexing, embedding, and preprocessing pipeline
config/         Canonical (v2 schema) experiment configs: full runs, ablations, HDFS sensitivity sweep
scripts/        Pipeline orchestrators, data download helpers, table/metric generation
docs/           Experiment protocol, algorithm description, run guides
tests/          Unit tests
requirements.txt
```

## Datasets

Public log datasets via Loghub/Zenodo:
- HDFS: https://zenodo.org/records/3227177/files/HDFS_1.tar.gz
- BGL: https://zenodo.org/records/3227177/files/BGL.tar.gz
- Thunderbird: https://zenodo.org/records/8196385/files/Thunderbird.tar.gz

See `scripts/download_hdfs.ps1` and `scripts/download_bgl.ps1` for automated download; Thunderbird download instructions are printed by `scripts/run_thunderbird_pipeline.py` if the raw log isn't found.

## Running experiments

See `docs/RUN_EXPERIMENTS.md` for full-run, ablation, and sweep commands per dataset. All experiments run offline (local FAISS/BM25/DuckDB indices, no external API calls) once data is processed and indexed.

## Reproducibility package

Archived retrieval outputs, independent audit/analysis scripts, and figures are published separately as a Zenodo reproducibility package (DOI to be added once deposited).

## Related work

This code shares its retrieval implementation and benchmark setting with the authors' earlier published work, *Quality–Latency Benchmarking of RCA Template Retrieval for GenAI-Driven Network Operations* (Applied Artificial Intelligence, 2026): https://doi.org/10.1080/08839514.2026.2684312

## License

MIT — see `LICENSE`.
