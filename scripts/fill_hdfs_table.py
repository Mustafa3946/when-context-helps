"""
Fill HDFS table placeholders in patch_hdfs_table.tex
with actual numbers from results/hdfs_full_run/results.json.

Usage:
    python scripts/fill_hdfs_table.py
    python scripts/fill_hdfs_table.py --results results/hdfs_full_run/results.json
"""
import json
import argparse
from pathlib import Path


PLACEHOLDER_MAP = {
    # BM25
    r"\HDFS@bm25@r10":   ("bm25_baseline",   "recall_at_10",  "{:.3f}"),
    r"\HDFS@bm25@mrr":   ("bm25_baseline",   "mrr_at_10",     "{:.4f}"),
    r"\HDFS@bm25@map":   ("bm25_baseline",   "map",           "{:.4f}"),
    r"\HDFS@bm25@p50":   ("bm25_baseline",   "latency_p50_ms","{:.1f}"),
    r"\HDFS@bm25@p95":   ("bm25_baseline",   "latency_p95_ms","{:.1f}"),
    # Dense
    r"\HDFS@dense@r10":  ("dense_baseline",  "recall_at_10",  "{:.3f}"),
    r"\HDFS@dense@mrr":  ("dense_baseline",  "mrr_at_10",     "{:.4f}"),
    r"\HDFS@dense@map":  ("dense_baseline",  "map",           "{:.4f}"),
    r"\HDFS@dense@p50":  ("dense_baseline",  "latency_p50_ms","{:.1f}"),
    r"\HDFS@dense@p95":  ("dense_baseline",  "latency_p95_ms","{:.1f}"),
    # Hybrid
    r"\HDFS@hybrid@r10": ("hybrid_baseline", "recall_at_10",  "{:.3f}"),
    r"\HDFS@hybrid@mrr": ("hybrid_baseline", "mrr_at_10",     "{:.4f}"),
    r"\HDFS@hybrid@map": ("hybrid_baseline", "map",           "{:.4f}"),
    r"\HDFS@hybrid@p50": ("hybrid_baseline", "latency_p50_ms","{:.1f}"),
    r"\HDFS@hybrid@p95": ("hybrid_baseline", "latency_p95_ms","{:.1f}"),
    # LAR-RAG
    r"\HDFS@lar@r10":    ("lar_rag_full",    "recall_at_10",  "{:.3f}"),
    r"\HDFS@lar@mrr":    ("lar_rag_full",    "mrr_at_10",     "{:.4f}"),
    r"\HDFS@lar@map":    ("lar_rag_full",    "map",           "{:.4f}"),
    r"\HDFS@lar@p50":    ("lar_rag_full",    "latency_p50_ms","{:.1f}"),
    r"\HDFS@lar@p95":    ("lar_rag_full",    "latency_p95_ms","{:.1f}"),
}


def fill_table(results_path: str, template_path: str, output_path: str):
    with open(results_path, 'r') as fh:
        results = json.load(fh)

    with open(template_path, 'r', encoding='utf-8') as fh:
        tex = fh.read()

    replacements = 0
    for placeholder, (method, metric, fmt) in PLACEHOLDER_MAP.items():
        value = results.get(method, {}).get(metric)
        if value is None:
            print(f"  WARN: {method}.{metric} not found in results")
            continue
        replacement = fmt.format(value)
        tex = tex.replace(placeholder, replacement)
        replacements += 1

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(tex)

    print(f"Filled {replacements}/{len(PLACEHOLDER_MAP)} placeholders → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default="results/hdfs_full_run/results.json")
    parser.add_argument("--template",
                        default="paper_artifacts/paper_versions/review_2/patch_hdfs_table.tex")
    parser.add_argument("--output",
                        default="paper_artifacts/paper_versions/review_2/patch_hdfs_table_filled.tex")
    args = parser.parse_args()

    fill_table(args.results, args.template, args.output)
