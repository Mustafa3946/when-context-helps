"""
Fill Thunderbird table placeholders in patch_thunderbird_table.tex
with actual numbers from results/thunderbird_full_run/results.json.

Usage:
    python scripts/fill_thunderbird_table.py
    python scripts/fill_thunderbird_table.py --results results/thunderbird_full_run/results.json
"""
import json
import argparse
from pathlib import Path


PLACEHOLDER_MAP = {
    # BM25
    r"\TB@bm25@r10":   ("bm25_baseline",   "recall_at_10",   "{:.3f}"),
    r"\TB@bm25@mrr":   ("bm25_baseline",   "mrr_at_10",      "{:.4f}"),
    r"\TB@bm25@map":   ("bm25_baseline",   "map",            "{:.4f}"),
    r"\TB@bm25@p50":   ("bm25_baseline",   "latency_p50_ms", "{:.1f}"),
    r"\TB@bm25@p95":   ("bm25_baseline",   "latency_p95_ms", "{:.1f}"),
    # Dense
    r"\TB@dense@r10":  ("dense_baseline",  "recall_at_10",   "{:.3f}"),
    r"\TB@dense@mrr":  ("dense_baseline",  "mrr_at_10",      "{:.4f}"),
    r"\TB@dense@map":  ("dense_baseline",  "map",            "{:.4f}"),
    r"\TB@dense@p50":  ("dense_baseline",  "latency_p50_ms", "{:.1f}"),
    r"\TB@dense@p95":  ("dense_baseline",  "latency_p95_ms", "{:.1f}"),
    # Hybrid
    r"\TB@hybrid@r10": ("hybrid_baseline", "recall_at_10",   "{:.3f}"),
    r"\TB@hybrid@mrr": ("hybrid_baseline", "mrr_at_10",      "{:.4f}"),
    r"\TB@hybrid@map": ("hybrid_baseline", "map",            "{:.4f}"),
    r"\TB@hybrid@p50": ("hybrid_baseline", "latency_p50_ms", "{:.1f}"),
    r"\TB@hybrid@p95": ("hybrid_baseline", "latency_p95_ms", "{:.1f}"),
    # LAR-RAG
    r"\TB@lar@r10":    ("lar_rag_full",    "recall_at_10",   "{:.3f}"),
    r"\TB@lar@mrr":    ("lar_rag_full",    "mrr_at_10",      "{:.4f}"),
    r"\TB@lar@map":    ("lar_rag_full",    "map",            "{:.4f}"),
    r"\TB@lar@p50":    ("lar_rag_full",    "latency_p50_ms", "{:.1f}"),
    r"\TB@lar@p95":    ("lar_rag_full",    "latency_p95_ms", "{:.1f}"),
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
                        default="results/thunderbird_full_run/results.json")
    parser.add_argument("--template",
                        default="paper_artifacts/paper_versions/review_2/patch_thunderbird_table.tex")
    parser.add_argument("--output",
                        default="paper_artifacts/paper_versions/review_2/patch_thunderbird_table_filled.tex")
    args = parser.parse_args()

    fill_table(args.results, args.template, args.output)
