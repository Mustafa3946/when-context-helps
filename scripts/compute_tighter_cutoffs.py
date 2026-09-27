#!/usr/bin/env python3
"""
compute_tighter_cutoffs.py

Computes MRR@1, nDCG@3, nDCG@5, Recall@3, and Recall@5 from stored per-query
JSONL result files, for every dataset (HDFS, BGL, Thunderbird) main-run and
ablation directory that exists under results/.
No experiments are re-run — ranked lists and relevance grades are read from disk.

Usage:
    python scripts/compute_tighter_cutoffs.py
"""

import json
import math
from pathlib import Path
from typing import Dict, List


def mrr_at_k(retrieved: List[str], relevant_ids: List[str], k: int) -> float:
    """MRR@k: reciprocal rank of the first relevant doc in top-k."""
    relevant_set = set(relevant_ids)
    for rank, tid in enumerate(retrieved[:k], start=1):
        if tid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant_grades: Dict[str, int], k: int) -> float:
    """nDCG@k using graded relevance (exponential gain, matching run_experiments_v2.py's canonical formula)."""
    def dcg(items):
        score = 0.0
        for i, tid in enumerate(items, start=1):
            rel = relevant_grades.get(tid, 0)
            score += (2 ** rel - 1) / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved[:k])

    # Ideal: top-k grades sorted descending
    ideal_items = sorted(relevant_grades.values(), reverse=True)[:k]
    ideal_dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_items))

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@k: fraction of the query's relevant template IDs found in top-k."""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant_set)
    return hits / len(relevant_set)


def compute_for_dir(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Compute tighter-cutoff metrics for all methods in a results directory."""
    method_metrics = {}

    for jsonl_path in sorted(results_dir.glob("per_query_*.jsonl")):
        method_name = jsonl_path.stem.replace("per_query_", "")
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

        mrr1_scores, ndcg3_scores, ndcg5_scores, recall3_scores, recall5_scores = [], [], [], [], []

        for row in rows:
            retrieved = row.get("retrieved_template_ids", [])
            grades = row.get("relevance_grades", {})
            relevant_ids = row.get("relevant_template_ids", [])

            mrr1_scores.append(mrr_at_k(retrieved, relevant_ids, k=1))
            ndcg3_scores.append(ndcg_at_k(retrieved, grades, k=3))
            ndcg5_scores.append(ndcg_at_k(retrieved, grades, k=5))
            recall3_scores.append(recall_at_k(retrieved, relevant_ids, k=3))
            recall5_scores.append(recall_at_k(retrieved, relevant_ids, k=5))

        n = len(rows)
        method_metrics[method_name] = {
            "n_queries": n,
            "MRR@1":    sum(mrr1_scores)    / n if n else 0.0,
            "nDCG@3":   sum(ndcg3_scores)   / n if n else 0.0,
            "nDCG@5":   sum(ndcg5_scores)   / n if n else 0.0,
            "Recall@3": sum(recall3_scores) / n if n else 0.0,
            "Recall@5": sum(recall5_scores) / n if n else 0.0,
        }

    return method_metrics


def print_table(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    col_w = 22
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Method':<{col_w}} {'MRR@1':>8} {'nDCG@3':>8} {'nDCG@5':>8} {'Recall@3':>10} {'Recall@5':>10}  (n)")
    print(f"{'-'*col_w} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}  ---")
    for method, m in metrics.items():
        print(
            f"{method:<{col_w}} {m['MRR@1']:>8.4f} {m['nDCG@3']:>8.4f} {m['nDCG@5']:>8.4f}"
            f" {m['Recall@3']:>10.4f} {m['Recall@5']:>10.4f}"
            f"  ({m['n_queries']})"
        )


def main():
    base = Path(__file__).resolve().parent.parent / "results"

    tables = {
        "HDFS — Main Results  (hdfs_full_run)":         base / "hdfs_full_run",
        "HDFS — Ablation Study (hdfs_ablations)":        base / "hdfs_ablations",
        "BGL — Main Results  (bgl_full_run)":            base / "bgl_full_run",
        "BGL — Ablation Study (bgl_ablations)":           base / "bgl_ablations",
        "Thunderbird — Main Results (thunderbird_full_run)": base / "thunderbird_full_run",
        "Thunderbird — Ablation Study (thunderbird_ablations)": base / "thunderbird_ablations",
    }

    all_results = {}
    for title, results_dir in tables.items():
        if not results_dir.exists():
            print(f"⚠  Directory not found: {results_dir}")
            continue
        metrics = compute_for_dir(results_dir)
        print_table(title, metrics)
        all_results[title] = metrics

        out_path = results_dir / "tighter_cutoffs.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"  Saved: {out_path.relative_to(results_dir.parent.parent)}")

    print(f"\n{'='*90}")
    print("  Recall@5 is cross-checked against results.json where present.")
    print("  Per-directory values also written to results/<run>/tighter_cutoffs.json.")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
