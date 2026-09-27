#!/usr/bin/env python3
"""
run_hyperparameter_sweep.py

Executes the parameter grid defined in a sweep-spec config (e.g.
config/hdfs_sweep.yaml) by generating one full canonical-v2 config per grid
point, invoking src.evaluation.run_experiments_v2 for each, then aggregating
nDCG@10 / MAP across all runs into best/median/IQR/worst-case numbers.

The base config's `run`/`data`/`ground_truth`/`retrieval`/`lar_rag`/`methods`/
`evaluation` sections are only ever run with lar_rag.alpha/beta/gamma and the
time/topo kernel taus overridden per grid point — nothing else changes, so
baseline (BM25/Dense/Hybrid) numbers are read from the dataset's existing
`*_full_run` results instead of being re-run.

Usage:
    python scripts/run_hyperparameter_sweep.py --config config/hdfs_sweep.yaml
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_run_config(base: Dict[str, Any], run_spec: Dict[str, Any], base_name: str) -> Dict[str, Any]:
    """Deep-copy the base config and apply one grid point's overrides."""
    cfg = copy.deepcopy(base)
    cfg.pop("sweep", None)

    run_name = run_spec["name"]
    cfg["run"]["name"] = f"{base_name}_{run_name}"
    cfg["run"]["output_dir"] = f"results/{base_name}/{run_name}"
    cfg["evaluation"]["output_dir"] = cfg["run"]["output_dir"]

    if "alpha" in run_spec:
        cfg["lar_rag"]["alpha"] = run_spec["alpha"]
    if "beta" in run_spec:
        cfg["lar_rag"]["beta"] = run_spec["beta"]
    if "gamma" in run_spec:
        cfg["lar_rag"]["gamma"] = run_spec["gamma"]
    if "tau_before" in run_spec:
        cfg["lar_rag"]["time_kernel"]["tau_before"] = run_spec["tau_before"]
    if "tau_after" in run_spec:
        cfg["lar_rag"]["time_kernel"]["tau_after"] = run_spec["tau_after"]
    if "tau_topo" in run_spec:
        cfg["lar_rag"]["topo_kernel"]["tau"] = run_spec["tau_topo"]

    return cfg


def run_one(generated_path: Path) -> bool:
    """Invoke run_experiments_v2 for a single generated config."""
    cmd = [sys.executable, "-m", "src.evaluation.run_experiments_v2", "--config", str(generated_path)]
    print(f"  -> {generated_path.name}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"     FAILED (exit {result.returncode}):\n{result.stderr[-2000:]}")
        return False
    return True


def load_baseline_reference(dataset_name: str) -> Dict[str, float]:
    """Read Dense baseline nDCG@10/MAP from the dataset's existing full run, for comparison."""
    full_run_path = REPO_ROOT / "results" / f"{dataset_name}_full_run" / "results.json"
    if not full_run_path.exists():
        return {}
    data = json.loads(full_run_path.read_text())
    dense = data.get("dense_baseline", {})
    return {"ndcg_at_10": dense.get("ndcg_at_10"), "map": dense.get("map")}


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "best": float(arr.max()),
        "median": float(np.median(arr)),
        "iqr_low": float(np.percentile(arr, 25)),
        "iqr_high": float(np.percentile(arr, 75)),
        "worst": float(arr.min()),
        "n_configs": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a LAR-RAG hyperparameter sensitivity sweep")
    parser.add_argument("--config", required=True, help="Path to a sweep-spec config, e.g. config/hdfs_sweep.yaml")
    parser.add_argument("--skip-run", action="store_true", help="Skip execution, only re-aggregate existing results")
    args = parser.parse_args()

    spec_path = Path(args.config)
    base = yaml.safe_load(spec_path.read_text())
    base_name = base["run"]["name"]
    runs = base["sweep"]["runs"]

    generated_dir = REPO_ROOT / "config" / "_generated" / base_name
    generated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep '{base_name}': {len(runs)} configurations")

    if not args.skip_run:
        for run_spec in runs:
            cfg = build_run_config(base, run_spec, base_name)
            generated_path = generated_dir / f"{run_spec['name']}.yaml"
            generated_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            ok = run_one(generated_path)
            if not ok:
                print(f"  Skipping aggregation entry for {run_spec['name']} (run failed)")

    # Aggregate
    ndcg_values, map_values, rows = [], [], []
    for run_spec in runs:
        run_name = run_spec["name"]
        results_path = REPO_ROOT / "results" / base_name / run_name / "results.json"
        if not results_path.exists():
            continue
        data = json.loads(results_path.read_text())
        method = data.get("lar_rag_full", {})
        if not method:
            continue
        ndcg_values.append(method["ndcg_at_10"])
        map_values.append(method["map"])
        rows.append({
            "run": run_name,
            "ndcg_at_10": method["ndcg_at_10"],
            "map": method["map"],
            "latency_p95_ms": method["latency_p95_ms"],
        })

    dataset_name = base_name.replace("_sweep", "")
    baseline_ref = load_baseline_reference(dataset_name)

    summary = {
        "sweep_name": base_name,
        "n_configs_completed": len(rows),
        "n_configs_total": len(runs),
        "dense_baseline_reference": baseline_ref,
        "ndcg_at_10": summarize(ndcg_values) if ndcg_values else None,
        "map": summarize(map_values) if map_values else None,
        "per_run": rows,
    }

    out_dir = REPO_ROOT / "results" / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitivity_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*70}\n  Sensitivity summary: {base_name} ({len(rows)}/{len(runs)} configs)\n{'='*70}")
    if ndcg_values:
        n = summary["ndcg_at_10"]
        print(f"nDCG@10  best={n['best']:.4f}  median={n['median']:.4f}  "
              f"IQR=[{n['iqr_low']:.4f}, {n['iqr_high']:.4f}]  worst={n['worst']:.4f}")
    if map_values:
        m = summary["map"]
        print(f"MAP      best={m['best']:.4f}  median={m['median']:.4f}  "
              f"IQR=[{m['iqr_low']:.4f}, {m['iqr_high']:.4f}]  worst={m['worst']:.4f}")
    if baseline_ref:
        print(f"(Dense baseline reference: nDCG@10={baseline_ref['ndcg_at_10']:.4f}, MAP={baseline_ref['map']:.4f})")
    print(f"Saved: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
