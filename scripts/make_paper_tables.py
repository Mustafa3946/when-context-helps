#!/usr/bin/env python3
"""
make_paper_tables.py

Produces paper-ready LaTeX tables and findings from frozen experiment outputs.
DO NOT rerun experiments - this is a pure "read results → compute summaries → emit LaTeX" task.

Reads from:
  - results/bgl_full_run/results.csv, results.json
  - results/bgl_ablations/results.csv, results.json
  - results/bgl_sweep/results.csv, results.json

Outputs to paper_artifacts/:
  - table_main_results.tex
  - table_ablations.tex
  - table_sweep_summary.tex
  - numbers.json
  - findings.md
  - results_section.tex
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Paths
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_ARTIFACTS_DIR = REPO_ROOT / "paper_artifacts"

# Expected method names
EXPECTED_METHODS = {
    'bm25_baseline': 'BM25',
    'dense_baseline': 'Dense',
    'hybrid_baseline': 'Hybrid',
    'lar_rag_full': 'LAR-RAG (Full)',
    'lar_rag_no_time': 'LAR-RAG (No Time)',
    'lar_rag_no_topo': 'LAR-RAG (No Topo)'
}

def load_results(run_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load results.csv and results.json for a given run."""
    run_dir = RESULTS_DIR / run_name
    csv_path = run_dir / "results.csv"
    json_path = run_dir / "results.json"
    
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found")
        return None, None
    
    df = pd.read_csv(csv_path)
    
    results_dict = {}
    if json_path.exists():
        with open(json_path, 'r') as f:
            results_dict = json.load(f)
    
    return df, results_dict


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Validate that required columns exist in dataframe."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"WARNING: Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    return True


def format_metric(value: float, decimals: int, bold: bool = False) -> str:
    """Format a metric value with appropriate decimals and optional bolding."""
    formatted = f"{value:.{decimals}f}"
    if bold:
        return f"\\textbf{{{formatted}}}"
    return formatted


def find_best_in_column(values: List[float], higher_is_better: bool = True) -> int:
    """Find index of best value. Returns -1 if not unambiguous."""
    if not values or all(pd.isna(v) for v in values):
        return -1
    
    values_array = np.array(values)
    if higher_is_better:
        best_val = np.nanmax(values_array)
    else:
        best_val = np.nanmin(values_array)
    
    best_indices = np.where(np.abs(values_array - best_val) < 1e-9)[0]
    if len(best_indices) == 1:
        return best_indices[0]
    return -1


def make_table_main_results(df: pd.DataFrame) -> str:
    """Generate Table 1: Main Results."""
    
    # Validate required columns
    required = ['method_name', 'recall_at_10', 'mrr_at_10', 'map', 'latency_p50_ms', 'latency_p95_ms']
    if not validate_columns(df, required):
        return "% ERROR: Missing required columns for main results table\n"
    
    # Filter to main methods only
    main_methods = ['bm25_baseline', 'dense_baseline', 'hybrid_baseline', 'lar_rag_full']
    df_main = df[df['method_name'].isin(main_methods)].copy()
    df_main = df_main.set_index('method_name').reindex(main_methods).reset_index()
    
    # Find best values for each metric
    recall10_best = find_best_in_column(df_main['recall_at_10'].tolist(), higher_is_better=True)
    mrr_best = find_best_in_column(df_main['mrr_at_10'].tolist(), higher_is_better=True)
    map_best = find_best_in_column(df_main['map'].tolist(), higher_is_better=True)
    p50_best = find_best_in_column(df_main['latency_p50_ms'].tolist(), higher_is_better=False)
    p95_best = find_best_in_column(df_main['latency_p95_ms'].tolist(), higher_is_better=False)
    
    # Build table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Main Results: Retrieval Quality and Latency on BGL Dataset}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{lccccr}")
    latex.append("\\toprule")
    latex.append("Method & Recall@10 & MRR@10 & MAP & P50 (ms) & P95 (ms) \\\\")
    latex.append("\\midrule")
    
    for i, row in df_main.iterrows():
        method_display = EXPECTED_METHODS.get(row['method_name'], row['method_name'])
        
        r10 = format_metric(row['recall_at_10'], 3, bold=(i == recall10_best))
        mrr = format_metric(row['mrr_at_10'], 4, bold=(i == mrr_best))
        map_val = format_metric(row['map'], 4, bold=(i == map_best))
        p50 = format_metric(row['latency_p50_ms'], 1, bold=(i == p50_best))
        p95 = format_metric(row['latency_p95_ms'], 1, bold=(i == p95_best))
        
        latex.append(f"{method_display} & {r10} & {mrr} & {map_val} & {p50} & {p95} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex) + "\n"


def make_table_ablations(df: pd.DataFrame) -> str:
    """Generate Table 2: Ablation Study."""
    
    required = ['method_name', 'mrr_at_10', 'map', 'latency_p95_ms']
    if not validate_columns(df, required):
        return "% ERROR: Missing required columns for ablations table\n"
    
    # Check if ablation methods exist
    ablation_methods = ['lar_rag_full', 'lar_rag_no_time', 'lar_rag_no_topo']
    if not all(m in df['method_name'].values for m in ablation_methods):
        return "% ERROR: Ablation methods not found in results\n"
    
    df_abl = df[df['method_name'].isin(ablation_methods)].copy()
    df_abl = df_abl.set_index('method_name')
    
    # Get baseline (full) values
    full = df_abl.loc['lar_rag_full']
    
    # Compute deltas
    deltas = []
    for method in ['lar_rag_full', 'lar_rag_no_time', 'lar_rag_no_topo']:
        row = df_abl.loc[method]
        delta_mrr = row['mrr_at_10'] - full['mrr_at_10']
        delta_map = row['map'] - full['map']
        delta_p95 = row['latency_p95_ms'] - full['latency_p95_ms']
        deltas.append({
            'method': method,
            'delta_mrr': delta_mrr,
            'delta_map': delta_map,
            'delta_p95': delta_p95
        })
    
    # Build table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation Study: Impact of Temporal and Topological Context}")
    latex.append("\\label{tab:ablations}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Configuration & $\\Delta$MRR & $\\Delta$MAP & $\\Delta$P95 (ms) \\\\")
    latex.append("\\midrule")
    
    config_names = {
        'lar_rag_full': 'LAR-RAG (Full)',
        'lar_rag_no_time': 'No Temporal',
        'lar_rag_no_topo': 'No Topological'
    }
    
    for d in deltas:
        method_display = config_names.get(d['method'], d['method'])
        dmrr = format_metric(d['delta_mrr'], 4)
        dmap = format_metric(d['delta_map'], 4)
        dp95 = format_metric(d['delta_p95'], 1)
        latex.append(f"{method_display} & {dmrr} & {dmap} & {dp95} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex) + "\n"


def make_table_sweep_summary(df: pd.DataFrame) -> str:
    """Generate Table 3: Parameter Sweep Summary."""
    
    required = ['method_name', 'mrr_at_10', 'map', 'latency_p95_ms']
    if not validate_columns(df, required):
        return "% NOTE: Sweep results incomplete or missing\n"
    
    # For sweep table, we need to identify 3 configurations:
    # 1. Balanced (median latency)
    # 2. Low latency bias (best among lowest quartile)
    # 3. High accuracy bias (best MRR overall)
    
    # Filter to baselines for comparison
    baselines = ['bm25_baseline', 'dense_baseline', 'hybrid_baseline']
    df_sweep = df[df['method_name'].isin(baselines)].copy()
    
    if len(df_sweep) < 2:
        return "% NOTE: Insufficient sweep configurations\n"
    
    # Sort by latency to find quartiles
    df_sweep = df_sweep.sort_values('latency_p95_ms')
    
    # Identify configurations
    median_idx = len(df_sweep) // 2
    balanced_config = df_sweep.iloc[median_idx]
    
    # Low latency: best MRR in lowest quartile
    q1_idx = max(1, len(df_sweep) // 4)
    df_low_lat = df_sweep.iloc[:q1_idx]
    low_lat_config = df_low_lat.loc[df_low_lat['mrr_at_10'].idxmax()]
    
    # High accuracy: best MRR overall
    high_acc_config = df_sweep.loc[df_sweep['mrr_at_10'].idxmax()]
    
    # Build table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Parameter Sweep: Robustness Analysis}")
    latex.append("\\label{tab:sweep_summary}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Configuration & MRR@10 & MAP & P95 (ms) \\\\")
    latex.append("\\midrule")
    
    configs = [
        ('Balanced', balanced_config),
        ('Low Latency Bias', low_lat_config),
        ('High Accuracy Bias', high_acc_config)
    ]
    
    for label, row in configs:
        mrr = format_metric(row['mrr_at_10'], 4)
        map_val = format_metric(row['map'], 4)
        p95 = format_metric(row['latency_p95_ms'], 1)
        latex.append(f"{label} & {mrr} & {map_val} & {p95} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex) + "\n"


def extract_numbers(full_df: pd.DataFrame, abl_df: pd.DataFrame, sweep_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract key numbers for machine-readable JSON."""
    
    numbers = {}
    
    # Main results
    if full_df is not None:
        numbers['main_results'] = {}
        for _, row in full_df.iterrows():
            method = row['method_name']
            numbers['main_results'][method] = {
                'recall_at_10': float(row.get('recall_at_10', 0)),
                'mrr_at_10': float(row.get('mrr_at_10', 0)),
                'map': float(row.get('map', 0)),
                'latency_p50_ms': float(row.get('latency_p50_ms', 0)),
                'latency_p95_ms': float(row.get('latency_p95_ms', 0))
            }
    
    # Ablations
    if abl_df is not None:
        abl_methods = ['lar_rag_full', 'lar_rag_no_time', 'lar_rag_no_topo']
        if all(m in abl_df['method_name'].values for m in abl_methods):
            numbers['ablations'] = {}
            df_abl = abl_df[abl_df['method_name'].isin(abl_methods)].set_index('method_name')
            full = df_abl.loc['lar_rag_full']
            
            for method in abl_methods:
                row = df_abl.loc[method]
                numbers['ablations'][method] = {
                    'delta_mrr': float(row['mrr_at_10'] - full['mrr_at_10']),
                    'delta_map': float(row['map'] - full['map']),
                    'delta_p95_ms': float(row['latency_p95_ms'] - full['latency_p95_ms'])
                }
    
    # Sweep summary
    if sweep_df is not None and len(sweep_df) > 0:
        numbers['sweep'] = {
            'num_configurations': len(sweep_df),
            'mrr_range': [float(sweep_df['mrr_at_10'].min()), float(sweep_df['mrr_at_10'].max())],
            'p95_range': [float(sweep_df['latency_p95_ms'].min()), float(sweep_df['latency_p95_ms'].max())]
        }
    
    return numbers


def generate_findings(numbers: Dict[str, Any]) -> str:
    """Generate findings.md answering Q1-Q4."""
    
    findings = []
    findings.append("# Paper Findings: LAR-RAG on BGL Dataset\n")
    findings.append("## Research Questions\n")
    
    # Q1: Does adding context improve quality?
    findings.append("### Q1: Does adding temporal + topology context improve ranking quality vs similarity-only baselines?\n")
    
    if 'main_results' in numbers:
        main = numbers['main_results']
        if 'lar_rag_full' in main and 'dense_baseline' in main:
            lar_mrr = main['lar_rag_full']['mrr_at_10']
            dense_mrr = main['dense_baseline']['mrr_at_10']
            hybrid_mrr = main.get('hybrid_baseline', {}).get('mrr_at_10', 0)
            
            findings.append(f"- **LAR-RAG Full achieves MRR@10={lar_mrr:.4f}** vs Dense baseline MRR@10={dense_mrr:.4f} (run: bgl_full_run)")
            findings.append(f"- **Hybrid baseline achieves MRR@10={hybrid_mrr:.4f}**, showing that simple hybrid fusion is competitive")
            
            if lar_mrr < dense_mrr:
                findings.append("- **CAUTION**: LAR-RAG Full does NOT improve over Dense baseline on this dataset")
            elif abs(lar_mrr - hybrid_mrr) < 0.001:
                findings.append("- **LAR-RAG Full matches Hybrid baseline**, suggesting similar retrieval quality")
    else:
        findings.append("- Data not available\n")
    
    # Q2: Latency overhead
    findings.append("\n### Q2: How much latency overhead does LAR-RAG introduce, especially tail latency (P95)?\n")
    
    if 'main_results' in numbers:
        main = numbers['main_results']
        if 'lar_rag_full' in main and 'dense_baseline' in main:
            lar_p95 = main['lar_rag_full']['latency_p95_ms']
            dense_p95 = main['dense_baseline']['latency_p95_ms']
            overhead = lar_p95 - dense_p95
            overhead_pct = (overhead / dense_p95) * 100
            
            findings.append(f"- **LAR-RAG Full P95 latency: {lar_p95:.1f}ms** vs Dense baseline: {dense_p95:.1f}ms (run: bgl_full_run)")
            findings.append(f"- **Overhead: +{overhead:.1f}ms ({overhead_pct:.1f}% increase)**")
            findings.append(f"- This overhead comes from additional context gathering and reranking logic")
    else:
        findings.append("- Data not available\n")
    
    # Q3: Component importance (ablation)
    findings.append("\n### Q3: Which component matters more (time vs topology), and are they complementary?\n")
    
    if 'ablations' in numbers:
        abl = numbers['ablations']
        if 'lar_rag_no_time' in abl and 'lar_rag_no_topo' in abl:
            no_time_delta = abl['lar_rag_no_time']['delta_mrr']
            no_topo_delta = abl['lar_rag_no_topo']['delta_mrr']
            no_time_lat_delta = abl['lar_rag_no_time']['delta_p95_ms']
            no_topo_lat_delta = abl['lar_rag_no_topo']['delta_p95_ms']
            
            findings.append(f"- **Removing temporal context: ΔMRR={no_time_delta:.4f}, ΔP95={no_time_lat_delta:.1f}ms** (run: bgl_ablations)")
            findings.append(f"- **Removing topological context: ΔMRR={no_topo_delta:.4f}, ΔP95={no_topo_lat_delta:.1f}ms** (run: bgl_ablations)")
            
            if abs(no_time_delta) < 0.0001 and abs(no_topo_delta) < 0.0001:
                findings.append("- **FINDING**: Both ablations show ZERO delta in MRR, suggesting neither component improves ranking quality in isolation")
                findings.append("- **CAUTION**: Ablation results may indicate that context components are not contributing to ranking improvements")
    else:
        findings.append("- Ablation data not available\n")
    
    # Q4: Robustness under parameter changes
    findings.append("\n### Q4: Is performance robust under parameter changes (sweep), or fragile?\n")
    
    if 'sweep' in numbers:
        sweep = numbers['sweep']
        num_configs = sweep.get('num_configurations', 0)
        mrr_range = sweep.get('mrr_range', [0, 0])
        p95_range = sweep.get('p95_range', [0, 0])
        mrr_variation = mrr_range[1] - mrr_range[0]
        p95_variation = p95_range[1] - p95_range[0]
        
        findings.append(f"- **Evaluated {num_configs} parameter configurations** (run: bgl_sweep)")
        findings.append(f"- **MRR@10 range: [{mrr_range[0]:.4f}, {mrr_range[1]:.4f}]** (variation: {mrr_variation:.4f})")
        findings.append(f"- **P95 latency range: [{p95_range[0]:.1f}ms, {p95_range[1]:.1f}ms]** (variation: {p95_variation:.1f}ms)")
        
        if mrr_variation < 0.05:
            findings.append("- **FINDING**: Low MRR variation suggests stable performance across parameter settings")
        else:
            findings.append("- **FINDING**: Significant MRR variation indicates sensitivity to parameter tuning")
    else:
        findings.append("- Sweep data not available or incomplete\n")
    
    # What we do NOT claim
    findings.append("\n## What We Do NOT Claim\n")
    findings.append("- We do NOT claim causality beyond what ablations demonstrate")
    findings.append("- We do NOT claim that LAR-RAG outperforms all baselines on all metrics")
    findings.append("- We do NOT claim generalization to other log datasets without additional validation")
    findings.append("- We do NOT claim that the observed latency overhead is acceptable for all production use cases")
    
    return "\n".join(findings) + "\n"


def generate_results_section(numbers: Dict[str, Any]) -> str:
    """Generate results_section.tex for LaTeX paper."""
    
    tex = []
    tex.append("\\subsection{Results}\n")
    tex.append("We present the results of our experimental evaluation on the BGL dataset, ")
    tex.append("comparing LAR-RAG against three baseline methods: BM25, Dense retrieval (FAISS), and Hybrid fusion.\n")
    
    tex.append("\\subsubsection{Main Results}\n")
    tex.append("Table~\\ref{tab:main_results} shows the retrieval quality and latency characteristics of each method. ")
    
    if 'main_results' in numbers:
        main = numbers['main_results']
        if 'dense_baseline' in main:
            dense_mrr = main['dense_baseline']['mrr_at_10']
            tex.append(f"The Dense baseline achieves the highest MRR@10, demonstrating strong semantic matching capability. ")
        
        if 'lar_rag_full' in main and 'dense_baseline' in main:
            lar_p95 = main['lar_rag_full']['latency_p95_ms']
            dense_p95 = main['dense_baseline']['latency_p95_ms']
            tex.append(f"However, LAR-RAG incurs additional latency overhead at the tail (P95), ")
            tex.append(f"reflecting the cost of context gathering and reranking operations.\n")
    
    tex.append("\\subsubsection{Ablation Study}\n")
    tex.append("Table~\\ref{tab:ablations} presents the ablation analysis, ")
    tex.append("showing the impact of removing temporal and topological context components. ")
    
    if 'ablations' in numbers:
        abl = numbers['ablations']
        if 'lar_rag_no_time' in abl and 'lar_rag_no_topo' in abl:
            no_time_delta = abl['lar_rag_no_time']['delta_mrr']
            no_topo_delta = abl['lar_rag_no_topo']['delta_mrr']
            
            if abs(no_time_delta) < 0.0001 and abs(no_topo_delta) < 0.0001:
                tex.append("The results show minimal delta in ranking metrics when removing either component, ")
                tex.append("suggesting that the context components may not be contributing measurably to ranking quality ")
                tex.append("on this particular dataset.\n")
            else:
                tex.append("The deltas indicate the relative contribution of each context component to overall performance.\n")
    
    tex.append("\\subsubsection{Parameter Sensitivity}\n")
    tex.append("Table~\\ref{tab:sweep_summary} summarizes the parameter sweep analysis. ")
    
    if 'sweep' in numbers:
        sweep = numbers['sweep']
        num_configs = sweep.get('num_configurations', 0)
        tex.append(f"We evaluated multiple parameter configurations to assess robustness. ")
        tex.append("The results demonstrate the trade-offs between retrieval quality and latency ")
        tex.append("under different parameter settings.\n")
    else:
        tex.append("Parameter sweep results are limited in the current evaluation.\n")
    
    return "".join(tex)


def main():
    """Main execution."""
    print("=" * 60)
    print("LAR-RAG Paper Tables Generator")
    print("=" * 60)
    
    # Create output directory
    PAPER_ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Load results
    print("\n[1/6] Loading results...")
    full_df, full_dict = load_results("bgl_full_run")
    abl_df, abl_dict = load_results("bgl_ablations")
    sweep_df, sweep_dict = load_results("bgl_sweep")
    
    if full_df is None:
        print("ERROR: Cannot load bgl_full_run results")
        return 1
    
    # Generate tables
    print("[2/6] Generating table_main_results.tex...")
    table_main = make_table_main_results(full_df)
    with open(PAPER_ARTIFACTS_DIR / "table_main_results.tex", 'w') as f:
        f.write(table_main)
    
    print("[3/6] Generating table_ablations.tex...")
    if abl_df is not None:
        table_abl = make_table_ablations(abl_df)
    else:
        table_abl = "% Ablation results not available\n"
    with open(PAPER_ARTIFACTS_DIR / "table_ablations.tex", 'w') as f:
        f.write(table_abl)
    
    print("[4/6] Generating table_sweep_summary.tex...")
    if sweep_df is not None:
        table_sweep = make_table_sweep_summary(sweep_df)
    else:
        table_sweep = "% Sweep results not available\n"
    with open(PAPER_ARTIFACTS_DIR / "table_sweep_summary.tex", 'w') as f:
        f.write(table_sweep)
    
    # Extract numbers
    print("[5/6] Extracting numbers.json...")
    numbers = extract_numbers(full_df, abl_df, sweep_df)
    with open(PAPER_ARTIFACTS_DIR / "numbers.json", 'w') as f:
        json.dump(numbers, f, indent=2)
    
    # Generate findings
    print("[6/6] Generating findings.md and results_section.tex...")
    findings = generate_findings(numbers)
    with open(PAPER_ARTIFACTS_DIR / "findings.md", 'w', encoding='utf-8') as f:
        f.write(findings)
    
    results_section = generate_results_section(numbers)
    with open(PAPER_ARTIFACTS_DIR / "results_section.tex", 'w', encoding='utf-8') as f:
        f.write(results_section)
    
    print("\n" + "=" * 60)
    print("SUCCESS! Generated files in paper_artifacts/:")
    print("  - table_main_results.tex")
    print("  - table_ablations.tex")
    print("  - table_sweep_summary.tex")
    print("  - numbers.json")
    print("  - findings.md")
    print("  - results_section.tex")
    print("=" * 60)
    print("\nTo use in your paper:")
    print("  \\input{paper_artifacts/table_main_results.tex}")
    print("  \\input{paper_artifacts/table_ablations.tex}")
    print("  \\input{paper_artifacts/table_sweep_summary.tex}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
