"""
Real statistical tests for LAR-RAG evaluation.

Implements proper paired statistical tests using per-query metrics.
No simulation - all tests use actual per-query scores.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from scipy import stats


class PairedStatisticalTests:
    """
    Statistical tests for comparing retrieval systems.
    
    All tests are PAIRED (same queries for both systems) and use
    real per-query scores (no simulation).
    """
    
    @staticmethod
    def load_per_query_results(jsonl_path: str) -> List[Dict]:
        """Load per-query results from JSONL file.
        
        Args:
            jsonl_path: Path to per_query_*.jsonl file
            
        Returns:
            List of per-query result dicts
        """
        results = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        return results
    
    @staticmethod
    def extract_metric(per_query_results: List[Dict], metric_name: str) -> np.ndarray:
        """Extract a specific metric from per-query results.
        
        Args:
            per_query_results: List of per-query dicts
            metric_name: Name of metric to extract (e.g., 'reciprocal_rank')
            
        Returns:
            NumPy array of metric values
        """
        return np.array([r[metric_name] for r in per_query_results])
    
    @staticmethod
    def paired_bootstrap_ci(
        baseline_scores: np.ndarray,
        model_scores: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        random_seed: int = 42
    ) -> Dict:
        """
        Bootstrap confidence interval for mean difference.
        
        Uses paired bootstrap resampling to estimate CI for:
        mean(model) - mean(baseline)
        
        Args:
            baseline_scores: Per-query scores for baseline (e.g., MRR)
            model_scores: Per-query scores for model
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default 95%)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dict with mean_diff, ci_lower, ci_upper, p_value_estimate
        """
        if len(baseline_scores) != len(model_scores):
            raise ValueError("Baseline and model must have same number of queries")
        
        np.random.seed(random_seed)
        n = len(baseline_scores)
        
        # Observed difference
        observed_diff = np.mean(model_scores) - np.mean(baseline_scores)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample indices (paired sampling)
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_baseline = baseline_scores[indices]
            bootstrap_model = model_scores[indices]
            
            # Calculate difference
            diff = np.mean(bootstrap_model) - np.mean(bootstrap_baseline)
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        
        # Estimate p-value (proportion of bootstrap samples with diff <= 0)
        p_value_estimate = np.mean(bootstrap_diffs <= 0)
        # Two-tailed p-value
        p_value_estimate = 2 * min(p_value_estimate, 1 - p_value_estimate)
        
        return {
            'mean_diff': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value_estimate': p_value_estimate,
            'baseline_mean': np.mean(baseline_scores),
            'model_mean': np.mean(model_scores),
            'n_queries': n,
            'n_bootstrap': n_bootstrap,
            'confidence': confidence
        }
    
    @staticmethod
    def wilcoxon_signed_rank(
        baseline_scores: np.ndarray,
        model_scores: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Non-parametric test for whether model performs better than baseline.
        Does not assume normal distribution.
        
        Args:
            baseline_scores: Per-query scores for baseline
            model_scores: Per-query scores for model
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dict with statistic, p_value, and interpretation
        """
        if len(baseline_scores) != len(model_scores):
            raise ValueError("Baseline and model must have same number of queries")
        
        # Calculate differences
        differences = model_scores - baseline_scores
        
        # Wilcoxon test
        try:
            statistic, p_value = stats.wilcoxon(differences, alternative=alternative)
        except ValueError as e:
            # All differences are zero
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'alternative': alternative,
                'significant': False,
                'interpretation': 'No difference between systems',
                'error': str(e)
            }
        
        # Interpretation
        alpha = 0.05
        significant = p_value < alpha
        
        if alternative == 'greater':
            interpretation = f"Model significantly better (p={p_value:.4f})" if significant else f"No significant improvement (p={p_value:.4f})"
        elif alternative == 'less':
            interpretation = f"Baseline significantly better (p={p_value:.4f})" if significant else f"No significant difference (p={p_value:.4f})"
        else:  # two-sided
            interpretation = f"Significant difference (p={p_value:.4f})" if significant else f"No significant difference (p={p_value:.4f})"
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'alternative': alternative,
            'significant': significant,
            'interpretation': interpretation,
            'mean_diff': np.mean(differences),
            'median_diff': np.median(differences),
            'n_queries': len(baseline_scores)
        }
    
    @staticmethod
    def paired_t_test(
        baseline_scores: np.ndarray,
        model_scores: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Paired t-test for comparing means.
        
        Parametric test assuming normal distribution of differences.
        Use Wilcoxon if normality assumption violated.
        
        Args:
            baseline_scores: Per-query scores for baseline
            model_scores: Per-query scores for model
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dict with statistic, p_value, confidence_interval
        """
        if len(baseline_scores) != len(model_scores):
            raise ValueError("Baseline and model must have same number of queries")
        
        # Paired t-test
        statistic, p_value = stats.ttest_rel(model_scores, baseline_scores, alternative=alternative)
        
        # Calculate 95% CI for mean difference
        differences = model_scores - baseline_scores
        mean_diff = np.mean(differences)
        se_diff = stats.sem(differences)
        ci = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=se_diff)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'alternative': alternative,
            'significant': p_value < 0.05,
            'mean_diff': mean_diff,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_queries': len(baseline_scores)
        }
    
    @staticmethod
    def compare_systems(
        baseline_path: str,
        model_path: str,
        metric: str = 'reciprocal_rank',
        verbose: bool = True
    ) -> Dict:
        """
        Complete statistical comparison of two systems.
        
        Runs multiple tests and returns comprehensive results.
        
        Args:
            baseline_path: Path to baseline per_query.jsonl
            model_path: Path to model per_query.jsonl
            metric: Metric to compare ('reciprocal_rank', 'average_precision', etc.)
            verbose: Print detailed results
            
        Returns:
            Dict with all test results
        """
        # Load results
        baseline_results = PairedStatisticalTests.load_per_query_results(baseline_path)
        model_results = PairedStatisticalTests.load_per_query_results(model_path)
        
        # Extract metric arrays
        baseline_scores = PairedStatisticalTests.extract_metric(baseline_results, metric)
        model_scores = PairedStatisticalTests.extract_metric(model_results, metric)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Statistical Comparison: {Path(baseline_path).stem} vs {Path(model_path).stem}")
            print(f"Metric: {metric}")
            print(f"{'='*80}\n")
            
            print(f"Baseline mean: {np.mean(baseline_scores):.4f} (SD: {np.std(baseline_scores):.4f})")
            print(f"Model mean:    {np.mean(model_scores):.4f} (SD: {np.std(model_scores):.4f})")
            print(f"Difference:    {np.mean(model_scores) - np.mean(baseline_scores):.4f}")
            print(f"% Improvement: {(np.mean(model_scores) / np.mean(baseline_scores) - 1) * 100:.1f}%\n")
        
        # Run tests
        bootstrap = PairedStatisticalTests.paired_bootstrap_ci(baseline_scores, model_scores)
        wilcoxon = PairedStatisticalTests.wilcoxon_signed_rank(baseline_scores, model_scores, alternative='greater')
        t_test = PairedStatisticalTests.paired_t_test(baseline_scores, model_scores, alternative='greater')
        
        if verbose:
            print(f"Bootstrap 95% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")
            print(f"  p-value (estimated): {bootstrap['p_value_estimate']:.4f}\n")
            
            print(f"Wilcoxon Signed-Rank Test:")
            print(f"  Statistic: {wilcoxon['statistic']:.2f}")
            print(f"  p-value: {wilcoxon['p_value']:.4f}")
            print(f"  {wilcoxon['interpretation']}\n")
            
            print(f"Paired t-test:")
            print(f"  Statistic: {t_test['statistic']:.2f}")
            print(f"  p-value: {t_test['p_value']:.4f}")
            print(f"  95% CI: [{t_test['ci_lower']:.4f}, {t_test['ci_upper']:.4f}]\n")
        
        return {
            'metric': metric,
            'baseline_path': str(baseline_path),
            'model_path': str(model_path),
            'bootstrap': bootstrap,
            'wilcoxon': wilcoxon,
            't_test': t_test
        }


def format_results_for_paper(comparison: Dict) -> str:
    """
    Format statistical test results for paper reporting.
    
    Args:
        comparison: Dict from compare_systems()
        
    Returns:
        LaTeX-ready string
    """
    bootstrap = comparison['bootstrap']
    wilcoxon = comparison['wilcoxon']
    
    # Calculate relative improvement
    baseline_mean = bootstrap['baseline_mean']
    model_mean = bootstrap['model_mean']
    rel_improvement = (model_mean / baseline_mean - 1) * 100
    
    # Format for paper (fix f-string backslash error)
    sig_text = r"\textbf{statistically significant}" if wilcoxon['significant'] else "not significant"
    result = f"""
\\textbf{{Statistical Test Results}}

Model achieves {model_mean:.4f} vs baseline {baseline_mean:.4f} ({rel_improvement:+.1f}\\% improvement).

Bootstrap 95\\% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}], $p < {bootstrap['p_value_estimate']:.3f}$

Wilcoxon signed-rank test: $W = {wilcoxon['statistic']:.0f}$, $p = {wilcoxon['p_value']:.4f}$

Result is {sig_text} at $\\alpha = 0.05$.
"""
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("Statistical Tests Module - LAR-RAG Evaluation")
    print("=" * 80)
    
    # Simulate per-query results for demonstration
    np.random.seed(42)
    n_queries = 30
    
    # Baseline: MRR around 0.05
    baseline_mrr = np.random.beta(2, 38, n_queries)  # Mean ~ 0.05
    
    # Model: MRR around 0.13 (+160%)
    model_mrr = np.random.beta(5, 35, n_queries)  # Mean ~ 0.125
    
    print(f"\nSimulated Experiment:")
    print(f"Baseline MRR: {np.mean(baseline_mrr):.4f} ± {np.std(baseline_mrr):.4f}")
    print(f"Model MRR:    {np.mean(model_mrr):.4f} ± {np.std(model_mrr):.4f}")
    print(f"Improvement:  +{(np.mean(model_mrr)/np.mean(baseline_mrr) - 1)*100:.1f}%")
    
    # Run statistical tests
    print("\n" + "="*80)
    print("Bootstrap Confidence Interval")
    print("="*80)
    bootstrap_results = PairedStatisticalTests.paired_bootstrap_ci(
        baseline_mrr, model_mrr, n_bootstrap=10000
    )
    print(f"Mean difference: {bootstrap_results['mean_diff']:.4f}")
    print(f"95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
    print(f"p-value (estimated): {bootstrap_results['p_value_estimate']:.4f}")
    
    print("\n" + "="*80)
    print("Wilcoxon Signed-Rank Test")
    print("="*80)
    wilcoxon_results = PairedStatisticalTests.wilcoxon_signed_rank(
        baseline_mrr, model_mrr, alternative='greater'
    )
    print(f"Statistic: {wilcoxon_results['statistic']:.2f}")
    print(f"p-value: {wilcoxon_results['p_value']:.4f}")
    print(f"{wilcoxon_results['interpretation']}")
    
    print("\n" + "="*80)
    print("Paired t-test")
    print("="*80)
    t_test_results = PairedStatisticalTests.paired_t_test(
        baseline_mrr, model_mrr, alternative='greater'
    )
    print(f"Statistic: {t_test_results['statistic']:.2f}")
    print(f"p-value: {t_test_results['p_value']:.4f}")
    print(f"95% CI: [{t_test_results['ci_lower']:.4f}, {t_test_results['ci_upper']:.4f}]")
    print(f"Significant: {t_test_results['significant']}")
    
    print("\n" + "="*80)
    print("Interpretation for Paper")
    print("="*80)
    print(f"\nLAR-RAG achieves MRR of {np.mean(model_mrr):.4f} compared to baseline's {np.mean(baseline_mrr):.4f},")
    print(f"representing a {(np.mean(model_mrr)/np.mean(baseline_mrr) - 1)*100:.1f}% improvement.")
    print(f"This difference is statistically significant (Wilcoxon test, p={wilcoxon_results['p_value']:.4f}).")
    print(f"Bootstrap 95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
    
    # Demonstrate proper caveats
    print("\n" + "="*80)
    print("IMPORTANT CAVEATS FOR PAPER")
    print("="*80)
    print("""
1. Sample Size: n=30 queries is relatively small for robust statistical inference.
   - Consider reporting effect size (Cohen's d) in addition to p-values
   - Acknowledge limited statistical power

2. Multiple Comparisons: If testing multiple ablations, apply Bonferroni correction.
   - For k=6 comparisons, use α=0.05/6=0.0083

3. Assumptions:
   - Queries are independent (check for dataset-specific effects)
   - Wilcoxon does not assume normality (good for skewed metrics like MRR)
   - Bootstrap CI is more robust than parametric CI for small samples

4. Reporting:
   - Always report: n, mean difference, 95% CI, p-value
   - State which test was primary (Wilcoxon recommended)
   - Show both absolute and relative improvements
   - Include standard deviations or error bars in figures

5. Honest Framing:
   - "statistically significant but small effect size" vs "large practical improvement"
   - Acknowledge single dataset limitation
   - Note that ground truth is constructed (not gold-standard human labels)
""")
