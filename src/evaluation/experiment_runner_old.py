"""
Config-driven experiment runner for LAR-RAG evaluation.

Loads experiments from YAML/JSON config, runs baselines and ablations,
exports metrics to CSV/JSON for analysis and plotting.
"""
import json
import yaml
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.two_stage_pipeline import TwoStageLARRAGPipeline
from evaluation.metrics import RetrievalMetrics, EvaluationFramework


class ExperimentRunner:
    """
    Run reproducible experiments from config files.
    
    Supports:
    - Multiple baseline configurations
    - Ablation studies
    - Ground truth evaluation
    - CSV/JSON metrics export
    """
    
    def __init__(self, config_path: str):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to experiment config YAML/JSON file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = []
    
    def _load_config(self) -> Dict:
        """Load experiment configuration."""
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        print(f"Loaded config from {self.config_path}")
        return config
    
    def _load_ground_truth(self) -> Dict:
        """Load ground truth dataset."""
        gt_path = Path(self.config['ground_truth_file'])
        with open(gt_path, 'r') as f:
            return json.load(f)
    
    def _build_test_queries(self, ground_truth: Dict) -> List[Dict]:
        """
        Convert ground truth incidents to test queries.
        
        Args:
            ground_truth: Ground truth dataset
            
        Returns:
            List of test queries with relevant log IDs
        """
        queries = []
        for incident in ground_truth['incidents']:
            queries.append({
                'query_id': incident['incident_id'],
                'query': incident['query'],
                'query_timestamp': incident.get('query_timestamp'),
                'query_node': incident.get('query_node'),
                'relevant': set(incident['relevant_log_ids'])
            })
        
        return queries
    
    def _create_retriever(self, model_config: Dict) -> TwoStageLARRAGPipeline:
        """
        Create retriever from model configuration.
        
        Args:
            model_config: Model configuration dict
            
        Returns:
            Configured pipeline
        """
        return TwoStageLARRAGPipeline(
            stage1_method=model_config.get('stage1_method', 'hybrid'),
            stage1_k=model_config.get('stage1_k', 50),
            stage2_alpha=model_config.get('alpha', 0.5),
            stage2_beta=model_config.get('beta', 0.3),
            stage2_gamma=model_config.get('gamma', 0.2),
            lambda_decay=model_config.get('lambda_decay', 0.001),
            lambda_post=model_config.get('lambda_post', 0.01),
            lambda_g=model_config.get('lambda_g', 0.5),
            topology_mode=model_config.get('topology_mode', 'synthetic'),
            topology_file=model_config.get('topology_file'),
            enable_temporal=model_config.get('enable_temporal', True),
            enable_topology=model_config.get('enable_topology', True)
        )
    
    def run_experiments(self):
        """Run all experiments defined in config."""
        print(f"\n=== Running Experiments: {self.config['experiment_name']} ===\n")
        
        # Load ground truth
        ground_truth = self._load_ground_truth()
        test_queries = self._build_test_queries(ground_truth)
        print(f"Loaded {len(test_queries)} test queries")
        
        # Get evaluation settings
        eval_config = self.config.get('evaluation', {})
        k_values = eval_config.get('k_values', [1, 3, 5, 10])
        
        # Run each model configuration
        for model_config in self.config['models']:
            model_name = model_config['name']
            print(f"\n--- Evaluating {model_name} ---")
            
            # Create retriever
            retriever = self._create_retriever(model_config)
            
            # Evaluate
            evaluator = EvaluationFramework()
            metrics = evaluator.evaluate(retriever, test_queries, k_values)
            
            # Get latency stats
            latency_stats = retriever.get_latency_stats()
            
            # Store results
            result = {
                'model_name': model_name,
                'config': model_config,
                'metrics': metrics,
                'latency': latency_stats,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Print summary
            print(f"  MRR: {metrics['MRR']:.4f}")
            print(f"  mAP: {metrics['mAP']:.4f}")
            print(f"  HR@5: {metrics['HR@5']:.4f}")
            print(f"  Latency (mean): {latency_stats['total_ms']['mean']:.2f}ms")
            print(f"  Latency (P95): {latency_stats['total_ms']['p95']:.2f}ms")
        
        print("\n=== Experiments Complete ===\n")
    
    def export_results(self):
        """Export results to CSV and JSON, including per-query metrics."""
        output_dir = Path(self.config.get('output_dir', 'results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config['experiment_name'].replace(' ', '_')
        
        # Create run-specific directory for per-query results
        run_id = f"{exp_name}_{timestamp}"
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to JSON (full results including per-query)
        json_path = run_dir / "full_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'experiment_name': self.config['experiment_name'],
                'run_id': run_id,
                'config': self.config,
                'results': self.results
            }, f, indent=2)
        print(f"Saved full results to {json_path}")
        
        # Export per-query results to JSONL for each model (for statistical tests)
        for result in self.results:
            model_name = result['model_name'].replace(' ', '_').replace('(', '').replace(')', '')
            per_query_path = run_dir / f"per_query_{model_name}.jsonl"
            with open(per_query_path, 'w') as f:
                for query_result in result['metrics'].get('per_query', []):
                    f.write(json.dumps(query_result) + '\n')
            print(f"Saved per-query results for {result['model_name']} to {per_query_path}")
        
        # Export to CSV (metrics summary)
        csv_path = run_dir / "summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['model_name', 'MRR', 'mAP']
            k_values = self.config.get('evaluation', {}).get('k_values', [1, 3, 5, 10])
            for k in k_values:
                header.append(f'HR@{k}')
            header.extend(['latency_mean_ms', 'latency_p95_ms', 'stage1_mean_ms', 'stage2_mean_ms'])
            writer.writerow(header)
            
            # Data rows
            for result in self.results:
                row = [
                    result['model_name'],
                    f"{result['metrics']['MRR']:.4f}",
                    f"{result['metrics']['mAP']:.4f}"
                ]
                for k in k_values:
                    row.append(f"{result['metrics'][f'HR@{k}']:.4f}")
                row.extend([
                    f"{result['latency']['total_ms']['mean']:.2f}",
                    f"{result['latency']['total_ms']['p95']:.2f}",
                    f"{result['latency']['stage1_ms']['mean']:.2f}",
                    f"{result['latency']['stage2_ms']['mean']:.2f}"
                ])
                writer.writerow(row)
        
        print(f"Saved summary CSV to {csv_path}")
        print(f"\nAll results saved to: {run_dir}")
    
    def print_comparison_table(self):
        """Print formatted comparison table."""
        print("\n=== Model Comparison ===\n")
        
        # Header
        print(f"{'Model':<30} {'MRR':<8} {'mAP':<8} {'HR@5':<8} {'Latency (P95)':<15}")
        print("-" * 75)
        
        # Rows
        for result in self.results:
            name = result['model_name'][:28]
            mrr = result['metrics']['MRR']
            map_score = result['metrics']['mAP']
            hr5 = result['metrics']['HR@5']
            lat_p95 = result['latency']['total_ms']['p95']
            
            print(f"{name:<30} {mrr:<8.4f} {map_score:<8.4f} {hr5:<8.4f} {lat_p95:<15.2f}")
        
        print()


def create_default_config(output_path: str = "experiments/default_config.yaml"):
    """Create default experiment configuration file."""
    config = {
        'experiment_name': 'LAR-RAG Baseline and Ablation Study',
        'ground_truth_file': 'data/evaluation/ground_truth.json',
        'output_dir': 'results',
        'evaluation': {
            'k_values': [1, 3, 5, 10]
        },
        'models': [
            {
                'name': 'BM25 Only',
                'stage1_method': 'bm25',
                'enable_temporal': False,
                'enable_topology': False
            },
            {
                'name': 'Dense Only',
                'stage1_method': 'dense',
                'enable_temporal': False,
                'enable_topology': False
            },
            {
                'name': 'Hybrid (BM25+Dense)',
                'stage1_method': 'hybrid',
                'enable_temporal': False,
                'enable_topology': False
            },
            {
                'name': 'LAR-RAG (Temporal Only)',
                'stage1_method': 'hybrid',
                'alpha': 0.7,
                'beta': 0.3,
                'gamma': 0.0,
                'lambda_decay': 0.001,
                'lambda_post': 0.01,
                'enable_temporal': True,
                'enable_topology': False
            },
            {
                'name': 'LAR-RAG (Topology Only)',
                'stage1_method': 'hybrid',
                'alpha': 0.7,
                'beta': 0.0,
                'gamma': 0.3,
                'lambda_g': 0.5,
                'enable_temporal': False,
                'enable_topology': True,
                'topology_mode': 'synthetic'
            },
            {
                'name': 'LAR-RAG (Full)',
                'stage1_method': 'hybrid',
                'alpha': 0.5,
                'beta': 0.3,
                'gamma': 0.2,
                'lambda_decay': 0.001,
                'lambda_post': 0.01,
                'lambda_g': 0.5,
                'enable_temporal': True,
                'enable_topology': True,
                'topology_mode': 'synthetic'
            }
        ]
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created default config at {output_path}")
    return output_path


if __name__ == "__main__":
    # Create default config
    config_path = create_default_config()
    
    # Run experiments
    runner = ExperimentRunner(str(config_path))
    runner.run_experiments()
    runner.print_comparison_table()
    runner.export_results()
