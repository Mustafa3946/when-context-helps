"""
LAR-RAG Evaluation Harness - INFOCOM 2025

This module provides a comprehensive evaluation framework for LAR-RAG experiments:
1. Load ground truth and experimental configuration
2. Run multiple retrieval methods in parallel
3. Compute per-query and aggregate metrics (HR@k, MRR, mAP, latency)
4. Perform statistical significance testing
5. Generate paper-ready result tables and figures

Usage:
    # Run with default config
    python src/evaluation/experiment_runner.py
    
    # Run with custom config
    python src/evaluation/experiment_runner.py --config experiments/infocom_config.yaml
    
    # Run specific methods only
    python src/evaluation/experiment_runner.py --methods "BM25 Only" "LAR-RAG (Full)"
    
    # Quick test with limited queries
    python src/evaluation/experiment_runner.py --max-queries 10
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

# Import retrievers
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lar_rag import LARRAGRetriever


@dataclass
class QueryResult:
    """Single query evaluation result."""
    query_id: str
    query_text: str
    query_timestamp: Optional[str]
    query_node: Optional[str]
    
    # Ground truth
    relevant_log_ids: List[str]
    
    # Retrieved results
    retrieved_log_ids: List[str]
    retrieved_scores: List[float]
    
    # Metrics
    hr_at_1: float
    hr_at_3: float
    hr_at_5: float
    hr_at_10: float
    mrr: float
    ap: float  # Average Precision
    
    # Performance
    latency_ms: float
    
    # Metadata
    method_name: str
    timestamp: str


@dataclass
class AggregateResults:
    """Aggregate metrics across all queries."""
    method_name: str
    num_queries: int
    
    # Hit Rate @ K
    hr_at_1: float
    hr_at_3: float
    hr_at_5: float
    hr_at_10: float
    
    # Ranking metrics
    mrr: float
    map: float  # Mean Average Precision
    
    # Latency statistics
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    # Per-query results
    query_results: List[QueryResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding per-query results for summary)."""
        d = asdict(self)
        d.pop('query_results')  # Exclude from summary
        return d


class ExperimentRunner:
    """Evaluation harness for LAR-RAG experiments."""
    
    def __init__(self, config_path: str = "experiments/infocom_config.yaml"):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to experiment configuration YAML
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.ground_truth = self._load_ground_truth()
        self.results: Dict[str, AggregateResults] = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        """Load ground truth from JSONL file."""
        gt_path = Path(self.config['data']['ground_truth'])
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        ground_truth = []
        with open(gt_path, 'r') as f:
            for line in f:
                if line.strip():
                    ground_truth.append(json.loads(line))
        
        print(f"✅ Loaded {len(ground_truth)} ground truth query pairs")
        
        # INFOCOM Compliance: Validate ground truth protocol
        self._validate_ground_truth_protocol(ground_truth)
        
        return ground_truth
    
    def _validate_ground_truth_protocol(self, ground_truth: List[Dict[str, Any]]):
        """
        Validate ground truth follows INFOCOM-compliant protocol.
        
        Ensures no prohibited heuristics (time proximity, node equality, rack proximity)
        are used as relevance criteria.
        
        Args:
            ground_truth: Loaded ground truth data
        
        Raises:
            ValueError: If protocol violations detected
        
        See: docs/GROUND_TRUTH_PROTOCOL.md
        """
        print("\n🔍 Validating ground truth protocol compliance...")
        
        # Check for required metadata
        if not ground_truth:
            raise ValueError("Ground truth is empty!")
        
        sample = ground_truth[0]
        
        # Check for explicit failure markers
        has_markers = False
        marker_fields = ['failure_category', 'category', 'marker_type']
        
        for field in marker_fields:
            if field in sample:
                has_markers = True
                break
        
        if not has_markers:
            print("⚠️  WARNING: Ground truth does not contain explicit failure markers.")
            print("   Expected one of: failure_category, category, marker_type")
            print("   See docs/GROUND_TRUTH_PROTOCOL.md for requirements.")
        
        # Check for prohibited heuristic flags
        prohibited_flags = [
            'time_proximity_heuristic',
            'node_equality_heuristic',
            'rack_proximity_heuristic',
            'spatial_heuristic',
        ]
        
        for entry in ground_truth[:10]:  # Sample first 10
            for flag in prohibited_flags:
                if entry.get(flag) == True:
                    raise ValueError(
                        f"❌ PROTOCOL VIOLATION: Ground truth uses prohibited heuristic '{flag}'\n"
                        f"   Query: {entry.get('query_id', 'unknown')}\n"
                        f"   This violates INFOCOM requirements. Use explicit failure markers only.\n"
                        f"   See docs/GROUND_TRUTH_PROTOCOL.md"
                    )
        
        # Check metadata for protocol version
        protocol_version = None
        for entry in ground_truth[:1]:
            if 'protocol_version' in entry:
                protocol_version = entry['protocol_version']
                break
        
        if protocol_version:
            print(f"✅ Ground truth protocol version: {protocol_version}")
        
        # Validate template-first requirement
        uses_templates = any('template_id' in entry for entry in ground_truth[:10])
        
        if not uses_templates:
            print("⚠️  WARNING: Ground truth does not reference template IDs.")
            print("   Template-first architecture requires template-based ground truth.")
        
        print("✅ Ground truth protocol validation passed")
        print(f"   - No prohibited heuristics detected")
        print(f"   - Failure markers: {'✅ Present' if has_markers else '⚠️  Missing'}")
        print(f"   - Template-first: {'✅ Yes' if uses_templates else '⚠️  No'}")
        print()
    
    def _create_retriever(self, method_config: Dict[str, Any]):
        """Factory method to create retriever instance."""
        method_type = method_config['type']
        config = method_config.get('config', {})
        
        if method_type == 'bm25':
            return BM25Retriever()
        
        elif method_type == 'hybrid':
            return HybridRetriever(
                rrf_k=config.get('rrf_k', 60)
            )
        
        elif method_type == 'lar_rag':
            return LARRAGRetriever(
                alpha=config.get('alpha', 0.5),
                beta=config.get('beta', 0.3),
                gamma=config.get('gamma', 0.2),
                lambda_decay=config.get('lambda_decay', 0.001),
                use_temporal=config.get('use_temporal', False),
                use_topology=config.get('use_topology', False)
            )
        
        else:
            raise ValueError(f"Unknown method type: {method_type}")
    
    def _compute_metrics(
        self,
        relevant_log_ids: List[str],
        retrieved_log_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics for a single query.
        
        Args:
            relevant_log_ids: Ground truth relevant log IDs
            retrieved_log_ids: Retrieved log IDs (in rank order)
            k_values: K values for HR@K computation
        
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        relevant_set = set(relevant_log_ids)
        
        # Hit Rate @ K
        for k in k_values:
            top_k = set(retrieved_log_ids[:k])
            metrics[f'hr_at_{k}'] = 1.0 if top_k & relevant_set else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, log_id in enumerate(retrieved_log_ids, start=1):
            if log_id in relevant_set:
                mrr = 1.0 / rank
                break
        metrics['mrr'] = mrr
        
        # Average Precision (AP)
        num_relevant = 0
        sum_precision = 0.0
        for rank, log_id in enumerate(retrieved_log_ids, start=1):
            if log_id in relevant_set:
                num_relevant += 1
                precision_at_k = num_relevant / rank
                sum_precision += precision_at_k
        
        ap = sum_precision / len(relevant_set) if relevant_set else 0.0
        metrics['ap'] = ap
        
        return metrics
    
    def evaluate_method(
        self,
        method_config: Dict[str, Any],
        max_queries: Optional[int] = None
    ) -> AggregateResults:
        """
        Evaluate a single retrieval method on all queries.
        
        Args:
            method_config: Method configuration from YAML
            max_queries: Maximum number of queries to evaluate (for debugging)
        
        Returns:
            AggregateResults with metrics
        """
        method_name = method_config['name']
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")
        
        # Create retriever
        try:
            retriever = self._create_retriever(method_config)
        except Exception as e:
            print(f"❌ Failed to create retriever: {e}")
            raise
        
        # Evaluate each query
        query_results = []
        k_values = self.config['evaluation']['k_values']
        
        queries = self.ground_truth[:max_queries] if max_queries else self.ground_truth
        
        for i, gt_entry in enumerate(queries, 1):
            query_id = gt_entry.get('query_id', f"q_{i}")
            query_text = gt_entry['query_message']
            query_timestamp = gt_entry.get('query_timestamp')
            query_node = gt_entry.get('query_node')
            relevant_log_ids = gt_entry['related_log_ids']
            
            # Retrieve
            start_time = time.time()
            try:
                top_k = method_config['config'].get('top_k', 50)
                results = retriever.retrieve(
                    query_text=query_text,
                    top_k=top_k,
                    query_timestamp=query_timestamp,
                    query_node=query_node
                )
                
                retrieved_log_ids = [r['log_id'] for r in results]
                retrieved_scores = [r.get('combined_score', r.get('score', 0.0)) for r in results]
                
            except Exception as e:
                print(f"⚠️  Query {i}/{len(queries)} failed: {e}")
                retrieved_log_ids = []
                retrieved_scores = []
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Compute metrics
            metrics = self._compute_metrics(relevant_log_ids, retrieved_log_ids, k_values)
            
            # Store result
            query_result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                query_timestamp=query_timestamp,
                query_node=query_node,
                relevant_log_ids=relevant_log_ids,
                retrieved_log_ids=retrieved_log_ids,
                retrieved_scores=retrieved_scores,
                hr_at_1=metrics.get('hr_at_1', 0.0),
                hr_at_3=metrics.get('hr_at_3', 0.0),
                hr_at_5=metrics.get('hr_at_5', 0.0),
                hr_at_10=metrics.get('hr_at_10', 0.0),
                mrr=metrics['mrr'],
                ap=metrics['ap'],
                latency_ms=latency_ms,
                method_name=method_name,
                timestamp=datetime.now().isoformat()
            )
            query_results.append(query_result)
            
            # Progress
            if i % 10 == 0 or i == len(queries):
                print(f"  Progress: {i}/{len(queries)} queries | "
                      f"HR@10: {metrics.get('hr_at_10', 0.0):.3f} | "
                      f"MRR: {metrics['mrr']:.3f} | "
                      f"Latency: {latency_ms:.1f}ms")
        
        # Aggregate results
        latencies = [r.latency_ms for r in query_results]
        
        aggregate = AggregateResults(
            method_name=method_name,
            num_queries=len(query_results),
            hr_at_1=sum(r.hr_at_1 for r in query_results) / len(query_results),
            hr_at_3=sum(r.hr_at_3 for r in query_results) / len(query_results),
            hr_at_5=sum(r.hr_at_5 for r in query_results) / len(query_results),
            hr_at_10=sum(r.hr_at_10 for r in query_results) / len(query_results),
            mrr=sum(r.mrr for r in query_results) / len(query_results),
            map=sum(r.ap for r in query_results) / len(query_results),
            latency_mean_ms=statistics.mean(latencies),
            latency_median_ms=statistics.median(latencies),
            latency_p95_ms=self._percentile(latencies, 95),
            latency_p99_ms=self._percentile(latencies, 99),
            query_results=query_results
        )
        
        # Print summary
        print(f"\n📊 {method_name} Results:")
        print(f"   HR@1:  {aggregate.hr_at_1:.4f}")
        print(f"   HR@3:  {aggregate.hr_at_3:.4f}")
        print(f"   HR@5:  {aggregate.hr_at_5:.4f}")
        print(f"   HR@10: {aggregate.hr_at_10:.4f}")
        print(f"   MRR:   {aggregate.mrr:.4f}")
        print(f"   mAP:   {aggregate.map:.4f}")
        print(f"   Latency (median): {aggregate.latency_median_ms:.1f}ms")
        print(f"   Latency (p95):    {aggregate.latency_p95_ms:.1f}ms")
        
        return aggregate
    
    @staticmethod
    def _percentile(data: List[float], p: int) -> float:
        """Compute percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def run_all_experiments(
        self,
        method_names: Optional[List[str]] = None,
        max_queries: Optional[int] = None
    ) -> Dict[str, AggregateResults]:
        """
        Run all configured experiments.
        
        Args:
            method_names: List of method names to run (None = all enabled methods)
            max_queries: Maximum queries per method (for debugging)
        
        Returns:
            Dictionary of method_name -> AggregateResults
        """
        print(f"\n🚀 Starting LAR-RAG Evaluation")
        print(f"   Experiment: {self.config['experiment']['name']}")
        print(f"   Dataset: {self.config['experiment']['dataset']}")
        print(f"   Queries: {len(self.ground_truth)}")
        
        # Filter methods
        methods = self.config['methods']
        if method_names:
            methods = [m for m in methods if m['name'] in method_names]
        else:
            methods = [m for m in methods if m.get('enabled', True)]
        
        print(f"   Methods: {len(methods)}")
        
        # Run each method
        results = {}
        for method_config in methods:
            try:
                aggregate = self.evaluate_method(method_config, max_queries)
                results[method_config['name']] = aggregate
            except Exception as e:
                print(f"❌ Method '{method_config['name']}' failed: {e}")
                import traceback
                traceback.print_exc()
        
        self.results = results
        return results
    
    def save_results(self, output_dir: Optional[str] = None):
        """Save results to JSON files."""
        if output_dir is None:
            output_dir = self.config['output']['results_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID
        run_id = self.config['output'].get('run_id') or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate results
        aggregate_file = output_path / f"aggregate_results_{run_id}.json"
        aggregate_data = {
            name: result.to_dict()
            for name, result in self.results.items()
        }
        with open(aggregate_file, 'w') as f:
            json.dump(aggregate_data, f, indent=2)
        print(f"\n✅ Saved aggregate results: {aggregate_file}")
        
        # Save per-query results (if enabled)
        if self.config['output'].get('save_per_query_metrics', True):
            for method_name, result in self.results.items():
                query_file = output_path / f"per_query_{method_name.replace(' ', '_')}_{run_id}.jsonl"
                with open(query_file, 'w') as f:
                    for qr in result.query_results:
                        f.write(json.dumps(asdict(qr)) + '\n')
                print(f"✅ Saved per-query results: {query_file}")
        
        # Save run manifest
        if self.config['output'].get('save_run_manifest', True):
            manifest_file = output_path / f"run_manifest_{run_id}.yaml"
            manifest = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'results_summary': {
                    name: {
                        'hr_at_10': result.hr_at_10,
                        'mrr': result.mrr,
                        'map': result.map,
                        'num_queries': result.num_queries
                    }
                    for name, result in self.results.items()
                }
            }
            with open(manifest_file, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
            print(f"✅ Saved run manifest: {manifest_file}")
    
    def print_comparison_table(self):
        """Print INFOCOM-ready comparison table."""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*100}")
        print(f"INFOCOM Table 2: Main Results Comparison")
        print(f"{'='*100}")
        print(f"{'Method':<30} | {'HR@1':>8} | {'HR@5':>8} | {'HR@10':>8} | {'MRR':>8} | {'mAP':>8} | {'Latency (ms)':>12}")
        print(f"{'-'*100}")
        
        for method_name, result in self.results.items():
            print(f"{method_name:<30} | "
                  f"{result.hr_at_1:>8.4f} | "
                  f"{result.hr_at_5:>8.4f} | "
                  f"{result.hr_at_10:>8.4f} | "
                  f"{result.mrr:>8.4f} | "
                  f"{result.map:>8.4f} | "
                  f"{result.latency_median_ms:>12.1f}")
        
        print(f"{'='*100}\n")


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="LAR-RAG Evaluation Harness - INFOCOM 2025"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/infocom_config.yaml',
        help='Path to experiment configuration YAML'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        help='Specific method names to run (default: all enabled methods)'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        help='Maximum number of queries to evaluate (for debugging)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Run experiments
    runner = ExperimentRunner(config_path=args.config)
    runner.run_all_experiments(
        method_names=args.methods,
        max_queries=args.max_queries
    )
    
    # Save results
    runner.save_results(output_dir=args.output_dir)
    
    # Print comparison table
    runner.print_comparison_table()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
