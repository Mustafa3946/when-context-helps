"""
LAR-RAG Evaluation Harness - INFOCOM 2025

Comprehensive evaluation framework for LAR-RAG experiments.
Supports multiple retrieval methods, metrics, and statistical testing.
"""

import json
import time
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.base import Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lar_rag_ranker import LARRAGRanker

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Single query evaluation result."""
    query_id: str
    query_text: str
    query_timestamp: Optional[float]
    query_node: Optional[str]
    
    # Ground truth
    relevant_template_ids: List[str]
    
    # Retrieved results
    retrieved_template_ids: List[str]
    retrieved_scores: List[float]
    
    # Metrics
    hr_at_1: float
    hr_at_3: float
    hr_at_5: float
    hr_at_10: float
    hr_at_20: float
    mrr: float
    ap: float
    
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
    hr_at_20: float
    
    # Ranking metrics
    mrr: float
    map: float
    
    # Latency statistics
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ExperimentRunner:
    """Evaluation harness for LAR-RAG experiments."""
    
    def __init__(
        self,
        config_path: str = "config/bgl_eval.yaml",
        max_queries: Optional[int] = None,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to experiment configuration YAML
            max_queries: Limit number of queries (for testing)
            output_dir: Override output directory
            run_id: Custom run ID
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.max_queries = max_queries or self.config['experiment'].get('max_queries')
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            results_base = Path(self.config['output']['results_dir'])
            if run_id:
                self.output_dir = results_base / run_id
            elif self.config['output'].get('run_id'):
                self.output_dir = results_base / self.config['output']['run_id']
            else:
                # Auto-generate run ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = results_base / f"run_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Results storage
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
        
        logger.info(f"✅ Loaded {len(ground_truth)} ground truth queries")
        
        return ground_truth
    
    def _create_retriever(self, method_config: Dict[str, Any]) -> Retriever:
        """
        Factory method to create retriever instance.
        
        Args:
            method_config: Method configuration from YAML
            
        Returns:
            Retriever instance
        """
        method_type = method_config['type']
        params = method_config.get('params', {})
        
        templates_file = self.config['data']['templates']
        template_map_file = self.config['data']['template_map']
        index_dir = self.config['data']['faiss_index']
        embedding_model = self.config['embedding']['model']
        
        if method_type == 'bm25':
            return BM25Retriever(
                templates_file=templates_file,
                template_map_file=template_map_file,
                k1=params.get('k1', 1.2),
                b=params.get('b', 0.75),
                expand_to_logs=False
            )
        
        elif method_type == 'dense':
            return DenseRetriever(
                index_dir=index_dir,
                embedding_model=embedding_model,
                template_map_file=template_map_file,
                expand_to_logs=False
            )
        
        elif method_type == 'hybrid':
            return HybridRetriever(
                templates_file=templates_file,
                index_dir=index_dir,
                template_map_file=template_map_file,
                embedding_model=embedding_model,
                expand_to_logs=False,
                stage1_k=params.get('stage1_k', 200),
                alpha=params.get('alpha', 0.5),
                beta=params.get('beta', 0.5)
            )
        
        elif method_type == 'lar_rag':
            return LARRAGRanker(
                templates_file=templates_file,
                index_dir=index_dir,
                template_map_file=template_map_file,
                embedding_model=embedding_model,
                alpha=params.get('alpha', 0.4),
                beta=params.get('beta', 0.3),
                gamma=params.get('gamma', 0.3),
                lambda_pre=params.get('lambda_pre', 0.0001),
                lambda_post=params.get('lambda_post', 0.001),
                lambda_g=params.get('lambda_g', 0.5),
                use_temporal=params.get('use_temporal', True),
                use_topology=params.get('use_topology', True),
                topology_mode=params.get('topology_mode', 'synthetic'),
                topology_file=self.config['data'].get('topology_graph'),
                stage1_k=params.get('stage1_k', 200),
                normalize_scores=True
            )
        
        else:
            raise ValueError(f"Unknown method type: {method_type}")
    
    def _compute_metrics(
        self,
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics for a single query.
        
        Args:
            relevant_ids: Ground truth relevant template IDs
            retrieved_ids: Retrieved template IDs (in rank order)
            k_values: K values for HR@K computation
        
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        relevant_set = set(relevant_ids)
        
        # Hit Rate @ K
        for k in k_values:
            top_k = set(retrieved_ids[:k])
            metrics[f'hr_at_{k}'] = 1.0 if top_k & relevant_set else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, tid in enumerate(retrieved_ids, start=1):
            if tid in relevant_set:
                mrr = 1.0 / rank
                break
        metrics['mrr'] = mrr
        
        # Average Precision (AP)
        num_relevant = 0
        sum_precision = 0.0
        for rank, tid in enumerate(retrieved_ids, start=1):
            if tid in relevant_set:
                num_relevant += 1
                precision_at_k = num_relevant / rank
                sum_precision += precision_at_k
        
        ap = sum_precision / len(relevant_set) if relevant_set else 0.0
        metrics['ap'] = ap
        
        return metrics
    
    def evaluate_method(self, method_config: Dict[str, Any]) -> AggregateResults:
        """
        Evaluate a single retrieval method.
        
        Args:
            method_config: Method configuration
            
        Returns:
            Aggregate results
        """
        method_name = method_config['name']
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {method_name}")
        logger.info(f"{'='*60}")
        
        # Create retriever
        retriever = self._create_retriever(method_config)
        logger.info(f"Retriever config: {retriever.get_config()}")
        
        # Evaluate on all queries
        query_results = []
        queries_to_process = self.ground_truth[:self.max_queries] if self.max_queries else self.ground_truth
        
        top_k = self.config['experiment']['top_k']
        k_values = self.config['metrics']['hit_rate_k']
        
        for i, query_data in enumerate(queries_to_process, start=1):
            query_id = query_data.get('query_id', f'q_{i}')
            query_text = query_data['query_text']
            query_time = query_data.get('incident_time')
            query_node = query_data.get('incident_node')
            relevant_ids = query_data.get('relevant_template_ids', [])
            
            # Retrieve
            start_time = time.time()
            results = retriever.retrieve(
                query_text=query_text,
                incident_time=query_time,
                incident_node=query_node,
                top_k=top_k
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract IDs and scores
            retrieved_ids = [r.template_id for r in results]
            retrieved_scores = [r.score for r in results]
            
            # Compute metrics
            metrics = self._compute_metrics(relevant_ids, retrieved_ids, k_values)
            
            # Create query result
            query_result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                query_timestamp=query_time,
                query_node=query_node,
                relevant_template_ids=relevant_ids,
                retrieved_template_ids=retrieved_ids,
                retrieved_scores=retrieved_scores,
                hr_at_1=metrics['hr_at_1'],
                hr_at_3=metrics['hr_at_3'],
                hr_at_5=metrics['hr_at_5'],
                hr_at_10=metrics['hr_at_10'],
                hr_at_20=metrics['hr_at_20'],
                mrr=metrics['mrr'],
                ap=metrics['ap'],
                latency_ms=latency_ms,
                method_name=method_name,
                timestamp=datetime.now().isoformat()
            )
            
            query_results.append(query_result)
            
            if (i % 10) == 0:
                logger.info(f"Processed {i}/{len(queries_to_process)} queries...")
        
        # Aggregate results
        num_queries = len(query_results)
        
        aggregate = AggregateResults(
            method_name=method_name,
            num_queries=num_queries,
            hr_at_1=sum(r.hr_at_1 for r in query_results) / num_queries,
            hr_at_3=sum(r.hr_at_3 for r in query_results) / num_queries,
            hr_at_5=sum(r.hr_at_5 for r in query_results) / num_queries,
            hr_at_10=sum(r.hr_at_10 for r in query_results) / num_queries,
            hr_at_20=sum(r.hr_at_20 for r in query_results) / num_queries,
            mrr=sum(r.mrr for r in query_results) / num_queries,
            map=sum(r.ap for r in query_results) / num_queries,
            latency_mean_ms=statistics.mean(r.latency_ms for r in query_results),
            latency_median_ms=statistics.median(r.latency_ms for r in query_results),
            latency_p95_ms=statistics.quantiles([r.latency_ms for r in query_results], n=20)[18],
            latency_p99_ms=statistics.quantiles([r.latency_ms for r in query_results], n=100)[98]
        )
        
        # Save per-query results
        if self.config['output']['save_per_query']:
            per_query_file = self.output_dir / f"{method_name.replace(' ', '_')}_per_query.jsonl"
            with open(per_query_file, 'w') as f:
                for result in query_results:
                    f.write(json.dumps(asdict(result)) + '\n')
            logger.info(f"Saved per-query results: {per_query_file}")
        
        logger.info(f"\n{method_name} Results:")
        logger.info(f"  MRR: {aggregate.mrr:.4f}")
        logger.info(f"  HR@1: {aggregate.hr_at_1:.4f}")
        logger.info(f"  HR@5: {aggregate.hr_at_5:.4f}")
        logger.info(f"  MAP: {aggregate.map:.4f}")
        logger.info(f"  Latency (median): {aggregate.latency_median_ms:.2f} ms")
        
        return aggregate
    
    def run_all_methods(
        self,
        method_filter: Optional[List[str]] = None
    ) -> Dict[str, AggregateResults]:
        """
        Run all configured methods.
        
        Args:
            method_filter: Optional list of method names to run
            
        Returns:
            Dictionary of method_name -> AggregateResults
        """
        methods = self.config['methods']
        
        # Filter methods if specified
        if method_filter:
            methods = [m for m in methods if m['name'] in method_filter]
        
        if not methods:
            raise ValueError("No methods to evaluate!")
        
        logger.info(f"Evaluating {len(methods)} methods on {len(self.ground_truth)} queries")
        
        results = {}
        
        for method_config in methods:
            try:
                aggregate = self.evaluate_method(method_config)
                results[method_config['name']] = aggregate
            except Exception as e:
                logger.error(f"Error evaluating {method_config['name']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary
        if self.config['output']['save_summary']:
            self._save_summary(results)
        
        # Save config snapshot
        if self.config['output']['save_config_snapshot']:
            self._save_config_snapshot()
        
        return results
    
    def _save_summary(self, results: Dict[str, AggregateResults]):
        """Save summary CSV file."""
        summary_file = self.output_dir / "summary.csv"
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'method_name', 'num_queries',
                'hr_at_1', 'hr_at_3', 'hr_at_5', 'hr_at_10', 'hr_at_20',
                'mrr', 'map',
                'latency_mean_ms', 'latency_median_ms', 'latency_p95_ms', 'latency_p99_ms'
            ])
            
            writer.writeheader()
            for result in results.values():
                writer.writerow(result.to_dict())
        
        logger.info(f"Saved summary: {summary_file}")
    
    def _save_config_snapshot(self):
        """Save configuration snapshot for reproducibility."""
        snapshot_file = self.output_dir / "config_snapshot.json"
        
        with open(snapshot_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved config snapshot: {snapshot_file}")
