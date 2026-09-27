"""
LAR-RAG Evaluation Harness v2 - INFOCOM 2026

Canonical evaluation framework using v2 components:
- OccurrenceStore (DuckDB, disk-backed)
- LAR-RAG v2 ranker (lazy, bounded memory)
- Parquet-sharded embeddings
- Streaming ground truth

ACADEMIC ARTIFACT - REPRODUCIBLE EXPERIMENTS
"""

import json
import time
import yaml
import logging
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import csv
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.config_validation import load_and_validate_config, compute_config_hash
from src.retrieval.base import Retriever, RetrievalResult
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.bm25_offline import OfflineBM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lar_rag_v2 import LARRAGRankerV2
from src.storage.occurrence_store import OccurrenceStore

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    method_name: str
    
    # Retrieval metrics
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float
    average_precision: float
    
    # Performance
    latency_ms: float
    
    # Retrieved results
    retrieved_template_ids: List[str]
    retrieved_scores: List[float]
    
    # Ground truth
    relevant_template_ids: List[str]
    relevance_grades: Dict[str, int]


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries."""
    method_name: str
    num_queries: int
    
    # Retrieval metrics (mean across queries)
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr_at_10: float
    ndcg_at_10: float
    map: float
    
    # Latency statistics
    latency_mean_ms: float
    latency_median_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


class ExperimentRunnerV2:
    """
    Canonical evaluation harness for LAR-RAG experiments.
    
    Enforces:
    - v2 configuration schema
    - Local-first retrievers only
    - Bounded memory (no full dataset loads)
    - Reproducible experiments (deterministic seeds, hashes)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to canonical v2 configuration YAML
        """
        self.config_path = Path(config_path)
        self.config = load_and_validate_config(str(self.config_path))
        self.config_hash = compute_config_hash(self.config)
        
        # Priority 4 (rev11): Log local_only mode status at startup
        local_only = self.config.get('run', {}).get('local_only', True)
        logger.debug(f"local_only mode: {local_only}")
        if local_only:
            import sys
            forbidden = ['openai', 'azure.core', 'httpx']
            loaded = [m for m in forbidden if m in sys.modules]
            logger.debug(f"Forbidden modules check at startup: {loaded if loaded else 'none loaded ✓'}")
        
        # Set random seed for reproducibility
        np.random.seed(self.config['run']['seed'])
        
        # Setup output directory
        self.output_dir = Path(self.config['run']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize occurrence store (shared across LAR-RAG methods)
        occurrence_db_path = self.config['data']['occurrence_store']['db_path']
        logger.info(f"Connecting to occurrence store: {occurrence_db_path}")
        self.occurrence_store = OccurrenceStore(occurrence_db_path)
        
        # Load ground truth
        self.incidents, self.qrels = self._load_ground_truth()
        
        # Apply train/test split
        self.test_incidents = self._apply_split()
        
        logger.info(f"Experiment initialized:")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Config hash: {self.config_hash[:16]}...")
        logger.info(f"  Seed: {self.config['run']['seed']}")
        logger.info(f"  Test incidents: {len(self.test_incidents)}")
        logger.info(f"  Output: {self.output_dir}")
    
    def _load_ground_truth(self) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Load ground truth from incidents and qrels files.
        
        Returns:
            (incidents, qrels_dict)
            where qrels_dict[incident_id] = {template_id: grade}
        """
        incidents_path = Path(self.config['ground_truth']['incidents_path'])
        qrels_path = Path(self.config['ground_truth']['qrels_path'])
        
        # Load incidents
        incidents = []
        with open(incidents_path, 'r') as f:
            for line in f:
                if line.strip():
                    incidents.append(json.loads(line))
        
        logger.info(f"Loaded {len(incidents)} incidents from {incidents_path}")
        
        # Load qrels
        qrels_dict = {}
        with open(qrels_path, 'r') as f:
            for line in f:
                if line.strip():
                    qrel = json.loads(line)
                    incident_id = qrel['incident_id']
                    qrels_dict[incident_id] = qrel.get('relevance_grade', {})
        
        logger.info(f"Loaded qrels for {len(qrels_dict)} incidents from {qrels_path}")
        
        return incidents, qrels_dict
    
    def _apply_split(self) -> List[Dict]:
        """
        Apply train/test split according to configuration.
        
        Returns:
            Test incidents
        """
        split_method = self.config['ground_truth']['split']['method']
        train_frac = self.config['ground_truth']['split']['train_frac']
        
        if split_method == 'time':
            # Sort by incident_time, split chronologically
            sorted_incidents = sorted(self.incidents, key=lambda x: x['incident_time'])
            split_idx = int(len(sorted_incidents) * train_frac)
            test_incidents = sorted_incidents[split_idx:]
            
            logger.info(f"Time-based split: {len(sorted_incidents[:split_idx])} train, {len(test_incidents)} test")
        
        elif split_method == 'random':
            # Shuffle and split (deterministic with seed)
            indices = list(range(len(self.incidents)))
            np.random.shuffle(indices)
            split_idx = int(len(indices) * train_frac)
            test_indices = indices[split_idx:]
            test_incidents = [self.incidents[i] for i in test_indices]
            
            logger.info(f"Random split: {split_idx} train, {len(test_incidents)} test")
        
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        return test_incidents
    
    def _create_retriever(self, method_name: str) -> Retriever:
        """
        Create retriever instance for method.
        
        Args:
            method_name: Method name from config
            
        Returns:
            Retriever instance
            
        Safety:
            Refuses to instantiate Azure-dependent retrievers in local_only mode
        """
        method_config = self.config['methods'][method_name]
        templates_path = self.config['data']['templates_path']
        local_only = self.config.get('run', {}).get('local_only', True)
        
        # PRIORITY 3 (rev10): Guard canonical local-only path - FAIL FAST
        # Prevent accidental Azure/OpenAI code paths in paper runs
        if local_only:
            # Sanity check: these modules should NOT be imported in local_only mode
            import sys
            forbidden_modules = ['openai', 'azure.core', 'httpx']
            loaded_forbidden = [m for m in forbidden_modules if m in sys.modules]
            if loaded_forbidden:
                raise RuntimeError(
                    f"❌ local_only=True but Azure/OpenAI modules are loaded: {loaded_forbidden}\n"
                    f"This indicates accidental use of Azure-dependent retrievers in paper runs.\n"
                    f"Review retriever instantiation and ensure all components use offline mode.\n"
                    f"For INFOCOM reproducibility, NO network calls are allowed during evaluation."
                )
        
        if method_name == 'bm25_baseline':
            bm25_index_dir = self.config['retrieval']['bm25']['index_dir']
            
            # Use offline BM25 for paper runs (disk-backed index)
            if local_only:
                bm25_index_path = Path(bm25_index_dir) / 'bm25_index.pkl'
                if not bm25_index_path.exists():
                    raise ValueError(
                        f"local_only=True requires prebuilt BM25 index at {bm25_index_path}\n"
                        f"Run BM25 index building step first."
                    )
                logger.info(f"Using offline BM25 (disk-backed index): {bm25_index_path}")
                return OfflineBM25Retriever(index_dir=bm25_index_dir)
            else:
                logger.warning("Building BM25 index in memory (not recommended for large datasets)")
                return BM25Retriever(
                    templates_file=templates_path,
                    template_map_file=None
                )
        
        elif method_name == 'dense_baseline':
            faiss_index_path = self.config['retrieval']['faiss']['index_path']
            faiss_dir = str(Path(faiss_index_path).parent)
            
            # Get embedding paths for offline mode
            embeddings_cfg = self.config['data'].get('embeddings', {})
            manifest_path = embeddings_cfg.get('manifest_path')
            shards_dir = embeddings_cfg.get('shards_dir')
            
            # Validate paths in local-only mode
            if local_only and (not manifest_path or not shards_dir):
                raise ValueError(
                    "local_only=True requires data.embeddings.manifest_path and data.embeddings.shards_dir in config"
                )
            
            return DenseRetriever(
                index_dir=faiss_dir,
                embedding_model='text-embedding-3-small',
                local_only=local_only,
                embeddings_manifest=manifest_path,
                embeddings_shards_dir=shards_dir
            )
        
        elif method_name == 'hybrid_baseline':
            faiss_dir = str(Path(self.config['retrieval']['faiss']['index_path']).parent)
            bm25_weight = method_config.get('bm25_weight', 0.3)
            dense_weight = method_config.get('dense_weight', 0.7)
            stage1_k = method_config.get('stage1_k', 1000)
            
            # Get embedding paths for offline mode
            embeddings_cfg = self.config['data'].get('embeddings', {})
            manifest_path = embeddings_cfg.get('manifest_path')
            shards_dir = embeddings_cfg.get('shards_dir')
            
            return HybridRetriever(
                templates_file=templates_path,
                index_dir=faiss_dir,
                template_map_file=None,
                embedding_model='text-embedding-3-small',
                expand_to_logs=False,
                stage1_k=stage1_k,
                alpha=bm25_weight,
                beta=dense_weight,
                local_only=local_only,
                embeddings_manifest=manifest_path,
                embeddings_shards_dir=shards_dir
            )
        
        elif method_name.startswith('lar_rag'):
            # LAR-RAG v2 (lazy loading)
            faiss_dir = str(Path(self.config['retrieval']['faiss']['index_path']).parent)
            occurrence_db_path = self.config['data']['occurrence_store']['db_path']
            
            # Get LAR-RAG parameters from config
            lar_cfg = self.config['lar_rag']
            
            # Method-specific overrides
            use_temporal = method_config.get('use_temporal', True)
            use_topology = method_config.get('use_topology', True)
            
            # Weights (allow method-level overrides)
            alpha = method_config.get('alpha', lar_cfg['alpha'])
            beta = method_config.get('beta', lar_cfg['beta'])
            gamma = method_config.get('gamma', lar_cfg['gamma'])
            
            # Normalize weights to sum to 1.0
            weight_sum = alpha + beta + gamma
            if weight_sum > 0:
                alpha /= weight_sum
                beta /= weight_sum
                gamma /= weight_sum
            
            # Tau parameters from kernel configs
            tau_before = lar_cfg['time_kernel']['tau_before']
            tau_after = lar_cfg['time_kernel']['tau_after']
            tau_topo = lar_cfg['topo_kernel']['tau']
            
            # Convert tau to lambda (λ = 1/tau)
            lambda_pre = 1.0 / tau_before if tau_before > 0 else 0.0001
            lambda_post = 1.0 / tau_after if tau_after > 0 else 0.001
            lambda_g = 1.0 / tau_topo if tau_topo > 0 else 0.5
            
            # Get embedding paths for offline mode
            embeddings_cfg = self.config['data'].get('embeddings', {})
            manifest_path = embeddings_cfg.get('manifest_path')
            shards_dir = embeddings_cfg.get('shards_dir')
            local_only = self.config.get('run', {}).get('local_only', True)
            
            return LARRAGRankerV2(
                templates_file=templates_path,
                index_dir=faiss_dir,
                occurrence_store_path=occurrence_db_path,
                embedding_model='text-embedding-3-small',
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                lambda_pre=lambda_pre,
                lambda_post=lambda_post,
                lambda_g=lambda_g,
                use_temporal=use_temporal,
                use_topology=use_topology,
                topology_mode='synthetic',
                stage1_k=lar_cfg['stage1']['candidates'],
                normalize_scores=True,
                local_only=local_only,
                embeddings_manifest=manifest_path,
                embeddings_shards_dir=shards_dir
            )
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def _compute_query_metrics(
        self,
        incident: Dict,
        results: List[RetrievalResult],
        method_name: str,
        latency_ms: float
    ) -> QueryMetrics:
        """
        Compute metrics for a single query.
        
        Args:
            incident: Incident dict
            results: Retrieved results
            method_name: Method name
            latency_ms: Query latency
            
        Returns:
            QueryMetrics object
        """
        incident_id = incident['incident_id']
        relevance_grades = self.qrels.get(incident_id, {})
        relevant_template_ids = list(relevance_grades.keys())
        
        # Extract retrieved template_ids and scores
        retrieved_template_ids = [r.template_id for r in results]
        retrieved_scores = [r.score for r in results]
        
        # Recall@K
        def recall_at_k(k):
            retrieved_k = set(retrieved_template_ids[:k])
            relevant = set(relevant_template_ids)
            if not relevant:
                return 0.0
            return len(retrieved_k & relevant) / len(relevant)
        
        recall_at_1 = recall_at_k(1)
        recall_at_5 = recall_at_k(5)
        recall_at_10 = recall_at_k(10)
        recall_at_20 = recall_at_k(20)
        
        # MRR
        mrr = 0.0
        for rank, template_id in enumerate(retrieved_template_ids, 1):
            if template_id in relevant_template_ids:
                mrr = 1.0 / rank
                break
        
        # NDCG@10
        ndcg_at_10 = self._compute_ndcg(retrieved_template_ids[:10], relevance_grades)
        
        # Average Precision
        ap = self._compute_average_precision(retrieved_template_ids, relevant_template_ids)
        
        return QueryMetrics(
            query_id=incident_id,
            method_name=method_name,
            recall_at_1=recall_at_1,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            recall_at_20=recall_at_20,
            mrr=mrr,
            ndcg_at_10=ndcg_at_10,
            average_precision=ap,
            latency_ms=latency_ms,
            retrieved_template_ids=retrieved_template_ids,
            retrieved_scores=retrieved_scores,
            relevant_template_ids=relevant_template_ids,
            relevance_grades=relevance_grades
        )
    
    def _compute_ndcg(self, retrieved: List[str], relevance_grades: Dict[str, int]) -> float:
        """Compute NDCG@K."""
        if not relevance_grades:
            return 0.0
        
        # DCG
        dcg = 0.0
        for rank, template_id in enumerate(retrieved, 1):
            rel = relevance_grades.get(template_id, 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)
        
        # IDCG (ideal DCG)
        ideal_rels = sorted(relevance_grades.values(), reverse=True)[:len(retrieved)]
        idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _compute_average_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """Compute Average Precision."""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        num_relevant = 0
        sum_precision = 0.0
        
        for rank, template_id in enumerate(retrieved, 1):
            if template_id in relevant_set:
                num_relevant += 1
                precision_at_k = num_relevant / rank
                sum_precision += precision_at_k
        
        return sum_precision / len(relevant_set) if relevant_set else 0.0
    
    def evaluate_method(self, method_name: str) -> Tuple[AggregateMetrics, List[QueryMetrics]]:
        """
        Evaluate a single retrieval method.
        
        Args:
            method_name: Method name from config
            
        Returns:
            (aggregate_metrics, per_query_metrics)
        """
        logger.info(f"="*80)
        logger.info(f"Evaluating: {method_name}")
        logger.info(f"="*80)
        
        # Create retriever
        retriever = self._create_retriever(method_name)
        
        # Evaluate on test set
        per_query_metrics = []
        top_k = self.config['run']['top_k']
        local_only = self.config.get('run', {}).get('local_only', True)
        
        for i, incident in enumerate(self.test_incidents, 1):
            logger.info(f"Query {i}/{len(self.test_incidents)}: {incident['incident_id']}")
            
            # Prepare query
            query_text = incident.get('incident_text', '')
            query_template_id = incident.get('incident_template_id')
            incident_time = incident.get('incident_time')
            incident_node = incident.get('incident_node')
            
            # Validate template_id in local_only mode
            if local_only and not query_template_id:
                raise ValueError(
                    f"local_only=True requires incident_template_id in ground truth incidents.\n"
                    f"Missing for incident: {incident['incident_id']}\n"
                    f"Rebuild ground truth with updated ground_truth_streaming.py"
                )
            
            # Retrieve
            start_time = time.time()
            results = retriever.retrieve(
                query_text=query_text,  # ALWAYS pass text (BM25/Hybrid need it)
                query_template_id=query_template_id,  # Dense uses this in local_only mode
                incident_time=incident_time,
                incident_node=incident_node,
                top_k=top_k
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Compute metrics
            metrics = self._compute_query_metrics(incident, results, method_name, latency_ms)
            per_query_metrics.append(metrics)
            
            logger.info(f"  Recall@10: {metrics.recall_at_10:.3f}, MRR: {metrics.mrr:.3f}, Latency: {latency_ms:.1f}ms")
        
        # Aggregate metrics
        aggregate = self._aggregate_metrics(method_name, per_query_metrics)
        
        logger.info(f"\n{method_name} Summary:")
        logger.info(f"  Recall@10: {aggregate.recall_at_10:.4f}")
        logger.info(f"  MRR@10: {aggregate.mrr_at_10:.4f}")
        logger.info(f"  NDCG@10: {aggregate.ndcg_at_10:.4f}")
        logger.info(f"  MAP: {aggregate.map:.4f}")
        logger.info(f"  Latency P50: {aggregate.latency_p50_ms:.1f}ms")
        logger.info(f"  Latency P95: {aggregate.latency_p95_ms:.1f}ms")
        
        return aggregate, per_query_metrics
    
    def _aggregate_metrics(self, method_name: str, per_query: List[QueryMetrics]) -> AggregateMetrics:
        """Aggregate per-query metrics."""
        latencies = [m.latency_ms for m in per_query]
        
        return AggregateMetrics(
            method_name=method_name,
            num_queries=len(per_query),
            recall_at_1=np.mean([m.recall_at_1 for m in per_query]),
            recall_at_5=np.mean([m.recall_at_5 for m in per_query]),
            recall_at_10=np.mean([m.recall_at_10 for m in per_query]),
            recall_at_20=np.mean([m.recall_at_20 for m in per_query]),
            mrr_at_10=np.mean([m.mrr for m in per_query]),
            ndcg_at_10=np.mean([m.ndcg_at_10 for m in per_query]),
            map=np.mean([m.average_precision for m in per_query]),
            latency_mean_ms=np.mean(latencies),
            latency_median_ms=np.median(latencies),
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99)
        )
    
    def run_all_methods(self) -> Dict[str, AggregateMetrics]:
        """
        Run all enabled methods.
        
        Returns:
            Dict mapping method_name to AggregateMetrics
        """
        results = {}
        all_per_query_metrics = {}
        
        for method_name, method_config in self.config['methods'].items():
            if not method_config.get('enabled', True):
                logger.info(f"Skipping disabled method: {method_name}")
                continue
            
            aggregate, per_query = self.evaluate_method(method_name)
            results[method_name] = aggregate
            all_per_query_metrics[method_name] = per_query
        
        # Save results
        self._save_results(results, all_per_query_metrics)
        
        # Save run metadata
        self._save_run_metadata()
        
        return results
    
    def _save_results(self, aggregate: Dict[str, AggregateMetrics], per_query: Dict[str, List[QueryMetrics]]):
        """Save experiment results."""
        # Save aggregate results (JSON)
        results_json = {
            method_name: asdict(metrics)
            for method_name, metrics in aggregate.items()
        }
        
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"Saved results to {json_path}")
        
        # Save aggregate results (CSV)
        csv_path = self.output_dir / 'results.csv'
        with open(csv_path, 'w', newline='') as f:
            if aggregate:
                writer = csv.DictWriter(f, fieldnames=list(asdict(list(aggregate.values())[0]).keys()))
                writer.writeheader()
                for metrics in aggregate.values():
                    writer.writerow(asdict(metrics))
        logger.info(f"Saved results to {csv_path}")
        
        # Save per-query results
        if self.config['evaluation']['save_per_query']:
            for method_name, metrics_list in per_query.items():
                per_query_path = self.output_dir / f'per_query_{method_name}.jsonl'
                with open(per_query_path, 'w') as f:
                    for metrics in metrics_list:
                        f.write(json.dumps(asdict(metrics)) + '\n')
                logger.info(f"Saved per-query results to {per_query_path}")
    
    def _save_run_metadata(self):
        """Save run metadata for reproducibility."""
        if not self.config['evaluation'].get('save_run_meta', True):
            return
        
        metadata = {
            'config_path': str(self.config_path),
            'config_hash': self.config_hash,
            'run_name': self.config['run']['name'],
            'seed': self.config['run']['seed'],
            'timestamp': datetime.now().isoformat(),
            'num_test_queries': len(self.test_incidents),
            'git_commit': self._get_git_commit(),
            'dataset_fingerprints': {
                'templates_path': str(self.config['data']['templates_path']),
                'occurrence_store': str(self.config['data']['occurrence_store']['db_path']),
                'faiss_index': str(self.config['retrieval']['faiss']['index_path'])
            }
        }
        
        meta_path = self.output_dir / 'run_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved run metadata to {meta_path}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'occurrence_store'):
            self.occurrence_store.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run LAR-RAG evaluation experiments (v2 canonical)"
    )
    parser.add_argument(
        '--config',
        default='config/bgl_canonical_v2.yaml',
        help='Path to configuration YAML'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        runner = ExperimentRunnerV2(args.config)
        runner.run_all_methods()
        runner.cleanup()
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
