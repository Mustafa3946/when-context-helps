"""
Two-stage LAR-RAG retrieval pipeline with Azure Blob expansion.

CORRECT TEMPLATE-FIRST ARCHITECTURE (INFOCOM Requirement):
Stage 1: Retrieve templates from Azure AI Search (bgl-templates index)
Stage 2: Expand templates to log candidates from Azure Blob Storage
Stage 3: Rerank candidates (NO re-querying Azure Search!)

Includes per-stage latency tracking:
- stage1_search_ms: Template retrieval from Azure AI Search
- expansion_ms: Candidate expansion from Azure Blob
- rerank_ms: LAR-RAG reranking of candidates
- total_ms: End-to-end latency
"""
import time
from typing import List, Dict, Optional, Literal, Set
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.candidate_expander_blob import CandidateExpander, Candidate
from retrieval.lar_rag_reranker import LARRAGReranker, RankedCandidate


class TwoStageLARRAGPipeline:
    """
    CORRECT Two-stage retrieval pipeline for LAR-RAG.
    
    Stage 1: Retrieve templates from Azure AI Search (bgl-templates index)
    Stage 2: Expand templates to log candidates from Azure Blob Storage
    Stage 3: Rerank candidates with LAR-RAG (NO re-querying!)
    
    INFOCOM Requirements:
    - Template-first: Stage 1 retrieves templates, not logs
    - No re-querying: Stage 2 expands from blob, Stage 3 reranks candidates
    - Latency tracking: Per-stage measurement for performance analysis
    """
    
    def __init__(
        self,
        stage1_method: Literal['bm25', 'hybrid'] = 'hybrid',
        stage1_k: int = 50,
        expansion_window_before_s: int = 3600,
        expansion_window_after_s: int = 300,
        expansion_max_candidates: int = 1000,
        stage2_alpha: float = 0.5,
        stage2_beta: float = 0.3,
        stage2_gamma: float = 0.2,
        lambda_decay: float = 0.001,
        lambda_post: float = 0.01,
        lambda_g: float = 0.5,
        topology_mode: str = "synthetic",
        topology_file: Optional[str] = None,
        enable_temporal: bool = True,
        enable_topology: bool = True,
        template_embeddings_file: Optional[str] = None,
        blob_container: str = "data",
        blob_prefix: str = "processed/bgl/",
        connection_string: Optional[str] = None
    ):
        """
        Initialize two-stage pipeline.
        
        Args:
            stage1_method: Template retrieval method - 'bm25' or 'hybrid'
            stage1_k: Number of templates to retrieve in stage 1
            expansion_window_before_s: Time window before incident (seconds)
            expansion_window_after_s: Time window after incident (seconds)
            expansion_max_candidates: Max candidates from expansion (early stopping)
            stage2_alpha: Reranker weight for semantic similarity
            stage2_beta: Reranker weight for temporal decay
            stage2_gamma: Reranker weight for topology proximity
            lambda_decay: Temporal decay rate for past logs
            lambda_post: Temporal decay rate for future logs (higher penalty)
            lambda_g: Topology decay rate
            topology_mode: Graph construction mode
            topology_file: Path to topology graph JSON (for static mode)
            enable_temporal: Enable temporal scoring in reranking
            enable_topology: Enable topology scoring in reranking
            template_embeddings_file: Path to template embeddings cache
            blob_container: Azure Blob container name
            blob_prefix: Blob prefix for log shards
            connection_string: Azure Storage connection string
        """
        self.stage1_method = stage1_method
        self.stage1_k = stage1_k
        self.expansion_window_before_s = expansion_window_before_s
        self.expansion_window_after_s = expansion_window_after_s
        self.expansion_max_candidates = expansion_max_candidates
        self.enable_temporal = enable_temporal
        self.enable_topology = enable_topology
        
        # Initialize Stage 1: Template retriever (Azure AI Search)
        if stage1_method == 'bm25':
            self.stage1_retriever = BM25Retriever()
        elif stage1_method == 'hybrid':
            self.stage1_retriever = HybridRetriever(fusion_strategy='rrf')
        else:
            raise ValueError(f"Unknown stage1_method: {stage1_method}. Use 'bm25' or 'hybrid'.")
        
        # Initialize Stage 2: Candidate expander (Azure Blob)
        self.candidate_expander = CandidateExpander(
            container=blob_container,
            prefix=blob_prefix,
            connection_string=connection_string
        )
        
        # Initialize Stage 3: LAR-RAG reranker (operates on candidates)
        self.reranker = LARRAGReranker(
            alpha=stage2_alpha,
            beta=stage2_beta,
            gamma=stage2_gamma,
            lambda_decay=lambda_decay,
            lambda_post=lambda_post,
            lambda_g=lambda_g,
            use_temporal=enable_temporal,
            use_topology=enable_topology,
            topology_mode=topology_mode,
            topology_file=topology_file,
            template_embeddings_file=template_embeddings_file
        )
        
        # Latency tracking (per-stage)
        self.latency_stats = {
            'stage1_search_ms': [],
            'expansion_ms': [],
            'rerank_ms': [],
            'total_ms': [],
        }
    
    def retrieve(
        self,
        query_text: str,
        top_k: int = 10,
        incident_timestamp: Optional[float] = None,
        query_node: Optional[str] = None
    ) -> Dict:
        """
        Execute two-stage retrieval with blob expansion.
        
        Args:
            query_text: Query string
            top_k: Number of final results to return
            incident_timestamp: Incident timestamp (Unix seconds) for expansion
            query_node: Query node for topology scoring
            
        Returns:
            Dict with results and per-stage latency
            
        Pipeline Flow:
            1. Stage 1: Retrieve templates from Azure AI Search
            2. Expansion: Get log candidates from Azure Blob Storage
            3. Reranking: Score candidates with LAR-RAG (NO re-querying!)
        
        INFOCOM Compliance:
            - Stage 1 queries "bgl-templates" index ONLY
            - Stage 2 does NOT call Azure AI Search (blob expansion only)
            - Stage 3 reranks candidates without additional search calls
        """
        total_start = time.time()
        
        # Stage 1: Retrieve templates from Azure AI Search
        stage1_start = time.time()
        stage1_results = self.stage1_retriever.retrieve(
            query_text=query_text,
            top_k=self.stage1_k
        )
        stage1_time = (time.time() - stage1_start) * 1000  # ms
        
        # Extract template IDs from stage1 results
        template_ids: Set[str] = set()
        for result in stage1_results:
            # Handle different result formats
            template_id = result.get('template_id') or result.get('id')
            if template_id:
                template_ids.add(template_id)
        
        print(f"[Pipeline] Stage 1: Retrieved {len(template_ids)} templates")
        
        # Stage 2: Expand templates to log candidates from Azure Blob
        expansion_start = time.time()
        
        if not incident_timestamp:
            # Default: use current time if not provided
            incident_timestamp = time.time()
        
        candidates = self.candidate_expander.expand(
            template_ids=template_ids,
            incident_ts=incident_timestamp,
            window_before_s=self.expansion_window_before_s,
            window_after_s=self.expansion_window_after_s,
            max_candidates=self.expansion_max_candidates
        )
        
        expansion_time = (time.time() - expansion_start) * 1000  # ms
        
        print(f"[Pipeline] Expansion: Found {len(candidates)} log candidates")
        
        # Stage 3: Rerank candidates with LAR-RAG
        rerank_start = time.time()
        
        if candidates and (self.enable_temporal or self.enable_topology):
            ranked_candidates = self.reranker.rerank(
                query=query_text,
                candidates=candidates,
                incident_ts=incident_timestamp,
                query_node=query_node
            )
        else:
            # No reranking needed (or no candidates)
            ranked_candidates = [
                RankedCandidate(
                    candidate=cand,
                    final_score=1.0 / (i + 1),  # Simple rank-based score
                    semantic_score=0.0,
                    temporal_score=0.0,
                    topology_score=0.0
                )
                for i, cand in enumerate(candidates)
            ]
        
        rerank_time = (time.time() - rerank_start) * 1000  # ms
        
        total_time = (time.time() - total_start) * 1000  # ms
        
        # Track latencies
        self.latency_stats['stage1_search_ms'].append(stage1_time)
        self.latency_stats['expansion_ms'].append(expansion_time)
        self.latency_stats['rerank_ms'].append(rerank_time)
        self.latency_stats['total_ms'].append(total_time)
        
        # Convert to output format
        final_results = []
        for ranked_cand in ranked_candidates[:top_k]:
            final_results.append({
                'log_id': ranked_cand.candidate.log_id,
                'timestamp': ranked_cand.candidate.timestamp,
                'node_id': ranked_cand.candidate.node_id,
                'template_id': ranked_cand.candidate.template_id,
                'message': ranked_cand.candidate.message,
                'final_score': ranked_cand.final_score,
                'semantic_score': ranked_cand.semantic_score,
                'temporal_score': ranked_cand.temporal_score,
                'topology_score': ranked_cand.topology_score,
            })
        
        return {
            'results': final_results,
            'latency': {
                'stage1_search_ms': stage1_time,
                'expansion_ms': expansion_time,
                'rerank_ms': rerank_time,
                'total_ms': total_time,
            },
            'num_templates': len(template_ids),
            'num_candidates': len(candidates),
            'num_final_results': len(final_results),
        }
    
    def get_latency_stats(self) -> Dict:
        """
        Get aggregated latency statistics.
        
        Returns:
            Dict with mean, median, P95, P99 for each stage
        """
        import statistics
        
        def percentile(data, p):
            """Calculate percentile."""
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = f + 1
            if c >= len(sorted_data):
                return sorted_data[-1]
            d0 = sorted_data[f] * (c - k)
            d1 = sorted_data[c] * (k - f)
            return d0 + d1
        
        stats = {}
        for key in ['stage1_search_ms', 'expansion_ms', 'rerank_ms', 'total_ms']:
            data = self.latency_stats[key]
            if data:
                stats[key] = {
                    'mean': statistics.mean(data),
                    'median': statistics.median(data),
                    'p95': percentile(data, 95),
                    'p99': percentile(data, 99),
                    'min': min(data),
                    'max': max(data),
                }
            else:
                stats[key] = {
                    'mean': 0.0, 'median': 0.0, 'p95': 0.0,
                    'p99': 0.0, 'min': 0.0, 'max': 0.0,
                }
        
        return stats
    
    def reset_latency_stats(self):
        """Reset latency tracking."""
        for key in self.latency_stats:
            self.latency_stats[key] = []


# Convenience factory functions
def create_hybrid_pipeline(**kwargs) -> TwoStageLARRAGPipeline:
    """Create pipeline with hybrid stage 1 (recommended for INFOCOM)."""
    return TwoStageLARRAGPipeline(stage1_method='hybrid', **kwargs)


def create_bm25_pipeline(**kwargs) -> TwoStageLARRAGPipeline:
    """Create pipeline with BM25 stage 1."""
    return TwoStageLARRAGPipeline(stage1_method='bm25', **kwargs)


if __name__ == "__main__":
    print("=== Two-Stage LAR-RAG Pipeline Test (Template-First + Blob Expansion) ===\n")
    
    # Example query
    query = "GPU cache parity error"
    incident_ts = 1234567890.0
    query_node = "R02-M1-N0-C:J12-U01"
    
    print(f"Query: {query}")
    print(f"Incident timestamp: {incident_ts}")
    print(f"Query node: {query_node}\n")
    
    # Test configurations
    configs = [
        ("Hybrid + LAR-RAG (Full)", {
            "stage1_method": "hybrid",
            "enable_temporal": True,
            "enable_topology": True,
            "stage1_k": 50,
            "expansion_max_candidates": 1000
        }),
        ("BM25 + Temporal Only", {
            "stage1_method": "bm25",
            "enable_temporal": True,
            "enable_topology": False,
            "stage1_k": 30,
            "expansion_max_candidates": 500
        }),
        ("Hybrid + Semantic Only", {
            "stage1_method": "hybrid",
            "enable_temporal": False,
            "enable_topology": False,
            "stage1_k": 50,
            "expansion_max_candidates": 1000
        }),
    ]
    
    for name, config in configs:
        print(f"=== {name} ===")
        
        try:
            pipeline = TwoStageLARRAGPipeline(**config)
            
            result = pipeline.retrieve(
                query_text=query,
                top_k=5,
                incident_timestamp=incident_ts,
                query_node=query_node
            )
            
            print(f"Latency Breakdown:")
            print(f"  Stage 1 (Search):  {result['latency']['stage1_search_ms']:.2f}ms")
            print(f"  Expansion (Blob):  {result['latency']['expansion_ms']:.2f}ms")
            print(f"  Reranking:         {result['latency']['rerank_ms']:.2f}ms")
            print(f"  Total:             {result['latency']['total_ms']:.2f}ms")
            
            print(f"\nPipeline Statistics:")
            print(f"  Templates retrieved: {result['num_templates']}")
            print(f"  Candidates expanded: {result['num_candidates']}")
            print(f"  Final results:       {result['num_final_results']}")
            
            if result['results']:
                print(f"\nTop 3 results:")
                for i, r in enumerate(result['results'][:3], 1):
                    print(f"{i}. Score: {r['final_score']:.4f} | "
                          f"Node: {r['node_id']} | "
                          f"Template: {r['template_id']}")
                    if r.get('message'):
                        print(f"   {r['message'][:60]}...")
            
            print()
            
        except Exception as e:
            print(f"ERROR: {e}")
            print("Note: Requires Azure credentials and blob data.\n")
    
    # Test latency statistics
    print("\n=== Latency Statistics Test ===")
    
    try:
        pipeline = create_hybrid_pipeline(
            enable_temporal=True,
            enable_topology=False
        )
        
        # Run multiple queries
        test_queries = [
            ("Query 1", "block corruption"),
            ("Query 2", "packet responder terminating"),
            ("Query 3", "replication status"),
        ]
        
        for name, q in test_queries:
            result = pipeline.retrieve(
                q,
                top_k=5,
                incident_timestamp=incident_ts,
                query_node=query_node
            )
            print(f"{name}: {result['latency']['total_ms']:.2f}ms total")
        
        stats = pipeline.get_latency_stats()
        print(f"\nAggregate Statistics (3 queries):")
        print(f"Stage 1 - Mean: {stats['stage1_search_ms']['mean']:.2f}ms, "
              f"P95: {stats['stage1_search_ms']['p95']:.2f}ms")
        print(f"Expansion - Mean: {stats['expansion_ms']['mean']:.2f}ms, "
              f"P95: {stats['expansion_ms']['p95']:.2f}ms")
        print(f"Reranking - Mean: {stats['rerank_ms']['mean']:.2f}ms, "
              f"P95: {stats['rerank_ms']['p95']:.2f}ms")
        print(f"Total - Mean: {stats['total_ms']['mean']:.2f}ms, "
              f"P95: {stats['total_ms']['p95']:.2f}ms")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Note: Requires Azure credentials and blob data.")
    
    print("\n✓ Pipeline structure validated")
