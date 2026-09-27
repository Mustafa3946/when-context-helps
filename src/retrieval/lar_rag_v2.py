"""
LAR-RAG Ranker v2: Lazy Loading with Occurrence Store

CRITICAL CHANGES:
- NO preloading of template_to_logs mapping
- Lazy on-demand queries to OccurrenceStore
- Bounded memory footprint O(K) instead of O(templates × logs)

INFOCOM REQUIREMENT:
- Disk-backed occurrence queries
- Efficient temporal/topology feature extraction
- Deterministic scoring with per-query normalization
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from src.retrieval.base import Retriever, RetrievalResult
from src.retrieval.hybrid import HybridRetriever
from src.storage.occurrence_store import OccurrenceStore
from src.utils.graph_utils import SystemTopologyGraph

logger = logging.getLogger(__name__)


class LARRAGRankerV2(Retriever):
    """
    LAR-RAG ranker with lazy occurrence loading.
    
    Architecture:
    1. Stage 1: Retrieve top-K candidates (BM25/Dense/Hybrid)
    2. Stage 2: For each candidate template:
       - Query OccurrenceStore on-demand
       - Compute temporal features (min Δt)
       - Compute topology features (min graph distance)
    3. Score: S(q,c) = α·sim + β·g_time + γ·g_topo
    4. Per-query normalization
    
    Memory: O(stage1_k) instead of O(all templates × logs)
    """
    
    def __init__(
        self,
        templates_file: str,
        index_dir: str,
        occurrence_store_path: str,
        embedding_model: str = "text-embedding-3-small",
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        lambda_pre: float = 0.0001,
        lambda_post: float = 0.001,
        lambda_g: float = 0.5,
        use_temporal: bool = True,
        use_topology: bool = True,
        topology_mode: str = "synthetic",
        topology_file: Optional[str] = None,
        stage1_k: int = 200,
        normalize_scores: bool = True,
        local_only: bool = True,
        embeddings_manifest: Optional[str] = None,
        embeddings_shards_dir: Optional[str] = None,
        max_cached_shards: int = 16
    ):
        """
        Initialize LAR-RAG ranker v2.
        
        Args:
            templates_file: Path to templates JSONL
            index_dir: FAISS index directory
            occurrence_store_path: Path to DuckDB occurrence store
            embedding_model: OpenAI embedding model
            alpha: Weight for semantic similarity
            beta: Weight for temporal decay
            gamma: Weight for topology proximity
            lambda_pre: Pre-incident decay rate
            lambda_post: Post-incident decay rate (higher penalty)
            lambda_g: Topology decay rate
            use_temporal: Enable temporal scoring
            use_topology: Enable topology scoring
            topology_mode: Graph mode
            topology_file: Path to topology JSON
            stage1_k: Number of candidates from Stage 1
            normalize_scores: Enable per-query normalization
            local_only: Use offline mode (no Azure API calls)
            embeddings_manifest: Path to manifest.json (required if local_only=True)
            embeddings_shards_dir: Path to embeddings shards (required if local_only=True)
        """
        # Validate weights
        if not np.isclose(alpha + beta + gamma, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {alpha + beta + gamma}")
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_pre = lambda_pre
        self.lambda_post = lambda_post
        self.lambda_g = lambda_g
        self.use_temporal = use_temporal
        self.use_topology = use_topology
        self.stage1_k = stage1_k
        self.normalize_scores = normalize_scores
        self.local_only = local_only
        
        # Initialize occurrence store
        logger.info(f"Connecting to occurrence store: {occurrence_store_path}")
        self.occurrence_store = OccurrenceStore(occurrence_store_path)
        
        # Log statistics
        stats = self.occurrence_store.get_stats()
        logger.info(f"✅ Store stats: {stats['total_occurrences']:,} occurrences, {stats['unique_templates']:,} templates")
        
        # Initialize topology graph
        self.topology_graph = None
        if self.use_topology:
            logger.info(f"Initializing topology graph (mode={topology_mode})...")
            self.topology_graph = SystemTopologyGraph(
                mode=topology_mode,
                graph_file=topology_file
            )
        
        # Initialize hybrid retriever for Stage 1
        logger.info("Initializing hybrid retriever (Stage 1)...")
        self.hybrid_retriever = HybridRetriever(
            templates_file=templates_file,
            index_dir=index_dir,
            template_map_file=None,  # Not needed for Stage 1
            embedding_model=embedding_model,
            expand_to_logs=False,
            stage1_k=stage1_k,
            local_only=local_only,
            embeddings_manifest=embeddings_manifest,
            embeddings_shards_dir=embeddings_shards_dir,
            max_cached_shards=max_cached_shards
        )
        
        logger.info("✅ LAR-RAG ranker v2 ready (lazy loading)")
    
    def g_time(self, delta_t: float) -> float:
        """
        Temporal kernel with asymmetric exponential decay.
        
        Args:
            delta_t: Time difference (incident_time - log_time) in seconds
            
        Returns:
            Temporal score in [0, 1]
        """
        if delta_t >= 0:
            # Pre-incident: slower decay
            return math.exp(-self.lambda_pre * delta_t)
        else:
            # Post-incident: faster decay
            return math.exp(-self.lambda_post * abs(delta_t))
    
    def g_topo(self, distance: int) -> float:
        """
        Topology kernel with exponential decay.
        
        Args:
            distance: Graph hop distance
            
        Returns:
            Topology score in [0, 1]
        """
        if distance == 0:
            return 1.0
        elif distance < 0 or distance == 999:
            return 0.0
        else:
            return math.exp(-self.lambda_g * distance)
    
    def _extract_features_lazy(
        self,
        template_id: str,
        incident_time: Optional[float],
        incident_node: Optional[str]
    ) -> Tuple[float, int]:
        """
        Extract temporal and topology features using lazy occurrence queries.
        
        Strategy: Query occurrence store for closest occurrence.
        
        Args:
            template_id: Template identifier
            incident_time: Incident timestamp (Unix epoch)
            incident_node: Incident node
            
        Returns:
            (best_delta_t, best_graph_distance)
        """
        # Query occurrence store
        if incident_time is not None:
            occurrence = self.occurrence_store.get_closest_occurrence(
                template_id,
                incident_time,
                prefer_before=True
            )
        else:
            # No incident time, get any occurrence
            occurrences = self.occurrence_store.get_occurrences(template_id, limit=1)
            occurrence = occurrences[0] if occurrences else None
        
        if not occurrence:
            # No occurrences found
            return (float('inf'), 999)
        
        # Temporal feature
        delta_t = float('inf')
        if incident_time is not None:
            delta_t = incident_time - occurrence['timestamp']
        
        # Topology feature
        graph_distance = 999
        if incident_node is not None and self.topology_graph:
            graph_distance = self.topology_graph.compute_graph_distance(
                incident_node,
                occurrence['node_id']
            )
            if graph_distance < 0:
                graph_distance = 999
        
        return (delta_t, graph_distance)
    
    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_template_id: Optional[str] = None,
        incident_time: Optional[float] = None,
        incident_node: Optional[str] = None,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve using LAR-RAG scoring with lazy loading.
        
        Args:
            query_text: Query text (optional in local_only mode)
            query_template_id: Query template ID (required for local_only mode)
            incident_time: Incident timestamp (Unix epoch)
            incident_node: Incident node identifier
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        # Validate inputs for local_only mode
        if self.local_only and not query_template_id:
            raise ValueError(
                "local_only=True requires query_template_id for template-first retrieval.\n"
                "Provide incident_template_id from ground truth incidents."
            )
        
        # Stage 1: Retrieve candidates
        logger.debug(f"Stage 1: Hybrid retrieval (k={self.stage1_k})")
        candidates = self.hybrid_retriever.retrieve(
            query_text=query_text,
            query_template_id=query_template_id,
            top_k=self.stage1_k,
            **kwargs
        )
        
        if not candidates:
            logger.warning("No candidates from Stage 1")
            return []
        
        # Stage 2: Lazy feature extraction and LAR-RAG scoring
        logger.debug(f"Stage 2: LAR-RAG scoring (lazy loading)")
        
        scored_results = []
        
        for candidate in candidates:
            template_id = candidate.template_id
            
            # Lazy query for features
            delta_t, graph_distance = self._extract_features_lazy(
                template_id,
                incident_time,
                incident_node
            )
            
            # Compute kernel scores
            time_score = self.g_time(delta_t) if self.use_temporal and delta_t != float('inf') else 0.0
            topo_score = self.g_topo(graph_distance) if self.use_topology else 0.0
            
            # Store raw scores
            scored_results.append({
                'template_id': template_id,
                'similarity': candidate.score,  # From Stage 1
                'time_score': time_score,
                'topo_score': topo_score,
                'delta_t': delta_t,
                'graph_distance': graph_distance
            })
        
        # Per-query normalization (INFOCOM requirement)
        if self.normalize_scores and scored_results:
            logger.debug("Normalizing scores per-query")
            
            # Extract component arrays
            sims = np.array([r['similarity'] for r in scored_results])
            times = np.array([r['time_score'] for r in scored_results])
            topos = np.array([r['topo_score'] for r in scored_results])
            
            # Min-max normalization
            def normalize(arr):
                if arr.max() > arr.min():
                    return (arr - arr.min()) / (arr.max() - arr.min())
                else:
                    return np.ones_like(arr)
            
            sims_norm = normalize(sims)
            times_norm = normalize(times) if self.use_temporal else np.zeros_like(times)
            topos_norm = normalize(topos) if self.use_topology else np.zeros_like(topos)
            
            # Compute final scores
            for i, result in enumerate(scored_results):
                final_score = (
                    self.alpha * sims_norm[i] +
                    self.beta * times_norm[i] +
                    self.gamma * topos_norm[i]
                )
                result['final_score'] = final_score
                result['sim_norm'] = sims_norm[i]
                result['time_norm'] = times_norm[i]
                result['topo_norm'] = topos_norm[i]
        else:
            # No normalization (use raw weighted sum)
            for result in scored_results:
                final_score = (
                    self.alpha * result['similarity'] +
                    self.beta * result['time_score'] +
                    self.gamma * result['topo_score']
                )
                result['final_score'] = final_score
        
        # Sort by final score
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Convert to RetrievalResult
        final_results = []
        for result in scored_results[:top_k]:
            final_results.append(
                RetrievalResult(
                    template_id=result['template_id'],
                    score=result['final_score'],
                    debug={
                        'similarity': result['similarity'],
                        'time_score': result['time_score'],
                        'topo_score': result['topo_score'],
                        'delta_t': result['delta_t'],
                        'graph_distance': result['graph_distance'],
                        'sim_norm': result.get('sim_norm', result['similarity']),
                        'time_norm': result.get('time_norm', result['time_score']),
                        'topo_norm': result.get('topo_norm', result['topo_score'])
                    }
                )
            )
        
        logger.debug(f"✅ Returned {len(final_results)} results")
        return final_results
    
    def get_name(self) -> str:
        """Return retriever name for reporting."""
        return "LAR-RAG-v2"
    
    def close(self):
        """Close resources."""
        if self.occurrence_store:
            self.occurrence_store.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
