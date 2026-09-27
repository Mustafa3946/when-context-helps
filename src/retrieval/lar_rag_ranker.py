"""
LAR-RAG Ranker: Latency-Aware Retrieval with Temporal and Topology Kernels.

Implements the core LAR-RAG contribution:
    S(q,c) = α * sim(q,c) + β * g_time(Δt) + γ * g_topo(dG)

Where:
- sim(q,c): Cosine similarity between query and candidate
- g_time(Δt): Temporal kernel (asymmetric exponential decay)
- g_topo(dG): Topology kernel (graph distance decay)

INFOCOM requirements:
- Per-query score normalization
- Configurable ablations (no-time, no-topology)
- Deterministic scoring
"""

import logging
import json
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

from src.retrieval.base import Retriever, RetrievalResult
from src.retrieval.hybrid import HybridRetriever
from src.utils.graph_utils import SystemTopologyGraph

logger = logging.getLogger(__name__)


class LARRAGRanker(Retriever):
    """
    LAR-RAG ranker with semantic, temporal, and topology scoring.
    
    Architecture:
    1. Stage 1: Retrieve candidates using hybrid retrieval
    2. Stage 2: Rerank with LAR-RAG scoring:
       - Semantic similarity (from Stage 1)
       - Temporal decay kernel
       - Topology proximity kernel
    3. Per-query normalization of components
    4. Weighted combination
    """
    
    def __init__(
        self,
        templates_file: str,
        index_dir: str,
        template_map_file: str,
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
        normalize_scores: bool = True
    ):
        """
        Initialize LAR-RAG ranker.
        
        Args:
            templates_file: Path to templates JSONL
            index_dir: FAISS index directory
            template_map_file: Template-to-log mapping file
            embedding_model: OpenAI embedding model
            alpha: Weight for semantic similarity (default: 0.4)
            beta: Weight for temporal decay (default: 0.3)
            gamma: Weight for topology proximity (default: 0.3)
            lambda_pre: Pre-incident decay rate (default: 0.0001)
            lambda_post: Post-incident decay rate (default: 0.001, higher penalty)
            lambda_g: Topology decay rate (default: 0.5)
            use_temporal: Enable temporal scoring
            use_topology: Enable topology scoring
            topology_mode: Graph mode ("static", "log-inferred", "synthetic")
            topology_file: Path to topology JSON
            stage1_k: Number of candidates from Stage 1
            normalize_scores: Enable per-query normalization (INFOCOM requirement)
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
        
        # Load template map for temporal/topology features
        logger.info(f"Loading template map from {template_map_file}...")
        self.template_to_logs = defaultdict(list)
        
        with open(template_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    mapping = json.loads(line)
                    self.template_to_logs[mapping['template_id']].append({
                        'log_id': mapping['log_id'],
                        'timestamp': mapping['timestamp'],
                        'node': mapping['node']
                    })
        
        logger.info(f"Loaded mappings for {len(self.template_to_logs)} templates")
        
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
            template_map_file=template_map_file,
            embedding_model=embedding_model,
            expand_to_logs=False,
            stage1_k=stage1_k
        )
        
        logger.info("✅ LAR-RAG ranker ready")
    
    def g_time(self, delta_t: float) -> float:
        """
        Temporal kernel with asymmetric exponential decay.
        
        Pre-incident logs (delta_t >= 0): slower decay
        Post-incident logs (delta_t < 0): faster decay (higher penalty)
        
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
        Topology kernel with exponential decay on graph distance.
        
        Args:
            distance: Graph hop distance
            
        Returns:
            Topology score in [0, 1]
        """
        if distance == 0:
            return 1.0
        elif distance < 0 or distance == 999:  # Disconnected
            return 0.0
        else:
            return math.exp(-self.lambda_g * distance)
    
    def _get_template_features(
        self,
        template_id: str,
        incident_time: Optional[float],
        incident_node: Optional[str]
    ) -> Tuple[float, int]:
        """
        Extract temporal and topology features for a template.
        
        Strategy: Use the log occurrence closest to incident_time
        (preferring pre-incident occurrences as proximal causes).
        
        Args:
            template_id: Template identifier
            incident_time: Incident timestamp
            incident_node: Incident node
            
        Returns:
            (best_delta_t, best_graph_distance)
        """
        logs = self.template_to_logs.get(template_id, [])
        
        if not logs:
            # No occurrences, use worst-case values
            return (float('inf'), 999)
        
        best_delta_t = float('inf')
        best_distance = 999
        
        for log in logs:
            # Temporal feature
            if incident_time is not None:
                delta_t = incident_time - log['timestamp']
                
                # Prefer pre-incident occurrences
                if delta_t >= 0:
                    if delta_t < abs(best_delta_t):
                        best_delta_t = delta_t
                else:
                    # Post-incident, only use if no pre-incident found
                    if best_delta_t == float('inf') or abs(delta_t) < abs(best_delta_t):
                        best_delta_t = delta_t
            
            # Topology feature
            if incident_node is not None and self.topology_graph:
                distance = self.topology_graph.distance(incident_node, log['node'])
                if distance >= 0 and distance < best_distance:
                    best_distance = distance
        
        return (best_delta_t, best_distance)
    
    def retrieve(
        self,
        query_text: str,
        incident_time: Optional[float] = None,
        incident_node: Optional[str] = None,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve using LAR-RAG scoring.
        
        Args:
            query_text: Query text
            incident_time: Incident timestamp (Unix epoch seconds)
            incident_node: Incident node identifier
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        # Stage 1: Retrieve candidates
        logger.debug(f"Stage 1: Hybrid retrieval (k={self.stage1_k})")
        candidates = self.hybrid_retriever.retrieve(
            query_text=query_text,
            top_k=self.stage1_k
        )
        
        if not candidates:
            logger.warning("No candidates from Stage 1")
            return []
        
        # Stage 2: LAR-RAG scoring
        logger.debug(f"Stage 2: LAR-RAG scoring ({len(candidates)} candidates)")
        
        scored_results = []
        
        for candidate in candidates:
            # Semantic score (from Stage 1)
            semantic_score = candidate.debug.get('dense_score', 0.0)
            
            # Temporal and topology scores
            temporal_score = 0.0
            topology_score = 0.0
            
            if self.use_temporal or self.use_topology:
                delta_t, graph_dist = self._get_template_features(
                    candidate.template_id,
                    incident_time,
                    incident_node
                )
                
                if self.use_temporal and incident_time is not None:
                    temporal_score = self.g_time(delta_t)
                
                if self.use_topology and incident_node is not None:
                    topology_score = self.g_topo(graph_dist)
            
            scored_results.append({
                'template_id': candidate.template_id,
                'semantic_score': semantic_score,
                'temporal_score': temporal_score,
                'topology_score': topology_score,
                'original_rank': candidate.rank
            })
        
        # Normalize scores per-query
        if self.normalize_scores:
            scored_results = self._normalize_scores(scored_results)
        
        # Compute final scores
        final_results = []
        
        for result in scored_results:
            final_score = (
                self.alpha * result['semantic_score'] +
                self.beta * result['temporal_score'] +
                self.gamma * result['topology_score']
            )
            
            final_results.append(RetrievalResult(
                template_id=result['template_id'],
                log_id=None,
                score=final_score,
                rank=0,  # Will be set after sorting
                method=self.get_name(),
                debug={
                    'semantic_score': result['semantic_score'],
                    'temporal_score': result['temporal_score'],
                    'topology_score': result['topology_score'],
                    'semantic_norm': result.get('semantic_norm', result['semantic_score']),
                    'temporal_norm': result.get('temporal_norm', result['temporal_score']),
                    'topology_norm': result.get('topology_norm', result['topology_score']),
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'gamma': self.gamma
                }
            ))
        
        # Sort by final score
        final_results.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(final_results[:top_k], start=1):
            result.rank = rank
        
        return final_results[:top_k]
    
    def _normalize_scores(self, scored_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply per-query min-max normalization.
        
        INFOCOM requirement: Normalize each component to [0,1] before weighted sum.
        
        Args:
            scored_results: List of result dictionaries
            
        Returns:
            Same list with normalized scores added
        """
        if not scored_results:
            return scored_results
        
        # Extract score arrays
        semantic_scores = [r['semantic_score'] for r in scored_results]
        temporal_scores = [r['temporal_score'] for r in scored_results]
        topology_scores = [r['topology_score'] for r in scored_results]
        
        # Normalize semantic
        min_sem = min(semantic_scores)
        max_sem = max(semantic_scores)
        range_sem = max_sem - min_sem if max_sem > min_sem else 1.0
        
        # Normalize temporal (if used)
        min_temp = min(temporal_scores) if self.use_temporal else 0.0
        max_temp = max(temporal_scores) if self.use_temporal else 0.0
        range_temp = max_temp - min_temp if max_temp > min_temp else 1.0
        
        # Normalize topology (if used)
        min_topo = min(topology_scores) if self.use_topology else 0.0
        max_topo = max(topology_scores) if self.use_topology else 0.0
        range_topo = max_topo - min_topo if max_topo > min_topo else 1.0
        
        # Apply normalization
        for r in scored_results:
            r['semantic_norm'] = (r['semantic_score'] - min_sem) / range_sem
            
            if self.use_temporal:
                r['temporal_norm'] = (r['temporal_score'] - min_temp) / range_temp
            else:
                r['temporal_norm'] = 0.0
            
            if self.use_topology:
                r['topology_norm'] = (r['topology_score'] - min_topo) / range_topo
            else:
                r['topology_norm'] = 0.0
            
            # Use normalized scores for final combination
            r['semantic_score'] = r['semantic_norm']
            r['temporal_score'] = r['temporal_norm']
            r['topology_score'] = r['topology_norm']
        
        return scored_results
    
    def get_name(self) -> str:
        """Get retriever name."""
        if not self.use_temporal and not self.use_topology:
            return "LAR-RAG (Semantic Only)"
        elif not self.use_temporal:
            return "LAR-RAG (No Time)"
        elif not self.use_topology:
            return "LAR-RAG (No Topology)"
        else:
            return "LAR-RAG (Full)"
    
    def get_config(self) -> dict:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "lambda_pre": self.lambda_pre,
            "lambda_post": self.lambda_post,
            "lambda_g": self.lambda_g,
            "use_temporal": self.use_temporal,
            "use_topology": self.use_topology,
            "stage1_k": self.stage1_k,
            "normalize_scores": self.normalize_scores,
            "hybrid_config": self.hybrid_retriever.get_config()
        })
        return config
