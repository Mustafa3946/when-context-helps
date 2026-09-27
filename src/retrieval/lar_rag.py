"""
LAR-RAG retrieval with semantic, temporal, and topology scoring.

TEMPLATE-FIRST ARCHITECTURE (INFOCOM requirement):
- Stage 1: Retrieve templates from "bgl-templates" index, expand to log candidates
- Stage 2: Rerank candidate logs (NO re-querying Azure AI Search!)
- Score normalization: Per-query min-max normalization for fair comparison
"""
import os
import ssl
import json
import math
from datetime import datetime
from collections import defaultdict
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.graph_utils import SystemTopologyGraph, graph_distance, g_topo
from utils.node_normalization import normalize_node_id

load_dotenv()

# Disable SSL verification for corporate proxy
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"

class LARRAGRetriever:
    def __init__(self, template_map_file="data/processed/bgl/template_map.jsonl",
                 alpha=0.5, beta=0.3, gamma=0.2, lambda_decay=0.001, lambda_post=0.01, lambda_g=0.5,
                 use_temporal=False, use_topology=False, normalize_scores=True,
                 topology_mode="synthetic", topology_file=None):
        """Initialize LAR-RAG retriever with configurable parameters.
        
        TEMPLATE-FIRST ARCHITECTURE:
        - Stage 1: Query "bgl-templates" index, expand to log candidates
        - Stage 2: Rerank candidates using semantic + temporal + topology (NO re-querying!)
        
        Args:
            template_map_file: Path to template-to-log mapping file (JSONL)
            alpha: Weight for semantic similarity (default: 0.5)
            beta: Weight for temporal decay (default: 0.3)
            gamma: Weight for topology proximity (default: 0.2)
            lambda_decay: Decay rate for past logs (default: 0.001)
            lambda_post: Decay rate for future logs - higher penalty (default: 0.01)
            lambda_g: Decay rate for topology scoring (default: 0.5)
            use_temporal: Enable temporal decay scoring
            use_topology: Enable topology proximity scoring
            normalize_scores: Enable per-query min-max normalization (default: True, INFOCOM requirement)
            topology_mode: Graph construction mode - "static", "log-inferred", or "synthetic"
            topology_file: Path to JSON graph file (for static mode)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        self.lambda_post = lambda_post
        self.lambda_g = lambda_g
        self.use_temporal = use_temporal
        self.use_topology = use_topology
        self.normalize_scores = normalize_scores
        
        # Load template-to-log mapping (for Stage 1 expansion)
        print(f"[LAR-RAG] Loading template map from {template_map_file}...")
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
        print(f"[LAR-RAG] Loaded mappings for {len(self.template_to_logs)} templates")
        
        # Initialize topology graph if needed
        if self.use_topology:
            self.topology_graph = SystemTopologyGraph(mode=topology_mode, graph_file=topology_file)
        else:
            self.topology_graph = None
        
        # Lazy imports for Azure-only dependencies
        from openai import AzureOpenAI
        import httpx
        
        http_client = httpx.Client(verify=False)
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            http_client=http_client
        )
        
        # Connect to TEMPLATE index (NOT log-embeddings!)
        transport = RequestsTransport(connection_verify=False)
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="bgl-templates",  # ✅ TEMPLATE-FIRST
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
            transport=transport
        )
    
    def _normalize_scores(self, scored_results):
        """
        Apply per-query min-max normalization to make scores comparable.
        
        INFOCOM requirement: Scores must be normalized per query to enable
        fair comparison across different queries and ablation studies.
        
        Normalization: score_norm = (score - min) / (max - min)
        If max == min, set all scores to 1.0 (all equal)
        
        Args:
            scored_results: List of dicts with semantic_score, temporal_score, topology_score
            
        Returns:
            Same list with normalized scores (in-place modification)
        """
        if not scored_results:
            return scored_results
        
        # Extract scores
        semantic_scores = [r['semantic_score'] for r in scored_results]
        temporal_scores = [r['temporal_score'] for r in scored_results]
        topology_scores = [r['topology_score'] for r in scored_results]
        
        # Normalize semantic scores
        min_sem = min(semantic_scores)
        max_sem = max(semantic_scores)
        range_sem = max_sem - min_sem if max_sem > min_sem else 1.0
        
        # Normalize temporal scores (if used)
        min_temp = min(temporal_scores) if self.use_temporal else 0.0
        max_temp = max(temporal_scores) if self.use_temporal else 0.0
        range_temp = max_temp - min_temp if max_temp > min_temp else 1.0
        
        # Normalize topology scores (if used)
        min_topo = min(topology_scores) if self.use_topology else 0.0
        max_topo = max(topology_scores) if self.use_topology else 0.0
        range_topo = max_topo - min_topo if max_topo > min_topo else 1.0
        
        # Apply normalization
        for r in scored_results:
            r['semantic_score'] = (r['semantic_score'] - min_sem) / range_sem
            
            if self.use_temporal:
                r['temporal_score'] = (r['temporal_score'] - min_temp) / range_temp
            
            if self.use_topology:
                r['topology_score'] = (r['topology_score'] - min_topo) / range_topo
        
        return scored_results
    
    def _parse_timestamp(self, ts_str):
        """Parse timestamp to datetime."""
        try:
            return datetime.fromisoformat(ts_str)
        except:
            try:
                return datetime.strptime(ts_str, "%y%m%d %H%M%S")
            except:
                return None
    
    def _temporal_decay(self, query_time, log_time):
        """
        Calculate temporal decay with RCA semantics.
        
        Root-cause logs are expected BEFORE or NEAR the incident.
        Logs AFTER incident are penalized more heavily.
        
        Formula:
            Δt = incident_time - log_time
            g_time(Δt) = exp(-λ_t * Δt) if Δt >= 0 (log before incident)
            g_time(Δt) = exp(-λ_post * |Δt|) if Δt < 0 (log after incident)
        """
        if query_time is None or log_time is None:
            return 0.0
        
        # Δt = incident_time - log_time
        # Positive Δt means log happened before incident (expected for RCA)
        # Negative Δt means log happened after incident (less relevant)
        delta_t = (query_time - log_time).total_seconds()
        
        if delta_t >= 0:
            # Log before incident: normal decay
            return math.exp(-self.lambda_decay * delta_t)
        else:
            # Log after incident: stronger penalty
            return math.exp(-self.lambda_post * abs(delta_t))
    
    def _parse_node(self, node_str):
        """Extract and normalize node identifier from node string."""
        if not node_str:
            return None
        return normalize_node_id(node_str)
    
    def _topology_proximity(self, query_node, log_node):
        """Calculate topology proximity using real graph distance: g_topo(d) = exp(-λ_g * d)"""
        if query_node is None or log_node is None:
            return 0.0
        if self.topology_graph is None:
            return 0.0
        
        # Normalize node IDs
        norm_query = normalize_node_id(query_node)
        norm_log = normalize_node_id(log_node)
        
        if not norm_query or not norm_log:
            return 0.0
        
        # Compute graph distance (hops)
        dist = graph_distance(self.topology_graph.graph, norm_query, norm_log, disconnected_value=999)
        
        # Apply topology kernel
        return g_topo(dist, self.lambda_g)
    
    def retrieve(self, query_text, top_k=10, top_k_templates=50, max_logs_per_template=100,
                 query_timestamp=None, query_node=None):
        """Retrieve top-k logs using LAR-RAG two-stage scoring.
        
        TEMPLATE-FIRST TWO-STAGE ARCHITECTURE:
        
        Stage 1 (Fast): Template Retrieval & Expansion
        - Query "bgl-templates" index with vector search
        - Retrieve top_k_templates templates
        - Expand templates to log candidates using template_map.jsonl
        - Return candidate logs (NO Azure queries after this!)
        
        Stage 2 (Expensive): Multi-Factor Reranking
        - Accept pre-retrieved candidate logs from Stage 1
        - Compute semantic scores (from template similarity)
        - Compute temporal scores (time decay)
        - Compute topology scores (graph distance)
        - Normalize scores (per-query min-max)
        - Weighted fusion: α·semantic + β·temporal + γ·topology
        - Return top-K reranked logs
        
        Args:
            query_text: Query string
            top_k: Number of final results to return
            top_k_templates: Number of templates to retrieve in Stage 1 (default: 50)
            max_logs_per_template: Max logs to expand per template (default: 100)
            query_timestamp: Query timestamp (ISO format) for temporal scoring
            query_node: Query node identifier for topology scoring
        
        Returns:
            List of scored results with semantic, temporal, topology scores
        """
        print(f"\n{'='*60}")
        print(f"LAR-RAG Two-Stage Retrieval")
        print(f"{'='*60}")
        
        # ========== STAGE 1: Template Retrieval & Expansion ==========
        print(f"\n[Stage 1] Template Retrieval")
        print(f"  → Query: '{query_text}'")
        print(f"  → Retrieving top {top_k_templates} templates...")
        
        # Generate query embedding (ONCE - used for template search)
        response = self.openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Vector search on TEMPLATES (NOT logs!)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k_templates,
            fields="embedding"
        )
        template_results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["template_id", "template", "count"]
        )
        
        # Expand templates to log candidates
        print(f"  → Expanding templates to log candidates...")
        candidate_logs = []
        
        for template_result in template_results:
            template_id = template_result["template_id"]
            template_score = template_result["@search.score"]  # Semantic similarity from template
            
            # Get logs for this template
            if template_id in self.template_to_logs:
                logs = self.template_to_logs[template_id]
                
                # Limit logs per template to avoid explosion
                for log in logs[:max_logs_per_template]:
                    candidate_logs.append({
                        'log_id': log['log_id'],
                        'timestamp': log['timestamp'],
                        'node': log['node'],
                        'template_id': template_id,
                        'semantic_score': template_score  # Inherit from template
                    })
        
        print(f"  ✓ Stage 1 complete: {len(candidate_logs)} candidate logs from {top_k_templates} templates")
        
        # ========== STAGE 2: Multi-Factor Reranking ==========
        print(f"\n[Stage 2] Multi-Factor Reranking")
        print(f"  → Reranking {len(candidate_logs)} candidates...")
        
        # Parse query context
        query_time = self._parse_timestamp(query_timestamp) if query_timestamp and self.use_temporal else None
        q_node = self._parse_node(query_node) if query_node and self.use_topology else None
        
        if query_time:
            print(f"  → Temporal context: {query_timestamp}")
        if q_node:
            print(f"  → Topology context: Node {query_node}")
        
        # Compute scores for each candidate (NO Azure queries!)
        scored_results = []
        for log in candidate_logs:
            semantic_score = log['semantic_score']  # From template similarity
            temporal_score = 0.0
            topology_score = 0.0
            
            # Calculate temporal decay
            if self.use_temporal and query_time:
                log_time = self._parse_timestamp(log['timestamp'])
                temporal_score = self._temporal_decay(query_time, log_time)
            
            # Calculate topology proximity
            if self.use_topology and q_node:
                log_node = self._parse_node(log['node'])
                topology_score = self._topology_proximity(q_node, log_node)
            
            scored_results.append({
                "log_id": log['log_id'],
                "timestamp": log['timestamp'],
                "node": log['node'],
                "template_id": log['template_id'],
                "semantic_score": semantic_score,
                "temporal_score": temporal_score,
                "topology_score": topology_score
            })
        
        # Normalize scores (per-query min-max normalization)
        if self.normalize_scores and scored_results:
            print(f"  → Normalizing scores (min-max per query)...")
            scored_results = self._normalize_scores(scored_results)
        
        # Compute combined scores with weighted fusion
        for r in scored_results:
            if self.use_temporal and self.use_topology:
                combined_score = (self.alpha * r['semantic_score'] + 
                                self.beta * r['temporal_score'] + 
                                self.gamma * r['topology_score'])
            elif self.use_temporal:
                combined_score = self.alpha * r['semantic_score'] + self.beta * r['temporal_score']
            elif self.use_topology:
                combined_score = self.alpha * r['semantic_score'] + self.gamma * r['topology_score']
            else:
                combined_score = r['semantic_score']
            
            r['combined_score'] = combined_score
        
        # Sort and return top-k
        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print(f"  ✓ Stage 2 complete: Returning top-{top_k} reranked logs")
        print(f"{'='*60}\n")
        
        return scored_results[:top_k]

if __name__ == "__main__":
    query = "instruction cache parity error corrected"
    query_time = "2004-06-03T18:45:00"  # BGL timestamp format
    query_node = "R02-M1-N0-C:J12-U11"  # BGL node format
    
    print("\n" + "="*80)
    print("LAR-RAG TEMPLATE-FIRST RETRIEVAL DEMO")
    print("="*80)
    
    # Baseline (Semantic Only)
    print("\n### TEST 1: BASELINE (Semantic Only) ###")
    retriever = LARRAGRetriever(
        template_map_file="data/processed/bgl/template_map.jsonl",
        normalize_scores=False  # No normalization needed for single component
    )
    print(f"Query: {query}\n")
    results = retriever.retrieve(query, top_k=5, top_k_templates=20)
    
    print("\nTop 5 Results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Score: {r['combined_score']:.4f}] {r['log_id']}")
        print(f"   Template: {r['template_id']}")
        print(f"   {r['timestamp']} | Node: {r['node']}\n")
    
    # Semantic + Temporal
    print("\n### TEST 2: LAR-RAG (Semantic + Temporal) ###")
    retriever = LARRAGRetriever(
        template_map_file="data/processed/bgl/template_map.jsonl",
        alpha=0.7, beta=0.3,
        use_temporal=True,
        normalize_scores=True  # Enable normalization
    )
    print(f"Query: {query}")
    print(f"Time: {query_time}\n")
    results = retriever.retrieve(query, top_k=5, top_k_templates=20, query_timestamp=query_time)
    
    print("\nTop 5 Results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Total: {r['combined_score']:.4f} = {r['semantic_score']:.4f}·α + {r['temporal_score']:.4f}·β]")
        print(f"   Log: {r['log_id']} | Template: {r['template_id']}")
        print(f"   {r['timestamp']} | Node: {r['node']}\n")
    
    # Semantic + Topology
    print("\n### TEST 3: LAR-RAG (Semantic + Topology) ###")
    retriever = LARRAGRetriever(
        template_map_file="data/processed/bgl/template_map.jsonl",
        alpha=0.7, gamma=0.3,
        use_topology=True,
        normalize_scores=True
    )
    print(f"Query: {query}")
    print(f"Node: {query_node}\n")
    results = retriever.retrieve(query, top_k=5, top_k_templates=20, query_node=query_node)
    
    print("\nTop 5 Results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Total: {r['combined_score']:.4f} = {r['semantic_score']:.4f}·α + {r['topology_score']:.4f}·γ]")
        print(f"   Log: {r['log_id']} | Template: {r['template_id']}")
        print(f"   {r['timestamp']} | Node: {r['node']}\n")
    
    # Full LAR-RAG (All Three Components)
    print("\n### TEST 4: FULL LAR-RAG (Semantic + Temporal + Topology) ###")
    retriever = LARRAGRetriever(
        template_map_file="data/processed/bgl/template_map.jsonl",
        alpha=0.5, beta=0.3, gamma=0.2,
        use_temporal=True,
        use_topology=True,
        normalize_scores=True  # Critical for INFOCOM!
    )
    print(f"Query: {query}")
    print(f"Time: {query_time}")
    print(f"Node: {query_node}\n")
    results = retriever.retrieve(query, top_k=5, top_k_templates=20, 
                                 query_timestamp=query_time, query_node=query_node)
    
    print("\nTop 5 Results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Total: {r['combined_score']:.4f} = {r['semantic_score']:.4f}·α + {r['temporal_score']:.4f}·β + {r['topology_score']:.4f}·γ]")
        print(f"   Log: {r['log_id']} | Template: {r['template_id']}")
        print(f"   {r['timestamp']} | Node: {r['node']}\n")
    
    print("="*80)
    print("DEMO COMPLETE - All tests use template-first architecture!")
    print("="*80)
