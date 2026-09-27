# LAR-RAG Algorithm Pseudocode

## LATeX Version (for paper)

```latex
\begin{algorithm}
\caption{LAR-RAG: Two-Stage Retrieval for RCA}
\begin{algorithmic}[1]
\REQUIRE Query $q$, incident time $t_q$, incident node $v_q$, corpus $\mathcal{C}$
\ENSURE Ranked list of logs $L$

\STATE \textbf{Stage 1: Initial Retrieval}
\STATE $C \leftarrow \text{HybridSearch}(q, \mathcal{C}, k=50)$ 
\COMMENT{BM25 + Dense with RRF fusion}

\STATE \textbf{Stage 2: LAR-RAG Re-ranking}
\FOR{each log $\ell \in C$}
    \STATE $s_{\text{sem}} \leftarrow \text{CosineSim}(\text{emb}(q), \text{emb}(\ell))$
    
    \STATE \COMMENT{Temporal decay with RCA semantics}
    \STATE $\Delta t \leftarrow t_q - \ell.t$
    \IF{$\Delta t \geq 0$} \COMMENT{Log before incident}
        \STATE $g_t \leftarrow \exp(-\lambda_t \cdot \Delta t)$
    \ELSE \COMMENT{Log after incident - penalize}
        \STATE $g_t \leftarrow \exp(-\lambda_{\text{post}} \cdot |\Delta t|)$
    \ENDIF
    
    \STATE \COMMENT{Topology proximity via graph distance}
    \STATE $d_g \leftarrow \text{ShortestPath}(v_q, \ell.v, G)$
    \STATE $g_g \leftarrow \exp(-\lambda_g \cdot d_g)$
    
    \STATE \COMMENT{Combined scoring}
    \STATE $\ell.\text{score} \leftarrow \alpha \cdot s_{\text{sem}} + \beta \cdot g_t + \gamma \cdot g_g$
\ENDFOR

\STATE $L \leftarrow \text{Sort}(C, \text{by score descending})$
\RETURN $L[1:k]$
\end{algorithmic}
\end{algorithm}
```

## Python-style Pseudocode

```python
def lar_rag_retrieve(query, incident_time, incident_node, corpus, k=10):
    """
    LAR-RAG: Latency-Aware Retrieval for Root Cause Analysis
    
    Parameters:
        query: Natural language incident description
        incident_time: Timestamp when incident was detected
        incident_node: System component where incident occurred
        corpus: Log corpus
        k: Number of results to return
    
    Returns:
        Top-k relevant logs for RCA
    """
    
    # Stage 1: Initial Retrieval (50 candidates)
    # Uses Reciprocal Rank Fusion of BM25 (lexical) + Dense (semantic)
    candidates = hybrid_search(query, corpus, k=50)
    
    # Stage 2: LAR-RAG Re-ranking
    for log in candidates:
        # Semantic similarity (from dense embedding)
        semantic_score = cosine_similarity(embed(query), embed(log))
        
        # Temporal decay (RCA-aware)
        Δt = incident_time - log.timestamp
        if Δt >= 0:  # Log before incident (expected)
            temporal_score = exp(-λ_t * Δt)
        else:  # Log after incident (less relevant)
            temporal_score = exp(-λ_post * |Δt|)
        
        # Topology proximity (via system graph)
        graph_distance = shortest_path(incident_node, log.node, system_graph)
        topology_score = exp(-λ_g * graph_distance)
        
        # Combined score
        log.score = α * semantic_score + β * temporal_score + γ * topology_score
    
    # Return top-k
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:k]
```

## Key Components

### 1. Two-Stage Architecture
- **Stage 1 (Hybrid Search)**: Fast initial retrieval combining BM25 and dense search
  - BM25: Lexical matching for exact term matches
  - Dense: Semantic embedding similarity
  - RRF: Reciprocal Rank Fusion to merge results
  - Output: ~50 candidate logs

- **Stage 2 (LAR-RAG Re-ranking)**: RCA-specific scoring
  - Semantic: Relevance to incident description
  - Temporal: Preference for logs BEFORE incident
  - Topology: Preference for logs from nearby components
  - Output: Top-k ranked logs

### 2. Temporal Prior with RCA Semantics
```
g_time(Δt) = exp(-λ_t * Δt)         if Δt ≥ 0  (past)
           = exp(-λ_post * |Δt|)     if Δt < 0  (future)
```
where:
- Δt = incident_time - log_time
- λ_t = 0.001 (gentle decay for past logs)
- λ_post = 0.01 (strong penalty for future logs)

Rationale: Root causes occur BEFORE failure detection, not after.

### 3. Topology Prior via System Graph
```
g_topology(v_query, v_log) = exp(-λ_g * d_graph)
```
where:
- d_graph = shortest path hop count between nodes
- λ_g = 0.5 (decay rate)
- System graph represents dependencies/communication links

Rationale: Failures propagate through connected components.

### 4. Hyperparameters
Default configuration (empirically tuned):
- α = 0.5 (semantic weight)
- β = 0.3 (temporal weight)
- γ = 0.2 (topology weight)
- λ_t = 0.001 (temporal decay)
- λ_post = 0.01 (future penalty)
- λ_g = 0.5 (topology decay)

## Complexity Analysis

**Time Complexity:**
- Stage 1: O(|C| log k) for top-k hybrid search
- Stage 2: O(k · d) where k=50 candidates, d=embedding dimension
- Total: O(|C| log k + k · d)

**Space Complexity:**
- O(|C| · d) for log embeddings
- O(|V|² ) for graph distances (cached)

**Latency:**
- Stage 1: ~300-400ms (network + search)
- Stage 2: ~200-300ms (re-ranking)
- Total: <700ms (well under 1s requirement)
