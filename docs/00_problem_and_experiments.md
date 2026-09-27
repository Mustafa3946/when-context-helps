# LAR-RAG Experiment Specification

## Problem Statement

Root cause analysis (RCA) in distributed systems requires finding relevant log entries given an incident. Traditional semantic search misses temporal and topological context. LAR-RAG combines dense embeddings with temporal proximity and system topology in a unified scoring function.

**Input**: Incident query (timestamp, node, description)  
**Output**: Ranked log sequences likely containing root cause  
**Challenge**: Balance semantic relevance with temporal/topology priors under latency constraints

## Datasets

Start simple, add more if time permits:
- **HDFS** (primary) - Hadoop distributed file system logs
- **BGL** (secondary) - Blue Gene/L supercomputer logs

## Metrics

- **HR@k** (Hit Rate) - Is relevant log in top-k?
- **MRR** (Mean Reciprocal Rank) - Position of first relevant result
- **mAP** (Mean Average Precision) - Overall ranking quality
- **Query Latency** - Time to retrieve results

Target: < 500ms per query

## Baselines

1. **BM25** - Traditional keyword search
2. **Dense** - Pure semantic embedding (Azure AI Search vector)
3. **Hybrid** - BM25 + Dense combined
4. **LAR-RAG** - Dense + Temporal decay + Topology distance

## LAR-RAG Scoring Function

```
Score(query, log) = α·semantic_sim(q, log) 
                    + β·temporal_decay(Δt) 
                    + γ·topology_proximity(d_graph)

temporal_decay(Δt) = exp(-λ·|t_query - t_log|)
topology_proximity(d) = 1 / (1 + d_graph)
```

Tune α, β, γ, λ on validation set.

## Implementation Plan

1. Dense baseline first (prove pipeline works)
2. Add temporal scoring
3. Add topology scoring
4. Ablation study: which component helps most?
