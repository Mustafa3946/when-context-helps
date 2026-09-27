"""
Evaluation metrics for LAR-RAG retrieval system.
Implements HR@k, MRR, mAP, and latency measurements.
"""
import time
import numpy as np
from typing import List, Dict, Set

class RetrievalMetrics:
    """Calculate standard information retrieval metrics."""
    
    @staticmethod
    def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Hit Rate @ k.
        
        HR@k = 1 if any relevant doc in top-k results, else 0
        
        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            k: Cutoff rank
        
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = set(retrieved[:k])
        return 1.0 if len(top_k & relevant) > 0 else 0.0
    
    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Reciprocal Rank.
        
        RR = 1/rank of first relevant document
        
        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
        
        Returns:
            Reciprocal rank (0.0 if no relevant docs found)
        """
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Average Precision.
        
        AP = (1/|relevant|) * sum(P(k) * rel(k))
        where P(k) is precision at rank k, rel(k) is 1 if k-th doc is relevant
        
        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
        
        Returns:
            Average precision score
        """
        if len(relevant) == 0:
            return 0.0
        
        num_hits = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_hits += 1
                precision_at_i = num_hits / i
                sum_precisions += precision_at_i
        
        return sum_precisions / len(relevant)
    
    @staticmethod
    def mean_reciprocal_rank(queries: List[Dict]) -> float:
        """Calculate Mean Reciprocal Rank across multiple queries.
        
        Args:
            queries: List of dicts with 'retrieved' and 'relevant' keys
        
        Returns:
            MRR score
        """
        rr_scores = [
            RetrievalMetrics.reciprocal_rank(q['retrieved'], q['relevant'])
            for q in queries
        ]
        return np.mean(rr_scores) if rr_scores else 0.0
    
    @staticmethod
    def mean_average_precision(queries: List[Dict]) -> float:
        """Calculate Mean Average Precision across multiple queries.
        
        Args:
            queries: List of dicts with 'retrieved' and 'relevant' keys
        
        Returns:
            mAP score
        """
        ap_scores = [
            RetrievalMetrics.average_precision(q['retrieved'], q['relevant'])
            for q in queries
        ]
        return np.mean(ap_scores) if ap_scores else 0.0

class LatencyMeasurement:
    """Measure query latency."""
    
    def __init__(self):
        self.latencies = []
    
    def measure(self, func, *args, **kwargs):
        """Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to func
        
        Returns:
            (result, latency_ms) tuple
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        self.latencies.append(latency)
        return result, latency
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics.
        
        Returns:
            Dict with mean, median, p95, p99 latencies
        """
        if not self.latencies:
            return {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0}
        
        return {
            'mean': np.mean(self.latencies),
            'median': np.median(self.latencies),
            'p95': np.percentile(self.latencies, 95),
            'p99': np.percentile(self.latencies, 99),
            'min': np.min(self.latencies),
            'max': np.max(self.latencies)
        }
    
    def reset(self):
        """Clear latency measurements."""
        self.latencies = []

class EvaluationFramework:
    """Complete evaluation framework for retrieval systems."""
    
    def __init__(self):
        self.metrics = RetrievalMetrics()
        self.latency = LatencyMeasurement()
    
    def evaluate(self, retriever, test_queries: List[Dict], k_values: List[int] = [1, 5, 10]) -> Dict:
        """Evaluate a retriever on test queries.
        
        Args:
            retriever: Retriever object with retrieve() method
            test_queries: List of dicts with 'query', 'timestamp', 'node', 'relevant_ids'
            k_values: List of k values for HR@k calculation
        
        Returns:
            Dict with all evaluation metrics
        """
        self.latency.reset()
        query_results = []
        hr_at_k = {k: [] for k in k_values}
        
        print(f"Evaluating {len(test_queries)} queries...")
        
        # NEW: Store per-query results for statistical tests
        per_query_results = []
        
        for query_data in test_queries:
            # Retrieve with latency measurement
            results, latency_ms = self.latency.measure(
                retriever.retrieve,
                query_data['query'],
                top_k=max(k_values),
                query_timestamp=query_data.get('query_timestamp'),
                query_node=query_data.get('query_node')
            )
            
            # Handle both direct list results and dict with 'results' key
            if isinstance(results, dict) and 'results' in results:
                results = results['results']
            
            # Extract retrieved IDs
            retrieved_ids = [r['log_id'] for r in results]
            relevant_ids = query_data['relevant']
            
            # Calculate per-query metrics
            rr = self.metrics.reciprocal_rank(retrieved_ids, relevant_ids)
            ap = self.metrics.average_precision(retrieved_ids, relevant_ids)
            
            # Calculate metrics
            query_results.append({
                'retrieved': retrieved_ids,
                'relevant': relevant_ids
            })
            
            # Calculate HR@k for different k values
            hr_dict = {}
            for k in k_values:
                hr = self.metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k)
                hr_at_k[k].append(hr)
                hr_dict[f'HR@{k}'] = hr
            
            # Store per-query result for statistical analysis
            per_query_results.append({
                'query_id': query_data.get('query_id', ''),
                'query': query_data['query'],
                'reciprocal_rank': rr,
                'average_precision': ap,
                **hr_dict,
                'latency_ms': latency_ms,
                'num_relevant': len(relevant_ids),
                'num_retrieved': len(retrieved_ids)
            })
        
        # Aggregate results
        results = {
            'num_queries': len(test_queries),
            'MRR': self.metrics.mean_reciprocal_rank(query_results),
            'mAP': self.metrics.mean_average_precision(query_results),
        }
        
        # Add HR@k for each k
        for k in k_values:
            results[f'HR@{k}'] = np.mean(hr_at_k[k])
        
        # Add latency stats
        latency_stats = self.latency.get_stats()
        results['latency_ms'] = latency_stats
        
        # Add per-query results for statistical testing
        results['per_query'] = per_query_results
        
        return results
    
    def compare_models(self, retrievers: Dict, test_queries: List[Dict], k_values: List[int] = [1, 5, 10]):
        """Compare multiple retriever models.
        
        Args:
            retrievers: Dict of {name: retriever_object}
            test_queries: List of test query dicts
            k_values: List of k values for HR@k
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        all_results = {}
        
        for name, retriever in retrievers.items():
            print(f"\nEvaluating: {name}")
            print("-" * 80)
            results = self.evaluate(retriever, test_queries, k_values)
            all_results[name] = results
            
            # Print results
            print(f"MRR: {results['MRR']:.4f}")
            print(f"mAP: {results['mAP']:.4f}")
            for k in k_values:
                print(f"HR@{k}: {results[f'HR@{k}']:.4f}")
            print(f"Latency (mean): {results['latency_ms']['mean']:.2f}ms")
            print(f"Latency (p95): {results['latency_ms']['p95']:.2f}ms")
        
        # Print comparison table
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(f"{'Model':<30} {'MRR':>8} {'mAP':>8} {'HR@5':>8} {'Latency':>10}")
        print("-" * 80)
        
        for name, results in all_results.items():
            print(f"{name:<30} {results['MRR']:>8.4f} {results['mAP']:>8.4f} "
                  f"{results['HR@5']:>8.4f} {results['latency_ms']['mean']:>9.2f}ms")
        
        return all_results

if __name__ == "__main__":
    # Example usage
    print("Evaluation Framework - Example Usage")
    print("="*60)
    
    # Simulate some test data
    test_queries = [
        {
            'query': 'block corruption',
            'relevant_ids': {'hdfs_10', 'hdfs_25', 'hdfs_50'}
        },
        {
            'query': 'datanode failure',
            'relevant_ids': {'hdfs_15', 'hdfs_30'}
        }
    ]
    
    # Simulate retrieval results
    retrieved_1 = ['hdfs_10', 'hdfs_8', 'hdfs_25', 'hdfs_100', 'hdfs_50']
    retrieved_2 = ['hdfs_5', 'hdfs_15', 'hdfs_30', 'hdfs_80', 'hdfs_90']
    
    # Calculate metrics
    metrics = RetrievalMetrics()
    
    print("\nQuery 1: block corruption")
    print(f"HR@3: {metrics.hit_rate_at_k(retrieved_1, test_queries[0]['relevant_ids'], 3)}")
    print(f"RR: {metrics.reciprocal_rank(retrieved_1, test_queries[0]['relevant_ids']):.4f}")
    print(f"AP: {metrics.average_precision(retrieved_1, test_queries[0]['relevant_ids']):.4f}")
    
    print("\nQuery 2: datanode failure")
    print(f"HR@3: {metrics.hit_rate_at_k(retrieved_2, test_queries[1]['relevant_ids'], 3)}")
    print(f"RR: {metrics.reciprocal_rank(retrieved_2, test_queries[1]['relevant_ids']):.4f}")
    print(f"AP: {metrics.average_precision(retrieved_2, test_queries[1]['relevant_ids']):.4f}")
    
    # Aggregate metrics
    query_results = [
        {'retrieved': retrieved_1, 'relevant': test_queries[0]['relevant_ids']},
        {'retrieved': retrieved_2, 'relevant': test_queries[1]['relevant_ids']}
    ]
    
    print(f"\nAggregate Metrics:")
    print(f"MRR: {metrics.mean_reciprocal_rank(query_results):.4f}")
    print(f"mAP: {metrics.mean_average_precision(query_results):.4f}")
