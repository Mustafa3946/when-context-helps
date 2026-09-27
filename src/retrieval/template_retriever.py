"""
Template-Based Retrieval for BGL Logs

Retrieves logs by:
1. Embedding query once
2. Retrieving top-K templates via vector search
3. Expanding templates to actual log instances
4. Filtering by time/topology
5. Returning ranked logs
"""

import os
import json
import ssl
from datetime import datetime
from typing import List, Dict
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv

load_dotenv()

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"


class TemplateRetriever:
    """
    Retrieves BGL logs using template-based search.
    
    This is much faster and cheaper than embedding every log.
    """
    
    def __init__(
        self,
        template_map_file: str,
        index_name: str = "bgl-templates"
    ):
        """
        Initialize template retriever.
        
        Args:
            template_map_file: Path to template_map.jsonl
            index_name: Azure AI Search index name
        """
        # Load template map
        print(f"Loading template map from {template_map_file}...")
        self.template_to_logs = {}
        with open(template_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    mapping = json.loads(line)
                    template_id = mapping['template_id']
                    
                    if template_id not in self.template_to_logs:
                        self.template_to_logs[template_id] = []
                    
                    self.template_to_logs[template_id].append({
                        'log_id': mapping['log_id'],
                        'timestamp': mapping['timestamp'],
                        'node': mapping['node']
                    })
        
        print(f"Loaded mappings for {len(self.template_to_logs)} templates")
        
        # Lazy imports for Azure-only dependencies
        from openai import AzureOpenAI
        import httpx
        
        # Azure clients
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        
        transport = RequestsTransport(connection_verify=False)
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(key),
            transport=transport
        )
        
        http_client = httpx.Client(verify=False)
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            http_client=http_client
        )
        
        self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    
    def retrieve(
        self,
        query_text: str,
        top_k_templates: int = 50,
        max_logs_per_template: int = 10,
        incident_time: str = None,
        time_window_seconds: int = 3600,
        incident_node: str = None
    ) -> List[Dict]:
        """
        Retrieve logs for an incident query.
        
        Args:
            query_text: Incident description
            top_k_templates: Number of templates to retrieve
            max_logs_per_template: Max log instances per template
            incident_time: ISO timestamp for temporal filtering
            time_window_seconds: Time window for filtering
            incident_node: Node ID for topology filtering
            
        Returns:
            List of log dicts with scores
        """
        # 1. Embed query (once)
        print(f"Embedding query: '{query_text}'")
        response = self.openai_client.embeddings.create(
            input=[query_text],
            model=self.deployment
        )
        query_embedding = response.data[0].embedding
        
        # 2. Search templates
        print(f"Searching for top {top_k_templates} templates...")
        results = self.search_client.search(
            search_text=None,
            vector_queries=[{
                "kind": "vector",
                "vector": query_embedding,
                "fields": "embedding",
                "k": top_k_templates
            }],
            select=["template_id", "template", "count"],
            top=top_k_templates
        )
        
        # 3. Expand templates to logs
        print("Expanding templates to log instances...")
        candidate_logs = []
        
        for result in results:
            template_id = result['template_id']
            template_score = result['@search.score']
            
            # Get logs for this template
            if template_id in self.template_to_logs:
                logs = self.template_to_logs[template_id]
                
                # Limit logs per template
                for log in logs[:max_logs_per_template]:
                    candidate_logs.append({
                        'log_id': log['log_id'],
                        'timestamp': log['timestamp'],
                        'node': log['node'],
                        'template_id': template_id,
                        'template_score': template_score
                    })
        
        print(f"Expanded to {len(candidate_logs)} candidate logs")
        
        # 4. Filter by time window
        if incident_time:
            incident_dt = datetime.fromisoformat(incident_time)
            filtered = []
            
            for log in candidate_logs:
                log_dt = datetime.fromisoformat(log['timestamp'])
                delta_s = abs((incident_dt - log_dt).total_seconds())
                
                if delta_s <= time_window_seconds:
                    log['time_delta_s'] = delta_s
                    filtered.append(log)
            
            candidate_logs = filtered
            print(f"After time filtering: {len(candidate_logs)} logs")
        
        # 5. Filter by node (optional)
        if incident_node:
            filtered = [log for log in candidate_logs if log['node'] == incident_node]
            print(f"After node filtering: {len(filtered)} logs")
            candidate_logs = filtered
        
        # 6. Sort by template score
        candidate_logs.sort(key=lambda x: x['template_score'], reverse=True)
        
        return candidate_logs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BGL template retrieval")
    parser.add_argument("--template-map", default="data/processed/bgl/template_map.jsonl",
                       help="Template map file")
    parser.add_argument("--query", default="instruction cache parity error",
                       help="Test query")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of templates to retrieve")
    
    args = parser.parse_args()
    
    # Test retrieval
    retriever = TemplateRetriever(args.template_map)
    
    results = retriever.retrieve(
        query_text=args.query,
        top_k_templates=args.top_k
    )
    
    print(f"\nRetrieved {len(results)} logs")
    print("\nTop 10 results:")
    for i, log in enumerate(results[:10], 1):
        print(f"{i}. {log['log_id']} (score: {log['template_score']:.4f})")
