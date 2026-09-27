"""
Azure AI Search Template Index

Creates and populates an index for BGL log templates with vector search.
"""

import os
import json
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchField,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def create_template_index(index_name: str = "bgl-templates"):
    """
    Create Azure AI Search index for BGL templates.
    
    Args:
        index_name: Name of the index to create
    """
    print(f"Creating index '{index_name}'...")
    
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    
    transport = RequestsTransport(connection_verify=False)
    index_client = SearchIndexClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
        transport=transport
    )
    
    # Check if exists
    try:
        existing = index_client.get_index(index_name)
        print(f"Index '{index_name}' already exists")
        return
    except:
        pass
    
    # Define schema
    fields = [
        SimpleField(name="template_id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="template", type=SearchFieldDataType.String),
        SimpleField(name="count", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchableField(name="example_message", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="embedding-profile"
        ),
    ]
    
    # Vector search config
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")
        ],
        profiles=[
            VectorSearchProfile(
                name="embedding-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ]
    )
    
    # Create index
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    index_client.create_index(index)
    print(f"Index '{index_name}' created successfully")


def upload_templates(
    input_file: str,
    index_name: str = "bgl-templates",
    batch_size: int = 100
):
    """
    Upload embedded templates to Azure AI Search.
    
    Args:
        input_file: Path to template_embeddings.jsonl
        index_name: Name of the index
        batch_size: Documents per batch
    """
    print("="*80)
    print("Uploading BGL Templates to Azure AI Search")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Index: {index_name}")
    print()
    
    # Create index if needed
    create_template_index(index_name)
    
    # Load embedded templates
    print("\nLoading embedded templates...")
    templates = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                templates.append(json.loads(line))
    
    print(f"Loaded {len(templates)} embedded templates")
    
    # Create search client
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    
    transport = RequestsTransport(connection_verify=False)
    search_client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(key),
        transport=transport
    )
    
    # Upload in batches
    print("\nUploading templates...")
    total_uploaded = 0
    
    for i in range(0, len(templates), batch_size):
        batch = templates[i:i + batch_size]
        
        # Prepare documents
        documents = []
        for template in batch:
            doc = {
                "template_id": template["template_id"],
                "template": template["template"],
                "count": template["count"],
                "example_message": template.get("example_message", ""),
                "embedding": template["embedding"]
            }
            documents.append(doc)
        
        # Upload batch
        try:
            result = search_client.upload_documents(documents=documents)
            success_count = sum(1 for r in result if r.succeeded)
            total_uploaded += success_count
            
            batch_num = i//batch_size + 1
            total_batches = (len(templates)-1)//batch_size + 1
            print(f"  Batch {batch_num}/{total_batches}: uploaded {success_count}/{len(batch)} templates")
        
        except Exception as e:
            print(f"  ERROR uploading batch: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Upload Complete!")
    print(f"{'='*80}")
    print(f"Successfully uploaded: {total_uploaded}/{len(templates)} templates")
    print(f"Index: {index_name}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index BGL templates in Azure AI Search")
    parser.add_argument("--input", default="data/processed/bgl/template_embeddings.jsonl",
                       help="Input template embeddings JSONL file")
    parser.add_argument("--index-name", default="bgl-templates",
                       help="Name of the Azure AI Search index")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Documents per batch")
    
    args = parser.parse_args()
    
    upload_templates(args.input, args.index_name, args.batch_size)
