"""
Template Index Creation for Azure AI Search

Creates a template-level index for LAR-RAG retrieval.
Templates are embedded and indexed separately from individual logs.

This enforces the template-first retrieval architecture required for INFOCOM.
"""
import os
import json
import ssl
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv

load_dotenv()

# Disable SSL verification for corporate proxy
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"

TEMPLATE_INDEX_NAME = "bgl-templates"


def create_template_index(search_endpoint=None, search_key=None):
    """
    Create Azure AI Search index for BGL templates.
    
    Schema:
    - template_id: Unique template identifier (key)
    - template: Template text with placeholders
    - example_message: Representative log message
    - count: Number of logs matching this template
    - embedding: Dense vector (1536-dim for text-embedding-3-small)
    
    Args:
        search_endpoint: Azure Search endpoint (or from env)
        search_key: Azure Search admin key (or from env)
    """
    endpoint = search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
    key = search_key or os.getenv("AZURE_SEARCH_ADMIN_KEY")
    
    if not endpoint or not key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_ADMIN_KEY must be set")
    
    transport = RequestsTransport(connection_verify=False)
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key), transport=transport)
    
    # Define template index schema
    fields = [
        SimpleField(
            name="template_id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchableField(
            name="template",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="standard.lucene"
        ),
        SearchableField(
            name="example_message",
            type=SearchFieldDataType.String,
            searchable=False  # Not used for search, just for display
        ),
        SimpleField(
            name="count",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # text-embedding-3-small
            vector_search_profile_name="template-vector-profile"
        )
    ]
    
    # Configure vector search with HNSW
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="template-vector-profile",
                algorithm_configuration_name="template-hnsw-config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="template-hnsw-config",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ]
    )
    
    # Create index
    index = SearchIndex(
        name=TEMPLATE_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )
    
    print(f"Creating template index: {TEMPLATE_INDEX_NAME}")
    try:
        index_client.create_or_update_index(index)
        print(f"✓ Template index '{TEMPLATE_INDEX_NAME}' created successfully")
    except Exception as e:
        print(f"✗ Error creating index: {e}")
        raise
    
    return TEMPLATE_INDEX_NAME


def upload_templates_to_index(templates_file, search_endpoint=None, search_key=None):
    """
    Upload BGL templates with embeddings to Azure AI Search.
    
    Expects templates_file to be a JSONL with format:
    {
        "template_id": "abc123...",
        "template": "RAS KERNEL INFO...",
        "example_message": "Original log message",
        "count": 42,
        "embedding": [0.1, 0.2, ...]
    }
    
    Args:
        templates_file: Path to template embeddings JSONL file
        search_endpoint: Azure Search endpoint (or from env)
        search_key: Azure Search admin key (or from env)
    """
    endpoint = search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
    key = search_key or os.getenv("AZURE_SEARCH_ADMIN_KEY")
    
    transport = RequestsTransport(connection_verify=False)
    search_client = SearchClient(
        endpoint=endpoint,
        index_name=TEMPLATE_INDEX_NAME,
        credential=AzureKeyCredential(key),
        transport=transport
    )
    
    print(f"Loading templates from {templates_file}...")
    
    # Load templates
    templates = []
    with open(templates_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                templates.append(json.loads(line))
    
    print(f"Loaded {len(templates)} templates")
    
    # Prepare documents for upload
    documents = []
    for template in templates:
        doc = {
            "template_id": template["template_id"],
            "template": template["template"],
            "example_message": template.get("example_message", ""),
            "count": template.get("count", 1),
            "embedding": template["embedding"]
        }
        documents.append(doc)
    
    # Upload in batches
    batch_size = 100
    total_uploaded = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            result = search_client.upload_documents(documents=batch)
            success_count = sum(1 for r in result if r.succeeded)
            total_uploaded += success_count
            print(f"  Uploaded batch {i//batch_size + 1}: {success_count}/{len(batch)} documents")
        except Exception as e:
            print(f"  ✗ Error uploading batch: {e}")
    
    print(f"\n✓ Total uploaded: {total_uploaded}/{len(documents)} templates")
    return total_uploaded


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and populate BGL template index")
    parser.add_argument("--create-index", action="store_true", help="Create the template index")
    parser.add_argument("--upload-templates", type=str, help="Upload templates from JSONL file")
    parser.add_argument("--endpoint", type=str, help="Azure Search endpoint")
    parser.add_argument("--key", type=str, help="Azure Search admin key")
    
    args = parser.parse_args()
    
    if args.create_index:
        create_template_index(args.endpoint, args.key)
    
    if args.upload_templates:
        upload_templates_to_index(args.upload_templates, args.endpoint, args.key)
    
    if not args.create_index and not args.upload_templates:
        # Default: create index and upload if file exists
        create_template_index(args.endpoint, args.key)
        
        default_file = "data/processed/bgl/template_embeddings.jsonl"
        if os.path.exists(default_file):
            upload_templates_to_index(default_file, args.endpoint, args.key)
        else:
            print(f"\nNo template file found at {default_file}")
            print("Run: python src/embeddings/embed_templates_blob.py")
