"""
Centralized configuration management for LAR-RAG.

This module provides:
1. Environment variable validation with fail-fast behavior
2. SSL/TLS configuration for corporate proxies
3. Azure service client factories
4. Experiment configuration loading

Usage:
    from src.config import get_azure_config, create_search_client, create_openai_client
    
    # Validate environment and get config
    config = get_azure_config()
    
    # Create Azure clients
    search_client = create_search_client(index_name="bgl-templates")
    openai_client = create_openai_client()
"""

import os
import ssl
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport

# Load environment variables once at module import
load_dotenv()

# Configure SSL for corporate proxies (must be done before any HTTPS calls)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"


@dataclass
class AzureConfig:
    """Azure service configuration."""
    
    # Azure OpenAI (required)
    openai_api_key: str
    openai_endpoint: str
    
    # Azure AI Search (required)
    search_endpoint: str
    search_admin_key: str
    
    # Azure OpenAI (with defaults)
    openai_api_version: str = "2023-05-15"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Azure AI Search (with defaults)
    search_template_index: str = "bgl-templates"
    search_log_index: str = "log-embeddings"  # Legacy, deprecated
    
    # Azure Blob Storage (optional)
    storage_connection_string: Optional[str] = None
    storage_account_name: Optional[str] = None
    
    # Template mapping
    template_map_path: str = "data/processed/template_map.jsonl"
    

def get_azure_config(fail_fast: bool = True) -> AzureConfig:
    """
    Load and validate Azure configuration from environment variables.
    
    Args:
        fail_fast: If True, raise ValueError on missing required variables.
                   If False, return None for missing variables.
    
    Returns:
        AzureConfig instance
    
    Raises:
        ValueError: If fail_fast=True and required environment variables are missing.
    
    Example:
        >>> config = get_azure_config()
        >>> print(config.openai_endpoint)
        https://your-resource.openai.azure.com
    """
    required_vars = {
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API key",
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI endpoint",
        "AZURE_SEARCH_ENDPOINT": "Azure AI Search endpoint",
        "AZURE_SEARCH_ADMIN_KEY": "Azure AI Search admin key",
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"  - {var}: {description}")
    
    if missing and fail_fast:
        raise ValueError(
            "Missing required environment variables:\n" + 
            "\n".join(missing) + 
            "\n\nPlease create a .env file with these variables or set them in your environment."
        )
    
    return AzureConfig(
        # Azure OpenAI
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        openai_embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        
        # Azure AI Search
        search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
        search_admin_key=os.getenv("AZURE_SEARCH_ADMIN_KEY", ""),
        search_template_index=os.getenv("AZURE_SEARCH_TEMPLATE_INDEX", "bgl-templates"),
        search_log_index=os.getenv("AZURE_SEARCH_LOG_INDEX", "log-embeddings"),
        
        # Azure Blob Storage (optional)
        storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
        
        # Template mapping
        template_map_path=os.getenv("TEMPLATE_MAP_PATH", "data/processed/template_map.jsonl"),
    )


def create_search_client(
    index_name: Optional[str] = None,
    config: Optional[AzureConfig] = None
) -> SearchClient:
    """
    Create Azure AI Search client with SSL bypass for corporate proxies.
    
    Args:
        index_name: Index name (default: bgl-templates from config)
        config: AzureConfig instance (default: load from environment)
    
    Returns:
        Configured SearchClient instance
    
    Example:
        >>> client = create_search_client(index_name="bgl-templates")
        >>> results = client.search(search_text="error")
    """
    if config is None:
        config = get_azure_config()
    
    if index_name is None:
        index_name = config.search_template_index
    
    transport = RequestsTransport(connection_verify=False)
    return SearchClient(
        endpoint=config.search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(config.search_admin_key),
        transport=transport
    )


def create_openai_client(config: Optional[AzureConfig] = None):
    """
    Create Azure OpenAI client with SSL bypass for corporate proxies.
    
    Args:
        config: AzureConfig instance (default: load from environment)
    
    Returns:
        Configured AzureOpenAI instance
    
    Example:
        >>> client = create_openai_client()
        >>> response = client.embeddings.create(
        ...     input="test query",
        ...     model="text-embedding-3-small"
        ... )
    """
    # Lazy import for Azure-only dependencies
    from openai import AzureOpenAI
    import httpx
    
    if config is None:
        config = get_azure_config()
    
    http_client = httpx.Client(verify=False)
    return AzureOpenAI(
        api_key=config.openai_api_key,
        api_version=config.openai_api_version,
        azure_endpoint=config.openai_endpoint,
        http_client=http_client
    )


def validate_environment(verbose: bool = True) -> bool:
    """
    Validate that all required environment variables are set.
    
    Args:
        verbose: If True, print validation results
    
    Returns:
        True if all required variables are set, False otherwise
    
    Example:
        >>> if not validate_environment():
        ...     print("Please configure your .env file")
        ...     sys.exit(1)
    """
    try:
        config = get_azure_config(fail_fast=True)
        if verbose:
            print("✅ Environment validation passed")
            print(f"   - Azure OpenAI: {config.openai_endpoint}")
            print(f"   - Azure Search: {config.search_endpoint}")
            print(f"   - Template Index: {config.search_template_index}")
        return True
    except ValueError as e:
        if verbose:
            print("❌ Environment validation failed:")
            print(str(e))
        return False


# Convenience: Pre-configured singleton instance (lazy-loaded)
_config_instance: Optional[AzureConfig] = None


def get_config() -> AzureConfig:
    """Get singleton config instance (cached after first call)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = get_azure_config()
    return _config_instance


if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("LAR-RAG Configuration Validation")
    print("=" * 60)
    
    if validate_environment(verbose=True):
        print("\n✅ All Azure services are configured correctly!")
        print("\nYou can now run experiments with:")
        print("  python src/evaluation/experiment_runner.py")
    else:
        print("\n❌ Configuration incomplete. Please check your .env file.")
        print("\nRequired variables:")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_SEARCH_ENDPOINT")
        print("  - AZURE_SEARCH_ADMIN_KEY")
