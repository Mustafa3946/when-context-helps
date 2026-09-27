"""
Configuration Validation for INFOCOM-Compliant LAR-RAG Experiments

Validates canonical v2 schema and fails fast on legacy configurations.

CANONICAL SCHEMA (v2):
- run: {name, seed, output_dir, top_k}
- data: {templates_path, embeddings: {manifest_path, shards_dir}, occurrence_store: {db_path, table}}
- ground_truth: {incidents_path, qrels_path, split: {method, train_frac}}
- retrieval: {bm25: {index_dir}, faiss: {index_path, meta_path, template_ids_path}}
- lar_rag: {alpha, beta, gamma, time_kernel: {...}, topo_kernel: {...}, stage1: {...}}
- methods: {method_name: {enabled, ...params}}
- evaluation: {metrics, save_per_query, save_summary, save_run_meta}

LEGACY KEYS (forbidden):
- embedding.model → embeddings.manifest_path
- embeddings_dir → embeddings.manifest_path
- template_map → occurrence_store.db_path
- faiss_index → retrieval.faiss.index_path
"""

import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Set
import logging

logger = logging.getLogger(__name__)


# Legacy keys that must not appear
FORBIDDEN_LEGACY_KEYS = {
    'embedding': 'Use embeddings.manifest_path instead',
    'embeddings_dir': 'Use embeddings.manifest_path + shards_dir instead',
    'template_map': 'Use occurrence_store.db_path instead',
    'faiss_index': 'Use retrieval.faiss.index_path instead',
}

# Required top-level keys
REQUIRED_KEYS = {
    'run', 'data', 'ground_truth', 'retrieval', 'lar_rag', 'methods', 'evaluation'
}


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def validate_config_schema(config: Dict[str, Any]) -> None:
    """
    Validate configuration against canonical v2 schema.
    
    Args:
        config: Parsed YAML configuration
        
    Raises:
        ConfigValidationError: If validation fails
    """
    logger.info("Validating configuration schema (v2 canonical)")
    
    # Check for legacy keys (anywhere in config)
    _check_no_legacy_keys(config)
    
    # Check required top-level keys
    _check_required_keys(config, REQUIRED_KEYS, path='root')
    
    # Validate run section
    _validate_run_section(config['run'])
    
    # Validate data section
    _validate_data_section(config['data'])
    
    # Validate ground_truth section
    _validate_ground_truth_section(config['ground_truth'])
    
    # Validate retrieval section
    _validate_retrieval_section(config['retrieval'])
    
    # Validate lar_rag section
    _validate_lar_rag_section(config['lar_rag'])
    
    # Validate methods section
    _validate_methods_section(config['methods'])
    
    # Validate evaluation section
    _validate_evaluation_section(config['evaluation'])
    
    logger.info("✅ Configuration schema validated")


def _check_no_legacy_keys(obj: Any, path: str = 'root') -> None:
    """Recursively check for legacy keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}"
            
            # Check if key is forbidden
            if key in FORBIDDEN_LEGACY_KEYS:
                raise ConfigValidationError(
                    f"❌ LEGACY KEY DETECTED: '{current_path}'\n"
                    f"   Reason: {FORBIDDEN_LEGACY_KEYS[key]}\n"
                    f"   Use canonical v2 schema (see config/bgl_canonical_v2.yaml)"
                )
            
            # Recurse
            _check_no_legacy_keys(value, current_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _check_no_legacy_keys(item, f"{path}[{i}]")


def _check_required_keys(obj: Dict, required: Set[str], path: str) -> None:
    """Check required keys exist."""
    missing = required - set(obj.keys())
    if missing:
        raise ConfigValidationError(
            f"❌ Missing required keys in '{path}': {missing}"
        )


def _validate_run_section(run: Dict[str, Any]) -> None:
    """Validate run section."""
    required = {'name', 'seed', 'output_dir', 'top_k'}
    _check_required_keys(run, required, 'run')
    
    if not isinstance(run['seed'], int):
        raise ConfigValidationError("run.seed must be an integer")
    
    if not isinstance(run['top_k'], int) or run['top_k'] <= 0:
        raise ConfigValidationError("run.top_k must be a positive integer")


def _validate_data_section(data: Dict[str, Any]) -> None:
    """Validate data section."""
    required = {'templates_path', 'embeddings', 'occurrence_store'}
    _check_required_keys(data, required, 'data')
    
    # Validate embeddings
    emb = data['embeddings']
    _check_required_keys(emb, {'manifest_path', 'shards_dir'}, 'data.embeddings')
    
    # Validate occurrence_store
    occ = data['occurrence_store']
    _check_required_keys(occ, {'db_path', 'table'}, 'data.occurrence_store')
    
    # Check for JSONL embeddings (forbidden)
    if 'manifest_path' in emb:
        manifest_path = str(emb['manifest_path'])
        if manifest_path.endswith('.jsonl'):
            raise ConfigValidationError(
                f"❌ JSONL embeddings forbidden: {manifest_path}\n"
                f"   Use v2 Parquet shards + manifest.json"
            )


def _validate_ground_truth_section(gt: Dict[str, Any]) -> None:
    """Validate ground_truth section."""
    required = {'incidents_path', 'qrels_path', 'split'}
    _check_required_keys(gt, required, 'ground_truth')
    
    # Validate split
    split = gt['split']
    _check_required_keys(split, {'method', 'train_frac'}, 'ground_truth.split')
    
    if split['method'] not in ['time', 'random']:
        raise ConfigValidationError(
            f"ground_truth.split.method must be 'time' or 'random', got {split['method']}"
        )
    
    if not (0.0 < split['train_frac'] < 1.0):
        raise ConfigValidationError(
            "ground_truth.split.train_frac must be in (0, 1)"
        )


def _validate_retrieval_section(retr: Dict[str, Any]) -> None:
    """Validate retrieval section."""
    required = {'bm25', 'faiss'}
    _check_required_keys(retr, required, 'retrieval')
    
    # Validate BM25
    _check_required_keys(retr['bm25'], {'index_dir'}, 'retrieval.bm25')
    
    # Validate FAISS
    faiss = retr['faiss']
    _check_required_keys(
        faiss,
        {'index_path', 'meta_path', 'template_ids_path'},
        'retrieval.faiss'
    )


def _validate_lar_rag_section(lar: Dict[str, Any]) -> None:
    """Validate lar_rag section."""
    required = {'alpha', 'beta', 'gamma', 'time_kernel', 'topo_kernel', 'stage1'}
    _check_required_keys(lar, required, 'lar_rag')
    
    # Check weights sum to reasonable value (allow some flexibility)
    weight_sum = lar['alpha'] + lar['beta'] + lar['gamma']
    if not (0.9 <= weight_sum <= 2.0):
        logger.warning(
            f"⚠️  LAR-RAG weights sum to {weight_sum:.2f} (expected ~1.0-2.0)"
        )
    
    # Validate kernels
    _check_required_keys(lar['time_kernel'], {'kind', 'tau_before', 'tau_after'}, 'lar_rag.time_kernel')
    _check_required_keys(lar['topo_kernel'], {'kind', 'tau'}, 'lar_rag.topo_kernel')
    
    # Validate stage1
    _check_required_keys(lar['stage1'], {'mode', 'candidates'}, 'lar_rag.stage1')
    
    if lar['stage1']['mode'] not in ['bm25', 'dense', 'hybrid']:
        raise ConfigValidationError(
            f"lar_rag.stage1.mode must be bm25|dense|hybrid, got {lar['stage1']['mode']}"
        )


def _validate_methods_section(methods: Dict[str, Any]) -> None:
    """Validate methods section."""
    if not methods:
        raise ConfigValidationError("methods section must define at least one method")
    
    # Each method must have 'enabled' key
    for method_name, method_config in methods.items():
        if not isinstance(method_config, dict):
            raise ConfigValidationError(
                f"methods.{method_name} must be a dict"
            )
        
        if 'enabled' not in method_config:
            raise ConfigValidationError(
                f"methods.{method_name} must have 'enabled' key"
            )


def _validate_evaluation_section(evalu: Dict[str, Any]) -> None:
    """Validate evaluation section."""
    required = {'metrics', 'save_per_query', 'save_summary', 'save_run_meta'}
    _check_required_keys(evalu, required, 'evaluation')
    
    if not evalu['metrics']:
        raise ConfigValidationError("evaluation.metrics must define at least one metric")
    
    # Validate metric names
    valid_metrics = {
        'recall@1', 'recall@3', 'recall@5', 'recall@10', 'recall@20',
        'mrr@10', 'mrr@20', 'mrr',
        'ndcg@10', 'ndcg@20', 'ndcg',
        'map', 'hit_rate@1', 'hit_rate@5', 'hit_rate@10'
    }
    
    for metric in evalu['metrics']:
        if metric not in valid_metrics:
            logger.warning(f"⚠️  Unknown metric: {metric}")


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of configuration.
    
    Args:
        config: Configuration dict
        
    Returns:
        SHA256 hash (hex string)
    """
    # Canonicalize by sorting keys
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to YAML configuration
        
    Returns:
        Validated configuration dict
        
    Raises:
        ConfigValidationError: If validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate schema
    validate_config_schema(config)
    
    # Compute hash
    config_hash = compute_config_hash(config)
    logger.info(f"Config hash: {config_hash[:16]}...")
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate LAR-RAG configuration"
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration YAML'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    try:
        config = load_and_validate_config(args.config)
        config_hash = compute_config_hash(config)
        print(f"\n✅ Configuration valid")
        print(f"   Hash: {config_hash}")
    except (ConfigValidationError, FileNotFoundError) as e:
        print(f"\n❌ Validation failed:\n{e}")
        exit(1)
