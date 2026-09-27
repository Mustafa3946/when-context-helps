"""
Build BM25 index from templates.

Saves BM25 index to disk for fast loading during experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse
import json
import pickle

logger = logging.getLogger(__name__)


def build_bm25_index(templates_path: str, output_dir: str):
    """
    Build BM25 index from templates.
    
    Args:
        templates_path: Path to templates.jsonl
        output_dir: Directory to save index files
    """
    from rank_bm25 import BM25Okapi
    
    logger.info("Loading templates...")
    templates = []
    template_ids = []
    
    with open(templates_path, 'r') as f:
        for line in f:
            t = json.loads(line)
            templates.append(t['template'].split())
            template_ids.append(t['template_id'])
    
    logger.info(f"Loaded {len(templates)} templates")
    
    # Build index
    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(templates)
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / 'bm25_index.pkl'
    ids_path = output_dir / 'template_ids.json'
    
    logger.info(f"Saving index to {index_path}")
    with open(index_path, 'wb') as f:
        pickle.dump(bm25, f)
    
    logger.info(f"Saving template IDs to {ids_path}")
    with open(ids_path, 'w') as f:
        json.dump(template_ids, f)
    
    logger.info("✅ BM25 index built successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 index from templates")
    parser.add_argument('--templates', required=True, help='Path to templates.jsonl')
    parser.add_argument('--output', required=True, help='Output directory for index')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    build_bm25_index(args.templates, args.output)
