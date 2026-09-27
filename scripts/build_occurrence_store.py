"""
Build Occurrence Store from Template Map

CLI entrypoint for building DuckDB occurrence store.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse
from src.storage.occurrence_store import build_occurrence_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DuckDB occurrence store from template_map.jsonl"
    )
    parser.add_argument(
        '--template-map',
        default='data/processed/bgl/template_map.jsonl',
        help='Path to template_map.jsonl'
    )
    parser.add_argument(
        '--output',
        default='data/processed/bgl/occurrences.duckdb',
        help='Output DuckDB database path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Insert batch size for performance'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    build_occurrence_store(
        args.template_map,
        args.output,
        args.batch_size
    )
