"""
Build DuckDB database from template_map.jsonl for memory-bounded ground truth generation.

RATIONALE:
- Full template_map.jsonl (1.169M templates) cannot fit in RAM
- DuckDB provides disk-backed SQL queries with indexing
- Ground truth builder queries log_id -> template_id without loading all data

Usage:
    python scripts/build_template_map_db.py \\
        --template-map data/processed/bgl/template_map.jsonl \\
        --output data/processed/bgl/template_map.duckdb
"""

import argparse
import json
import logging
from pathlib import Path
import duckdb

logger = logging.getLogger(__name__)


def build_template_map_db(
    template_map_file: str,
    output_db: str,
    batch_size: int = 50000
):
    """
    Build DuckDB database from template_map.jsonl.
    
    Schema:
        CREATE TABLE template_map (
            log_id VARCHAR,
            template_id VARCHAR,
            timestamp BIGINT,
            node_id VARCHAR
        );
        
    Indexes:
        - idx_tm_log_id ON (log_id)
        - idx_tm_time_node ON (timestamp, node_id)
    
    Args:
        template_map_file: Path to template_map.jsonl
        output_db: Output DuckDB file path
        batch_size: Rows per batch (default: 50k for speed)
    """
    logger.info("="*80)
    logger.info("BUILDING TEMPLATE MAP DUCKDB DATABASE")
    logger.info("="*80)
    logger.info(f"Input: {template_map_file}")
    logger.info(f"Output: {output_db}")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info("="*80)
    
    # Create output directory
    output_path = Path(output_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database
    if output_path.exists():
        logger.info(f"Removing existing database: {output_db}")
        output_path.unlink()
    
    # Connect to DuckDB
    conn = duckdb.connect(str(output_db))
    
    # Create table
    logger.info("Creating table schema...")
    conn.execute("""
        CREATE TABLE template_map (
            log_id VARCHAR,
            template_id VARCHAR,
            timestamp BIGINT,
            node_id VARCHAR
        )
    """)
    
    # Stream and insert in batches
    logger.info("Streaming template_map.jsonl...")
    batch = []
    total_rows = 0
    
    with open(template_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            mapping = json.loads(line)
            batch.append((
                mapping['log_id'],
                mapping['template_id'],
                mapping.get('timestamp', 0),  # Unix timestamp
                mapping.get('node_id', '')
            ))
            
            # Insert batch
            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO template_map VALUES (?, ?, ?, ?)",
                    batch
                )
                total_rows += len(batch)
                logger.info(f"  Inserted {total_rows:,} rows...")
                batch = []
        
        # Insert remaining
        if batch:
            conn.executemany(
                "INSERT INTO template_map VALUES (?, ?, ?, ?)",
                batch
            )
            total_rows += len(batch)
    
    logger.info(f"✅ Inserted {total_rows:,} total rows")
    
    # Create indexes
    logger.info("Creating indexes...")
    
    logger.info("  - idx_tm_log_id ON (log_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_log_id ON template_map(log_id)")
    
    logger.info("  - idx_tm_time_node ON (timestamp, node_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_time_node ON template_map(timestamp, node_id)")
    
    logger.info("  - idx_tm_template_time ON (template_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_template_time ON template_map(template_id, timestamp)")
    
    # Verify
    count = conn.execute("SELECT COUNT(*) FROM template_map").fetchone()[0]
    logger.info(f"✅ Database built: {count:,} rows")
    
    # Show sample
    logger.info("Sample rows:")
    sample = conn.execute("SELECT * FROM template_map LIMIT 3").fetchall()
    for row in sample:
        logger.info(f"  {row}")
    
    conn.close()
    
    logger.info("="*80)
    logger.info("✅ TEMPLATE MAP DATABASE BUILD COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DuckDB database from template_map.jsonl"
    )
    parser.add_argument(
        '--template-map',
        required=True,
        help='Path to template_map.jsonl'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output DuckDB file path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50000,
        help='Rows per batch (default: 50000)'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    build_template_map_db(
        template_map_file=args.template_map,
        output_db=args.output,
        batch_size=args.batch_size
    )
