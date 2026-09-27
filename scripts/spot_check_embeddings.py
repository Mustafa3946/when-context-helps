"""
Spot check and validate BGL template embeddings.

Validates:
- Total embedding count matches manifest
- All vectors have correct dimensionality
- No NaN/Inf values
- Random sampling of embeddings

Usage:
    python scripts/spot_check_embeddings.py --embeddings-dir data/processed/bgl/template_embeddings.jsonl
"""

import argparse
import logging
import sys
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.shard_reader import EmbeddingShardReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Validate BGL template embeddings"
    )
    
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory containing template embeddings shards and manifest"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of random embeddings to sample (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Initialize shard reader
    try:
        logger.info(f"Loading embeddings from {args.embeddings_dir}")
        reader = EmbeddingShardReader(args.embeddings_dir)
        
        # Print info
        info = reader.get_info()
        logger.info("=" * 60)
        logger.info("Embeddings Info:")
        logger.info(f"  Directory: {info['embeddings_dir']}")
        logger.info(f"  Shard files: {info['shard_files_count']}")
        logger.info(f"  Model: {info['manifest'].get('model', 'unknown')}")
        logger.info(f"  Dimension: {info['manifest']['embedding_dimension']}")
        logger.info(f"  Total templates: {info['manifest']['total_templates']}")
        logger.info("=" * 60)
        
        # Count embeddings
        logger.info("\nCounting embeddings...")
        actual_count = reader.count_embeddings()
        expected_count = info['manifest']['total_templates']
        
        logger.info(f"Expected count: {expected_count:,}")
        logger.info(f"Actual count: {actual_count:,}")
        
        if actual_count == expected_count:
            logger.info("✅ Count matches manifest")
        else:
            logger.error(f"❌ Count mismatch! Difference: {actual_count - expected_count:,}")
        
        # Validate embeddings
        logger.info("\nValidating embeddings integrity...")
        validation = reader.validate(verbose=True)
        
        if validation['valid']:
            logger.info("✅ All validation checks passed")
        else:
            logger.error("❌ Validation failed!")
            logger.error(f"Details: {validation}")
            sys.exit(1)
        
        # Random sampling
        logger.info(f"\nRandom sampling {args.sample_size} embeddings...")
        
        # Collect all template IDs
        all_template_ids = []
        for template_id, _, _ in reader.iter_embeddings():
            all_template_ids.append(template_id)
        
        logger.info(f"Collected {len(all_template_ids):,} template IDs")
        
        # Sample random IDs
        sample_ids = random.sample(all_template_ids, min(args.sample_size, len(all_template_ids)))
        
        # Fetch embeddings for samples
        logger.info(f"\nFetching {len(sample_ids)} sample embeddings...")
        sample_embeddings = reader.get_embeddings_batch(sample_ids)
        
        logger.info("=" * 60)
        logger.info("Sample Embeddings:")
        logger.info("=" * 60)
        
        for i, template_id in enumerate(sample_ids[:5], start=1):
            if template_id in sample_embeddings:
                embedding = sample_embeddings[template_id]
                logger.info(f"{i}. Template ID: {template_id}")
                logger.info(f"   Dimension: {len(embedding)}")
                logger.info(f"   First 5 values: {embedding[:5]}")
                logger.info(f"   L2 norm: {(embedding ** 2).sum() ** 0.5:.4f}")
            else:
                logger.warning(f"{i}. Template ID {template_id} not found!")
        
        if len(sample_ids) > 5:
            logger.info(f"... and {len(sample_ids) - 5} more")
        
        logger.info("=" * 60)
        logger.info("✅ Spot check complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
