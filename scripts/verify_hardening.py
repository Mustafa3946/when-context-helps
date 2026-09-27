"""
Verify LAR-RAG implementation hardening.

Tests all critical fixes:
1. Embedding manifest reading
2. Spot-check validation
3. Shard reader functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_manifest_reading(embeddings_dir: str):
    """Test that manifest is read correctly."""
    from src.embeddings.shard_reader import EmbeddingShardReader
    
    logger.info("=" * 60)
    logger.info("Test 1: Manifest Reading")
    logger.info("=" * 60)
    
    try:
        reader = EmbeddingShardReader(embeddings_dir)
        info = reader.get_info()
        
        logger.info(f"✅ Manifest loaded successfully")
        logger.info(f"   Total templates: {reader.total_templates:,}")
        logger.info(f"   Embedding dimension: {info['manifest']['embedding_dimension']}")
        logger.info(f"   Shard count: {len(reader.shard_files)}")
        
        if reader.total_templates == 0:
            logger.error("❌ Total templates is 0 - manifest reading failed!")
            return False
        
        if reader.total_templates > 1_000_000:
            logger.info(f"✅ Correct count detected: {reader.total_templates:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Manifest reading failed: {e}")
        return False


def test_counting(embeddings_dir: str):
    """Test that embedding counting works."""
    from src.embeddings.shard_reader import EmbeddingShardReader
    
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Embedding Counting")
    logger.info("=" * 60)
    
    try:
        reader = EmbeddingShardReader(embeddings_dir)
        
        logger.info("Counting embeddings...")
        count = reader.count_embeddings()
        
        logger.info(f"✅ Counted {count:,} embeddings")
        
        if count != reader.total_templates and reader.total_templates > 0:
            logger.warning(f"⚠️  Count mismatch: {count:,} vs {reader.total_templates:,}")
        else:
            logger.info(f"✅ Count matches manifest")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Counting failed: {e}")
        return False


def test_validation(embeddings_dir: str, sample_size: int = 1000):
    """Test validation with sample."""
    from src.embeddings.shard_reader import EmbeddingShardReader
    
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Validation (Sample)")
    logger.info("=" * 60)
    
    try:
        reader = EmbeddingShardReader(embeddings_dir)
        
        logger.info(f"Validating first {sample_size:,} embeddings...")
        
        count = 0
        wrong_dims = 0
        expected_dim = reader.manifest["embedding_dimension"]
        
        for template_id, embedding, _ in reader.iter_embeddings():
            if len(embedding) != expected_dim:
                wrong_dims += 1
                logger.error(f"  Wrong dimension: {template_id} has {len(embedding)}, expected {expected_dim}")
            
            count += 1
            if count >= sample_size:
                break
        
        logger.info(f"✅ Validated {count:,} embeddings")
        logger.info(f"   Wrong dimensions: {wrong_dims}")
        
        if wrong_dims == 0:
            logger.info(f"✅ All sampled embeddings have correct dimension")
        else:
            logger.error(f"❌ Found {wrong_dims} embeddings with wrong dimensions")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify LAR-RAG implementation hardening"
    )
    
    parser.add_argument(
        "--embeddings-dir",
        default="data/processed/bgl/template_embeddings.jsonl",
        help="Directory containing embeddings"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of embeddings to validate (default: 1000)"
    )
    
    args = parser.parse_args()
    
    logger.info("LAR-RAG Implementation Hardening Verification")
    logger.info("=" * 60)
    
    # Run tests
    results = {}
    
    results['manifest'] = test_manifest_reading(args.embeddings_dir)
    results['counting'] = test_counting(args.embeddings_dir)
    results['validation'] = test_validation(args.embeddings_dir, args.sample_size)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
