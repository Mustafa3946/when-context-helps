"""
Build FAISS index from template embeddings.

Usage:
    python scripts/build_faiss_index.py --embeddings-dir data/processed/bgl/template_embeddings.jsonl --out data/index/bgl_faiss
    
    python scripts/build_faiss_index.py --embeddings-dir data/processed/bgl/template_embeddings.jsonl --out data/index/bgl_faiss --index-type HNSW --normalize
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.faiss_index import FAISSIndexBuilder
from src.embeddings.shard_reader import EmbeddingShardReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from template embeddings"
    )
    
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory containing template embeddings shards and manifest"
    )
    
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for FAISS index"
    )
    
    parser.add_argument(
        "--index-type",
        choices=["Flat", "IVF", "HNSW"],
        default="Flat",
        help="FAISS index type (default: Flat for exact search)"
    )
    
    parser.add_argument(
        "--metric",
        choices=["L2", "IP"],
        default="L2",
        help="Distance metric: L2 (Euclidean) or IP (inner product)"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings for cosine similarity (recommended)"
    )
    
    parser.add_argument(
        "--nlist",
        type=int,
        default=100,
        help="Number of clusters for IVF index (default: 100)"
    )
    
    parser.add_argument(
        "--m",
        type=int,
        default=16,
        help="Number of connections for HNSW index (default: 16)"
    )
    
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=40,
        help="HNSW construction parameter (default: 40)"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip index verification (faster, not recommended)"
    )
    
    args = parser.parse_args()
    
    # Validate embeddings directory
    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        sys.exit(1)
    
    # Check manifest
    manifest_path = embeddings_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    
    # Validate embeddings
    logger.info("Validating embeddings...")
    try:
        reader = EmbeddingShardReader(str(embeddings_dir))
        validation = reader.validate(verbose=True)
        
        if not validation["valid"]:
            logger.error("Embeddings validation failed!")
            logger.error(f"Validation report: {validation}")
            sys.exit(1)
        
        logger.info("✅ Embeddings validation passed")
        
    except Exception as e:
        logger.error(f"Error validating embeddings: {e}")
        sys.exit(1)
    
    # Build index
    try:
        builder = FAISSIndexBuilder(
            embeddings_dir=str(embeddings_dir),
            output_dir=args.out,
            index_type=args.index_type,
            metric=args.metric,
            nlist=args.nlist,
            m=args.m,
            ef_construction=args.ef_construction
        )
        
        metadata = builder.build(
            normalize=args.normalize,
            verify=not args.no_verify
        )
        
        logger.info("=" * 60)
        logger.info("FAISS Index Build Summary")
        logger.info("=" * 60)
        logger.info(f"Index type: {metadata['index_type']}")
        logger.info(f"Metric: {metadata['metric']}")
        logger.info(f"Total vectors: {metadata['total_vectors']}")
        logger.info(f"Dimension: {metadata['embedding_dimension']}")
        logger.info(f"Normalized: {metadata['normalized']}")
        logger.info(f"Checksum: {metadata['checksum'][:16]}...")
        logger.info(f"Output: {args.out}")
        logger.info("=" * 60)
        logger.info("✅ Index build complete!")
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
