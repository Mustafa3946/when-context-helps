"""
FAISS Index Builder for Template Embeddings.

Builds a FAISS index from sharded template embeddings for fast vector retrieval.
Supports multiple index types and deterministic builds.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import faiss

from src.embeddings.shard_reader import EmbeddingShardReader

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """
    Build FAISS index from template embeddings.
    
    Supports:
    - Flat (exact) index for small datasets
    - IVF (inverted file) index for large datasets
    - HNSW (hierarchical NSW) index for balanced speed/accuracy
    
    Output structure:
        index_dir/
            index.faiss
            template_ids.npy
            meta.json
    """
    
    def __init__(
        self,
        embeddings_dir: str,
        output_dir: str,
        index_type: str = "Flat",
        metric: str = "L2",
        nlist: int = 100,
        m: int = 16,
        ef_construction: int = 40
    ):
        """
        Initialize FAISS index builder.
        
        Args:
            embeddings_dir: Directory containing template embeddings shards
            output_dir: Directory to save FAISS index
            index_type: Index type - "Flat", "IVF", or "HNSW"
            metric: Distance metric - "L2" or "IP" (inner product)
            nlist: Number of clusters for IVF (default: 100)
            m: Number of connections for HNSW (default: 16)
            ef_construction: HNSW construction parameter (default: 40)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.m = m
        self.ef_construction = ef_construction
        
        # Load embeddings
        self.reader = EmbeddingShardReader(str(self.embeddings_dir))
        self.dimension = self.reader.manifest["embedding_dimension"]
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized FAISSIndexBuilder: type={index_type}, "
            f"metric={metric}, dim={self.dimension}"
        )
    
    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index based on type.
        
        Returns:
            FAISS index object
        """
        if self.metric == "L2":
            metric_type = faiss.METRIC_L2
        elif self.metric == "IP":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        if self.index_type == "Flat":
            # Exact search (brute force)
            index = faiss.IndexFlatL2(self.dimension) if metric_type == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
            logger.info("Created Flat index (exact search)")
            
        elif self.index_type == "IVF":
            # Inverted file index (faster approximate search)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_type)
            logger.info(f"Created IVF index with {self.nlist} clusters")
            
        elif self.index_type == "HNSW":
            # Hierarchical NSW (good speed/accuracy tradeoff)
            index = faiss.IndexHNSWFlat(self.dimension, self.m, metric_type)
            index.hnsw.efConstruction = self.ef_construction
            logger.info(f"Created HNSW index (M={self.m}, ef_construction={self.ef_construction})")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def build(self, normalize: bool = False, verify: bool = True) -> Dict[str, Any]:
        """
        Build FAISS index from embeddings.
        
        Args:
            normalize: L2-normalize embeddings before indexing (for cosine similarity)
            verify: Verify index after building
            
        Returns:
            Build statistics dictionary
        """
        logger.info("Building FAISS index...")
        
        # Create index
        index = self._create_index()
        
        # Collect embeddings and IDs
        embeddings_list = []
        template_ids_list = []
        
        logger.info("Loading embeddings from shards...")
        for template_id, embedding, _ in self.reader.iter_embeddings():
            embeddings_list.append(embedding)
            template_ids_list.append(template_id)
        
        # Convert to numpy arrays
        embeddings = np.array(embeddings_list, dtype=np.float32)
        template_ids = np.array(template_ids_list, dtype=object)
        
        logger.info(f"Loaded {len(embeddings)} embeddings, shape={embeddings.shape}")
        
        # Normalize if needed
        if normalize:
            logger.info("L2-normalizing embeddings for cosine similarity...")
            faiss.normalize_L2(embeddings)
        
        # Train index if needed (IVF requires training)
        if self.index_type == "IVF":
            logger.info(f"Training IVF index on {len(embeddings)} vectors...")
            index.train(embeddings)
            logger.info("Training complete")
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        logger.info(f"Added {index.ntotal} vectors to index")
        
        # Verify index
        if verify:
            logger.info("Verifying index...")
            self._verify_index(index, embeddings[:100])
        
        # Save index
        index_path = self.output_dir / "index.faiss"
        ids_path = self.output_dir / "template_ids.npy"
        meta_path = self.output_dir / "meta.json"
        
        logger.info(f"Saving index to {index_path}...")
        faiss.write_index(index, str(index_path))
        
        logger.info(f"Saving template IDs to {ids_path}...")
        np.save(str(ids_path), template_ids)
        
        # Compute checksum
        checksum = self._compute_checksum(embeddings)
        
        # Save metadata
        metadata = {
            "model": self.reader.manifest.get("model", "unknown"),
            "embedding_dimension": self.dimension,
            "total_vectors": int(index.ntotal),
            "index_type": self.index_type,
            "metric": self.metric,
            "normalized": normalize,
            "checksum": checksum,
            "embeddings_dir": str(self.embeddings_dir),
            "index_params": {
                "nlist": self.nlist if self.index_type == "IVF" else None,
                "m": self.m if self.index_type == "HNSW" else None,
                "ef_construction": self.ef_construction if self.index_type == "HNSW" else None
            }
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {meta_path}")
        logger.info("✅ FAISS index built successfully")
        
        return metadata
    
    def _verify_index(self, index: faiss.Index, sample_vectors: np.ndarray):
        """
        Verify index by searching sample vectors.
        
        Args:
            index: FAISS index
            sample_vectors: Sample embeddings to search
        """
        k = min(5, index.ntotal)
        
        try:
            distances, indices = index.search(sample_vectors, k)
            logger.info(f"Verification search successful: retrieved {len(indices)} results")
        except Exception as e:
            logger.error(f"Index verification failed: {e}")
            raise
    
    def _compute_checksum(self, embeddings: np.ndarray) -> str:
        """
        Compute checksum of embeddings.
        
        Args:
            embeddings: Embedding matrix
            
        Returns:
            SHA256 checksum hex string
        """
        # Use a sample for large datasets
        if len(embeddings) > 10000:
            sample = embeddings[:10000]
        else:
            sample = embeddings
        
        checksum = hashlib.sha256(sample.tobytes()).hexdigest()
        return checksum


class FAISSIndexLoader:
    """
    Load and search FAISS index.
    
    Usage:
        loader = FAISSIndexLoader("data/index/bgl_faiss")
        results = loader.search(query_embedding, k=10)
    """
    
    def __init__(self, index_dir: str):
        """
        Initialize FAISS index loader.
        
        Args:
            index_dir: Directory containing index.faiss, template_ids.npy, meta.json
        """
        self.index_dir = Path(index_dir)
        
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        # Load metadata
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load FAISS index
        index_path = self.index_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        logger.info(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        
        # Load template IDs
        ids_path = self.index_dir / "template_ids.npy"
        if not ids_path.exists():
            raise FileNotFoundError(f"Template IDs file not found: {ids_path}")
        
        self.template_ids = np.load(str(ids_path), allow_pickle=True)
        
        logger.info(
            f"Loaded FAISS index: {self.index.ntotal} vectors, "
            f"type={self.metadata['index_type']}"
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        normalize: bool = None
    ) -> List[Tuple[str, float]]:
        """
        Search index for nearest neighbors.
        
        Args:
            query_embedding: Query vector (1D array)
            k: Number of results to return
            normalize: L2-normalize query (if None, use index metadata)
            
        Returns:
            List of (template_id, distance) tuples
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize if needed
        if normalize is None:
            normalize = self.metadata.get("normalized", False)
        
        if normalize:
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert to list of tuples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # Valid index
                template_id = self.template_ids[idx]
                results.append((template_id, float(dist)))
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get index information.
        
        Returns:
            Dictionary with index metadata
        """
        return {
            "index_dir": str(self.index_dir),
            "metadata": self.metadata,
            "ntotal": self.index.ntotal
        }
