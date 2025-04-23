"""
Vector database management for storing and retrieving document embeddings.
"""
import logging
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import faiss
import uuid

from config.config import settings

logger = logging.getLogger(__name__)

class VectorDBManager:
    """Class to manage vector database operations."""
    
    def __init__(self, vector_db_type: str = settings.VECTOR_DB_TYPE):
        """
        Initialize the vector database manager.
        
        Args:
            vector_db_type: Type of vector database to use (faiss, qdrant, milvus)
        """
        self.vector_db_type = vector_db_type
        self.vector_db = self._initialize_vector_db()
    
    def _initialize_vector_db(self) -> Any:
        """
        Initialize the vector database based on the configured type.
        
        Returns:
            Initialized vector database client
        """
        if self.vector_db_type == "faiss":
            return FAISSVectorDB()
        elif self.vector_db_type == "qdrant":
            return QdrantVectorDB()
        elif self.vector_db_type == "milvus":
            return MilvusVectorDB()
        else:
            logger.warning(f"Unknown vector DB type: {self.vector_db_type}, defaulting to FAISS")
            return FAISSVectorDB()
    
    def add_documents(self, document_id: str, texts: List[str], 
                     embeddings: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add document chunks with embeddings to the vector database.
        
        Args:
            document_id: Document identifier
            texts: List of text chunks
            embeddings: NumPy array of embeddings
            metadata: Document metadata
            
        Returns:
            List of inserted chunk IDs
        """
        return self.vector_db.add_documents(document_id, texts, embeddings, metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embedding.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text, score, and metadata
        """
        return self.vector_db.search(query_embedding, top_k)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        return self.vector_db.delete_document(document_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with statistics
        """
        return self.vector_db.get_stats()


class FAISSVectorDB:
    """FAISS implementation of vector database."""
    
    def __init__(self, index_path: Optional[str] = None, 
                dimension: int = settings.EMBEDDING_DIMENSION):
        """
        Initialize FAISS vector database.
        
        Args:
            index_path: Path to load existing index from
            dimension: Dimension of embeddings
        """
        self.dimension = dimension
        self.index_path = index_path or os.path.join(settings.PROCESSED_DATA_DIR, "faiss_index")
        self.metadata_path = os.path.join(settings.PROCESSED_DATA_DIR, "faiss_metadata.json")
        
        # Initialize index
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
        
        # Internal tracking
        self.next_id = len(self.metadata)
        
        logger.info(f"FAISS index initialized with {self.next_id} vectors")
    
    def _load_or_create_index(self) -> faiss.Index:
        """
        Load existing index or create a new one.
        
        Returns:
            FAISS index
        """
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Loading FAISS index from {self.index_path}")
                return faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
        
        logger.info(f"Creating new FAISS index with dimension {self.dimension}")
        index = faiss.IndexFlatL2(self.dimension)  # L2 distance
        return index
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load metadata for indexed vectors.
        
        Returns:
            List of metadata dictionaries
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        return []
    
    def _save_metadata(self):
        """Save metadata to disk."""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _save_index(self):
        """Save FAISS index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
    
    def add_documents(self, document_id: str, texts: List[str], 
                     embeddings: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add document chunks with embeddings to FAISS.
        
        Args:
            document_id: Document identifier
            texts: List of text chunks
            embeddings: NumPy array of embeddings
            metadata: Document metadata
            
        Returns:
            List of inserted chunk IDs
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError(f"Number of texts ({len(texts)}) does not match number of embeddings ({embeddings.shape[0]})")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        chunk_ids = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Generate unique ID for this chunk
            chunk_id = f"{document_id}_{i}"
            chunk_ids.append(chunk_id)
            
            # Add to index
            self.index.add(embedding.reshape(1, -1))
            
            # Store metadata
            chunk_metadata = {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_index": i,
                "text": text
            }
            
            # Add document metadata if provided
            if metadata:
                chunk_metadata["metadata"] = metadata
            
            self.metadata.append(chunk_metadata)
            self.next_id += 1
        
        # Save index and metadata
        self._save_index()
        self._save_metadata()
        
        logger.info(f"Added {len(texts)} chunks for document {document_id} to FAISS index")
        return chunk_ids
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embedding.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with text, score, and metadata
        """
        # Ensure embedding is float32 and properly shaped
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    "id": self.metadata[idx]["id"],
                    "document_id": self.metadata[idx]["document_id"],
                    "chunk_index": self.metadata[idx]["chunk_index"],
                    "text": self.metadata[idx]["text"],
                    "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "distance": float(distance)
                }
                
                # Add metadata if available
                if "metadata" in self.metadata[idx]:
                    result["metadata"] = self.metadata[idx]["metadata"]
                
                results.append(result)
        
        return results
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document from FAISS.
        
        Note: FAISS doesn't support direct deletion, so we create a new index without the deleted chunks.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        # Find all chunks for this document
        doc_indices = [i for i, meta in enumerate(self.metadata) if meta["document_id"] == document_id]
        
        if not doc_indices:
            logger.warning(f"No chunks found for document {document_id}")
            return False
        
        # Create a new index and metadata without the deleted document
        new_index = faiss.IndexFlatL2(self.dimension)
        new_metadata = []
        
        for i, meta in enumerate(self.metadata):
            if meta["document_id"] != document_id:
                # Get embedding from original index
                embedding = np.zeros((1, self.dimension), dtype=np.float32)
                self.index.reconstruct(i, embedding.reshape(-1))
                
                # Add to new index
                new_index.add(embedding)
                new_metadata.append(meta)
        
        # Replace index and metadata
        self.index = new_index
        self.metadata = new_metadata
        self.next_id = len(self.metadata)
        
        # Save new index and metadata
        self._save_index()
        self._save_metadata()
        
        logger.info(f"Deleted {len(doc_indices)} chunks for document {document_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "type": "faiss",
            "vector_count": self.index.ntotal,
            "dimension": self.dimension,
            "document_count": len(set(meta["document_id"] for meta in self.metadata))
        }


class QdrantVectorDB:
    """Qdrant implementation of vector database."""
    
    def __init__(self, collection_name: str = settings.QDRANT_COLLECTION,
                host: str = settings.QDRANT_HOST,
                port: int = settings.QDRANT_PORT,
                dimension: int = settings.EMBEDDING
