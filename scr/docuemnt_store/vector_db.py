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
                dimension: int = settings.EMBEDDING_DIMENSION):
        """
        Initialize Qdrant vector database.
        
        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant server host
            port: Qdrant server port
            dimension: Dimension of embeddings
        """
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Import Qdrant client
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            self.models = models
            self.client = QdrantClient(host=host, port=port)
            
            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection '{collection_name}'")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE
                    )
                )
            
            logger.info(f"Connected to Qdrant collection '{collection_name}'")
            
        except ImportError:
            logger.error("Qdrant client not installed. Install with 'pip install qdrant-client'")
            raise
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
    
    def add_documents(self, document_id: str, texts: List[str], 
                     embeddings: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add document chunks with embeddings to Qdrant.
        
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
        
        chunk_ids = []
        points = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Generate unique ID for this chunk
            chunk_id = f"{document_id}_{i}"
            chunk_ids.append(chunk_id)
            
            # Create point
            point_id = str(uuid.uuid4())
            
            payload = {
                "text": text,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": i
            }
            
            # Add document metadata if provided
            if metadata:
                for key, value in metadata.items():
                    payload[f"metadata_{key}"] = value
            
            points.append(
                self.models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        
        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Added {len(texts)} chunks for document {document_id} to Qdrant")
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
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        results = []
        for scored_point in search_result:
            payload = scored_point.payload
            
            # Extract metadata from payload
            metadata = {}
            for key, value in payload.items():
                if key.startswith("metadata_"):
                    metadata[key[9:]] = value
            
            result = {
                "id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "score": scored_point.score,
                "metadata": metadata
            }
            
            results.append(result)
        
        return results
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document from Qdrant.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            # Delete by filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.models.FilterSelector(
                    filter=self.models.Filter(
                        must=[
                            self.models.FieldCondition(
                                key="document_id",
                                match=self.models.MatchValue(
                                    value=document_id
                                )
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted document {document_id} from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id} from Qdrant: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Qdrant collection.
        
        Returns:
            Dictionary with statistics
        """
        collection_info = self.client.get_collection(self.collection_name)
        
        return {
            "type": "qdrant",
            "collection": self.collection_name,
            "vector_count": collection_info.vectors_count,
            "dimension": self.dimension,
            "indexed_vectors_count": collection_info.indexed_vectors_count
        }


class MilvusVectorDB:
    """Milvus implementation of vector database."""
    
    def __init__(self, collection_name: str = "pdf_documents",
                host: str = "localhost",
                port: int = 19530,
                dimension: int = settings.EMBEDDING_DIMENSION):
        """
        Initialize Milvus vector database.
        
        Args:
            collection_name: Name of the Milvus collection
            host: Milvus server host
            port: Milvus server port
            dimension: Dimension of embeddings
        """
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Import Milvus client
        try:
            from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
            self.pymilvus = __import__('pymilvus')
            
            # Connect to Milvus
            connections.connect(host=host, port=port)
            
            # Check if collection exists, create if not
            if not utility.has_collection(collection_name):
                logger.info(f"Creating Milvus collection '{collection_name}'")
                
                # Define fields
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
                ]
                
                schema = CollectionSchema(fields)
                
                # Create collection
                self.collection = Collection(name=collection_name, schema=schema)
                
                # Create index
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                }
                self.collection.create_index("embedding", index_params)
            else:
                self.collection = Collection(name=collection_name)
                self.collection.load()
            
            logger.info(f"Connected to Milvus collection '{collection_name}'")
            
        except ImportError:
            logger.error("Milvus client not installed. Install with 'pip install pymilvus'")
            raise
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            raise
    
    def add_documents(self, document_id: str, texts: List[str], 
                     embeddings: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add document chunks with embeddings to Milvus.
        
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
        
        chunk_ids = []
        
        # Prepare data
        ids = []
        document_ids = []
        chunk_indices = []
        text_list = []
        embedding_list = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Generate unique ID for this chunk
            chunk_id = f"{document_id}_{i}"
            chunk_ids.append(chunk_id)
            
            ids.append(chunk_id)
            document_ids.append(document_id)
            chunk_indices.append(i)
            text_list.append(text)
            embedding_list.append(embedding.tolist())
        
        # Insert data
        data = [
            ids,
            document_ids,
            chunk_indices,
            text_list,
            embedding_list
        ]
        
        try:
            self.collection.insert(data)
            
            # If metadata provided, store it separately 
            # (Milvus doesn't directly support complex metadata)
            if metadata:
                # In a production system, you would store this in a separate database
                # For simplicity in this example, we're skipping actual metadata storage
                pass
            
            logger.info(f"Added {len(texts)} chunks for document {document_id} to Milvus")
            
        except Exception as e:
            logger.error(f"Error inserting into Milvus: {e}")
            raise
        
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
        # Prepare search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        # Execute search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["document_id", "chunk_index", "text"]
        )
        
        # Format results
        formatted_results = []
        
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "document_id": hit.entity.get('document_id'),
                    "chunk_index": hit.entity.get('chunk_index'),
                    "text": hit.entity.get('text'),
                    "score": hit.score,
                    "metadata": {}  # In production, you would fetch metadata from separate storage
                }
                
                formatted_results.append(result)
        
        return formatted_results
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document from Milvus.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            # Execute delete query
            expr = f'document_id == "{document_id}"'
            self.collection.delete(expr)
            
            logger.info(f"Deleted document {document_id} from Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id} from Milvus: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Milvus collection.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.collection.num_entities
        
        return {
            "type": "milvus",
            "collection": self.collection_name,
            "vector_count": stats,
            "dimension": self.dimension
        }
