"""
Document database management for storing document metadata and content.
"""
import logging
import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid
from datetime import datetime

from config.config import settings

logger = logging.getLogger(__name__)

class DocumentStore:
    """Class to manage document storage operations."""
    
    def __init__(self, db_type: str = "mongodb"):
        """
        Initialize the document store.
        
        Args:
            db_type: Type of database to use (mongodb, json)
        """
        self.db_type = db_type
        self.db = self._initialize_db()
    
    def _initialize_db(self) -> Any:
        """
        Initialize the database based on the configured type.
        
        Returns:
            Initialized database client
        """
        if self.db_type == "mongodb":
            return MongoDBDocumentStore()
        else:
            return JSONDocumentStore()
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """
        Add document to the store.
        
        Args:
            document: Document dictionary
            
        Returns:
            Document ID
        """
        return self.db.add_document(document)
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        return self.db.get_document(document_id)
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update document.
        
        Args:
            document_id: Document identifier
            updates: Dictionary of updates
            
        Returns:
            True if successful
        """
        return self.db.update_document(document_id, updates)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        return self.db.delete_document(document_id)
    
    def list_documents(self, filter_dict: Optional[Dict[str, Any]] = None, 
                       limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List documents with optional filtering.
        
        Args:
            filter_dict: Dictionary of filter criteria
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document dictionaries
        """
        return self.db.list_documents(filter_dict, limit, offset)


class MongoDBDocumentStore:
    """MongoDB implementation of document store."""
    
    def __init__(self, uri: str = settings.MONGODB_URI,
                db_name: str = settings.MONGODB_DB,
                collection_name: str = settings.MONGODB_COLLECTION):
        """
        Initialize MongoDB document store.
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        
        # Initialize MongoDB client
        try:
            from pymongo import MongoClient
            
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            
            logger.info(f"Connected to MongoDB collection '{collection_name}'")
            
        except ImportError:
            logger.error("MongoDB client not installed. Install with 'pip install pymongo'")
            raise
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """
        Add document to MongoDB.
        
        Args:
            document: Document dictionary
            
        Returns:
            Document ID
        """
        # Ensure document has an ID
        if "id" not in document:
            document["id"] = str(uuid.uuid4())
        
        # Add timestamps
        document["created_at"] = datetime.utcnow().isoformat()
        document["updated_at"] = document["created_at"]
        
        try:
            result = self.collection.insert_one(document)
            logger.info(f"Added document {document['id']} to MongoDB")
            return document["id"]
            
        except Exception as e:
            logger.error(f"Error adding document to MongoDB: {e}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID from MongoDB.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        try:
            document = self.collection.find_one({"id": document_id})
            
            if document:
                # Remove MongoDB's _id field
                document.pop("_id", None)
                return document
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {document_id} from MongoDB: {e}")
            return None
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update document in MongoDB.
        
        Args:
            document_id: Document identifier
            updates: Dictionary of updates
            
        Returns:
            True if successful
        """
        try:
            # Add updated timestamp
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.collection.update_one(
                {"id": document_id},
                {"$set": updates}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated document {document_id} in MongoDB")
            else:
                logger.warning(f"Document {document_id} not found in MongoDB")
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating document {document_id} in MongoDB: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document from MongoDB.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            result = self.collection.delete_one({"id": document_id})
            
            success = result.deleted_count > 0
            if success:
                logger.info(f"Deleted document {document_id} from MongoDB")
            else:
                logger.warning(f"Document {document_id} not found in MongoDB")
                
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id} from MongoDB: {e}")
            return False
    
    def list_documents(self, filter_dict: Optional[Dict[str, Any]] = None, 
                       limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List documents from MongoDB with optional filtering.
        
        Args:
            filter_dict: Dictionary of filter criteria
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document dictionaries
        """
        try:
            query = filter_dict or {}
            
            cursor = self.collection.find(query).skip(offset).limit(limit)
            
            documents = []
            for doc in cursor:
                # Remove MongoDB's _id field
                doc.pop("_id", None)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents from MongoDB: {e}")
            return []


class JSONDocumentStore:
    """JSON file-based implementation of document store."""
    
    def __init__(self, data_dir: Path = settings.PROCESSED_DATA_DIR):
        """
        Initialize JSON document store.
        
        Args:
            data_dir: Directory to store JSON files
        """
        self.data_dir = data_dir
        self.index_file = data_dir / "document_index.json"
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load or create index
        self.index = self._load_or_create_index()
    
    def _load_or_create_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load existing index or create a new one.
        
        Returns:
            Document index dictionary
        """
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading document index: {e}")
        
        return {}
    
    def _save_index(self):
        """Save index to disk."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
    
    def _get_document_path(self, document_id: str) -> Path:
        """
        Get file path for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Path to document file
        """
        return self.data_dir / f"{document_id}.json"
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """
        Add document to JSON store.
        
        Args:
            document: Document dictionary
            
        Returns:
            Document ID
        """
        # Ensure document has an ID
        if "id" not in document:
            document["id"] = str(uuid.uuid4())
        
        document_id = document["id"]
        
        # Add timestamps
        document["created_at"] = datetime.utcnow().isoformat()
        document["updated_at"] = document["created_at"]
        
        # Save document
        document_path = self._get_document_path(document_id)
        with open(document_path, "w", encoding="utf-8") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        
        # Update index
        self.index[document_id] = {
            "id": document_id,
            "filename": document.get("filename", ""),
            "title": document.get("metadata", {}).get("title", ""),
            "created_at": document["created_at"],
            "updated_at": document["updated_at"],
            "path": str(document_path)
        }
        
        self._save_index()
        
        logger.info(f"Added document {document_id} to JSON store")
        return document_id
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID from JSON store.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        if document_id not in self.index:
            return None
        
        document_path = self._get_document_path(document_id)
        
        if not os.path.exists(document_path):
            return None
        
        try:
            with open(document_path, "r", encoding="utf-8") as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading document {document_id}: {e}")
            return None
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update document in JSON store.
        
        Args:
            document_id: Document identifier
            updates: Dictionary of updates
            
        Returns:
            True if successful
        """
        document = self.get_document(document_id)
        
        if not document:
            logger.warning(f"Document {document_id} not found in JSON store")
            return False
        
        # Update document
        document.update(updates)
        document["updated_at"] = datetime.utcnow().isoformat()
        
        # Save document
        document_path = self._get_document_path(document_id)
        with open(document_path, "w", encoding="utf-8") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        
        # Update index
        if document_id in self.index:
            self.index[document_id]["updated_at"] = document["updated_at"]
            
            if "metadata" in updates and "title" in updates["metadata"]:
                self.index[document_id]["title"] = updates["metadata"]["title"]
                
            self._save_index()
        
        logger.info(f"Updated document {document_id} in JSON store")
        return True
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document from JSON store.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        if document_id not in self.index:
            logger.warning(f"Document {document_id} not found in JSON store")
            return False
        
        # Delete document file
        document_path = self._get_document_path(document_id)
        if os.path.exists(document_path):
            os.remove(document_path)
        
        # Update index
        del self.index[document_id]
        self._save_index()
        
        logger.info(f"Deleted document {document_id} from JSON store")
        return True
    
    def list_documents(self, filter_dict: Optional[Dict[str, Any]] = None, 
                       limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List documents from JSON store with optional filtering.
        
        Args:
            filter_dict: Dictionary of filter criteria
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document dictionaries
        """
        # Get all documents from index
        documents = list(self.index.values())
        
        # Apply filtering if provided
        if filter_dict:
            filtered_docs = []
            for doc in documents:
                match = True
                for key, value in filter_dict.items():
                    if key not in doc or doc[key] != value:
                        match = False
                        break
                
                if match:
                    filtered_docs.append(doc)
            
            documents = filtered_docs
        
        # Apply pagination
        start = min(offset, len(documents))
        end = min(offset + limit, len(documents))
        
        return documents[start:end]
