"""
Search-related API routes.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field

from config.config import settings
from src.text_analyzer.embedder import TextEmbedder, BGEEmbedder
from src.document_store.document_db import DocumentStore
from src.document_store.vector_db import VectorDBManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
document_store = DocumentStore()
vector_db = VectorDBManager()

# Initialize embedder
if "bge" in settings.EMBEDDING_MODEL.lower():
    embedder = BGEEmbedder(model_name=settings.EMBEDDING_MODEL)
else:
    embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL)

# Pydantic models for request/response schemas
class SearchResult(BaseModel):
    """Search result schema."""
    document_id: str
    chunk_id: Optional[str] = None
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    """Search response schema."""
    query: str
    results: List[SearchResult]
    total: int

class DocumentMetadata(BaseModel):
    """Document metadata schema."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    page_count: Optional[int] = None

@router.get("/", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1),
    document_ids: Optional[str] = Query(None, description="Comma-separated list of document IDs to search within"),
    top_k: int = Query(5, ge=1, le=50),
    min_score: float = Query(0.2, ge=0.0, le=1.0)
):
    """
    Search for documents using semantic search.
    
    Args:
        query: Search query
        document_ids: Optional comma-separated list of document IDs to search within
        top_k: Number of results to return
        min_score: Minimum similarity score
        
    Returns:
        Search results
    """
    try:
        # Generate query embedding
        query_embedding = None
        if isinstance(embedder, BGEEmbedder):
            query_embedding = embedder.embed_query(query)
        else:
            query_embedding = embedder.embed_text(query)
        
        # Search vector database
        search_results = vector_db.search(query_embedding, top_k=top_k * 2)  # Get extra results for filtering
        
        # Filter by document IDs if provided
        if document_ids:
            doc_ids_list = [doc_id.strip() for doc_id in document_ids.split(",")]
            search_results = [
                result for result in search_results
                if result.get("document_id") in doc_ids_list
            ]
        
        # Filter by score
        search_results = [
            result for result in search_results
            if result.get("score", 0) >= min_score
        ]
        
        # Limit to top_k
        search_results = search_results[:top_k]
        
        # Format response
        formatted_results = []
        for result in search_results:
            # Get document metadata
            document = None
            try:
                document = document_store.get_document(result.get("document_id"))
            except Exception as e:
                logger.warning(f"Error getting document {result.get('document_id')}: {e}")
            
            # Create result object
            formatted_result = {
                "document_id": result.get("document_id"),
                "chunk_id": result.get("id"),
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "metadata": {}
            }
            
            # Add document metadata
            if document:
                formatted_result["metadata"] = document.get("metadata", {})
                
                # Add filename
                formatted_result["metadata"]["filename"] = document.get("filename", "")
            
            formatted_results.append(formatted_result)
        
        return {
            "query": query,
            "results": formatted_results,
            "total": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )

@router.get("/related", response_model=SearchResponse)
async def find_related_documents(
    document_id: str = Query(...),
    chunk_id: Optional[str] = Query(None),
    top_k: int = Query(5, ge=1, le=50),
    exclude_same_document: bool = Query(True)
):
    """
    Find related documents or chunks to a given document or chunk.
    
    Args:
        document_id: Document ID
        chunk_id: Optional chunk ID
        top_k: Number of results to return
        exclude_same_document: Whether to exclude chunks from the same document
        
    Returns:
        Related document chunks
    """
    try:
        # Get document
        document = document_store.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Get text to use as query
        query_text = ""
        
        if chunk_id:
            # Find specific chunk
            chunks = document.get("chunks", [])
            for chunk in chunks:
                if chunk.get("chunk_id") == chunk_id or str(chunk.get("chunk_id")) == chunk_id:
                    query_text = chunk.get("text", "")
                    break
            
            if not query_text:
                raise HTTPException(
                    status_code=404,
                    detail=f"Chunk {chunk_id} not found in document {document_id}"
                )
        else:
            # Use document summary or first chunk
            if len(document.get("text", "")) > 0:
                query_text = document.get("text", "")[:1000]  # Use first 1000 chars
            elif document.get("chunks", []):
                query_text = document.get("chunks", [])[0].get("text", "")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {document_id} has no text content"
                )
        
        # Generate query embedding
        query_embedding = embedder.embed_text(query_text)
        
        # Search vector database
        search_results = vector_db.search(query_embedding, top_k=top_k * 2)  # Get extra results for filtering
        
        # Filter out the original document if requested
        if exclude_same_document:
            search_results = [
                result for result in search_results
                if result.get("document_id") != document_id
            ]
        
        # Limit to top_k
        search_results = search_results[:top_k]
        
        # Format response
        formatted_results = []
        for result in search_results:
            # Get document metadata
            related_doc = None
            try:
                related_doc = document_store.get_document(result.get("document_id"))
            except Exception as e:
                logger.warning(f"Error getting document {result.get('document_id')}: {e}")
            
            # Create result object
            formatted_result = {
                "document_id": result.get("document_id"),
                "chunk_id": result.get("id"),
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "metadata": {}
            }
            
            # Add document metadata
            if related_doc:
                formatted_result["metadata"] = related_doc.get("metadata", {})
                
                # Add filename
                formatted_result["metadata"]["filename"] = related_doc.get("filename", "")
            
            formatted_results.append(formatted_result)
        
        return {
            "query": query_text[:100] + "...",  # Truncate query for readability
            "results": formatted_results,
            "total": len(formatted_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding related documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error finding related documents: {str(e)}"
        )

@router.get("/keywords", response_model=Dict[str, Any])
async def extract_keywords(
    query: str = Query(..., min_length=1)
):
    """
    Extract keywords from a query to help with search.
    
    Args:
        query: Search query
        
    Returns:
        Extracted keywords and suggestions
    """
    try:
        # Import keyword extractor
        from src.text_analyzer.keyword_extractor import KeywordExtractor
        
        # Extract keywords
        keyword_extractor = KeywordExtractor(max_keywords=5)
        keywords = keyword_extractor.extract_keywords_tfidf(query, top_n=5)
        
        # Extract keyphrases
        keyphrases = keyword_extractor.extract_keyphrases(query, top_n=3)
        
        # Format results
        result = {
            "original_query": query,
            "keywords": [{"text": k, "score": float(s)} for k, s in keywords],
            "keyphrases": [{"text": k, "score": float(s)} for k, s in keyphrases],
            "suggestions": []
        }
        
        # Generate search suggestions
        if len(keywords) > 1:
            suggestions = []
            
            # Combination of top keywords
            top_keywords = [k for k, _ in keywords[:3]]
            suggestions.append(" ".join(top_keywords))
            
            # Top keyphrase
            if keyphrases:
                suggestions.append(keyphrases[0][0])
            
            result["suggestions"] = suggestions
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting keywords: {str(e)}"
        )
