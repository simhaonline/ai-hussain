"""
Document-related API routes.
"""
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.config import settings
from src.pdf_processor.extractor import PDFExtractor
from src.text_analyzer.embedder import TextEmbedder, BGEEmbedder
from src.document_store.document_db import DocumentStore
from src.document_store.vector_db import VectorDBManager
from src.model_manager.summarizer import Summarizer

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
document_store = DocumentStore()
vector_db = VectorDBManager()
pdf_extractor = PDFExtractor()

# Initialize embedder
if "bge" in settings.EMBEDDING_MODEL.lower():
    embedder = BGEEmbedder(model_name=settings.EMBEDDING_MODEL)
else:
    embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL)

# Pydantic models for request/response schemas
class DocumentMetadata(BaseModel):
    """Document metadata schema."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: Optional[int] = None

class DocumentResponse(BaseModel):
    """Document response schema."""
    id: str
    filename: str
    metadata: DocumentMetadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processing_status: str = "complete"
    chunk_count: Optional[int] = None

class DocumentList(BaseModel):
    """Document list response schema."""
    documents: List[DocumentResponse]
    total: int
    page: int
    limit: int

class DocumentSummaryResponse(BaseModel):
    """Document summary response schema."""
    document_id: str
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessingStatus(BaseModel):
    """Processing status response schema."""
    document_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[float] = None

# Background processing function
def process_pdf_background(file_path: str, document_id: str, filename: str):
    """
    Process a PDF file in the background.
    
    Args:
        file_path: Path to the temporary PDF file
        document_id: Document ID
        filename: Original filename
    """
    try:
        # Extract text from PDF
        result = pdf_extractor.process_pdf(Path(file_path))
        
        # Set the document ID and filename
        result["id"] = document_id
        result["filename"] = filename
        
        # Store in document database
        document_store.add_document(result)
        
        # Generate embeddings for chunks
        chunks = result.get("chunks", [])
        if chunks:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = embedder.embed_texts(texts)
            
            # Store in vector database
            vector_db.add_documents(
                document_id=document_id,
                texts=texts,
                embeddings=embeddings,
                metadata=result.get("metadata", {})
            )
        
        # Update processing status
        document_store.update_document(
            document_id=document_id,
            updates={"processing_status": "complete"}
        )
        
        logger.info(f"Completed processing document {document_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        
        # Update processing status
        document_store.update_document(
            document_id=document_id,
            updates={
                "processing_status": "failed",
                "processing_error": str(e)
            }
        )
    
    finally:
        # Clean up temporary file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file {file_path}: {e}")

@router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_async: bool = Form(True)
):
    """
    Upload and process a PDF document.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded PDF file
        process_async: Whether to process the document asynchronously
        
    Returns:
        Document information
    """
    # Check file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            
            # Save the uploaded file
            content = await file.read()
            temp_file.write(content)
        
        # Create initial document entry
        from uuid import uuid4
        document_id = str(uuid4())
        
        document = {
            "id": document_id,
            "filename": file.filename,
            "processing_status": "pending" if process_async else "processing",
            "metadata": {}
        }
        
        # Add to document store
        document_store.add_document(document)
        
        if process_async:
            # Process in background
            background_tasks.add_task(
                process_pdf_background,
                temp_file_path,
                document_id,
                file.filename
            )
        else:
            # Process synchronously
            try:
                # Extract text from PDF
                result = pdf_extractor.process_pdf(Path(temp_file_path))
                
                # Set the document ID and filename
                result["id"] = document_id
                result["filename"] = file.filename
                
                # Store in document database
                document_store.update_document(document_id, result)
                
                # Generate embeddings for chunks
                chunks = result.get("chunks", [])
                if chunks:
                    texts = [chunk["text"] for chunk in chunks]
                    embeddings = embedder.embed_texts(texts)
                    
                    # Store in vector database
                    vector_db.add_documents(
                        document_id=document_id,
                        texts=texts,
                        embeddings=embeddings,
                        metadata=result.get("metadata", {})
                    )
                
                # Update document with metadata and status
                document = result
                document["processing_status"] = "complete"
                
            except Exception as e:
                logger.error(f"Error processing document {document_id}: {e}")
                
                # Update document with error status
                document["processing_status"] = "failed"
                document["processing_error"] = str(e)
                
                document_store.update_document(
                    document_id=document_id,
                    updates={
                        "processing_status": "failed",
                        "processing_error": str(e)
                    }
                )
            
            finally:
                # Clean up temporary file
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file {temp_file_path}: {e}")
        
        # Prepare response
        response = {
            "id": document_id,
            "filename": file.filename,
            "metadata": document.get("metadata", {}),
            "created_at": document.get("created_at"),
            "updated_at": document.get("updated_at"),
            "processing_status": document.get("processing_status", "pending"),
            "chunk_count": len(document.get("chunks", []))
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )

@router.get("/", response_model=DocumentList)
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    """
    List all documents.
    
    Args:
        page: Page number
        limit: Items per page
        
    Returns:
        List of documents with pagination
    """
    try:
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get documents
        documents = document_store.list_documents(
            filter_dict=None,
            limit=limit,
            offset=offset
        )
        
        # Get total count (in a real implementation, you might want to optimize this)
        total = len(document_store.list_documents(limit=1000000))
        
        # Format response
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "id": doc.get("id"),
                "filename": doc.get("filename", ""),
                "metadata": doc.get("metadata", {}),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "processing_status": doc.get("processing_status", "complete"),
                "chunk_count": len(doc.get("chunks", []))
            })
        
        return {
            "documents": formatted_docs,
            "total": total,
            "page": page,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    Get document by ID.
    
    Args:
        document_id: Document ID
        
    Returns:
        Document information
    """
    try:
        # Get document
        document = document_store.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Format response
        response = {
            "id": document.get("id"),
            "filename": document.get("filename", ""),
            "metadata": document.get("metadata", {}),
            "created_at": document.get("created_at"),
            "updated_at": document.get("updated_at"),
            "processing_status": document.get("processing_status", "complete"),
            "chunk_count": len(document.get("chunks", []))
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting document: {str(e)}"
        )

@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str):
    """
    Delete document by ID.
    
    Args:
        document_id: Document ID
    """
    try:
        # Check if document exists
        document = document_store.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Delete from document store
        document_store.delete_document(document_id)
        
        # Delete from vector database
        vector_db.delete_document(document_id)
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

@router.get("/{document_id}/summary", response_model=DocumentSummaryResponse)
async def get_document_summary(document_id: str):
    """
    Get a summary of a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        Document summary
    """
    try:
        # Get document
        document = document_store.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Generate summary
        summarizer = Summarizer()
        summary_result = summarizer.summarize_document(document)
        
        return summary_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary for document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )

@router.get("/{document_id}/status", response_model=ProcessingStatus)
async def get_processing_status(document_id: str):
    """
    Get the processing status of a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        Processing status
    """
    try:
        # Get document
        document = document_store.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Get status
        status = document.get("processing_status", "complete")
        message = document.get("processing_error", None)
        progress = None
        
        if status == "processing":
            # Estimate progress (in a real implementation, you would track this)
            progress = 0.5
        
        return {
            "document_id": document_id,
            "status": status,
            "message": message,
            "progress": progress
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting processing status: {str(e)}"
        )
