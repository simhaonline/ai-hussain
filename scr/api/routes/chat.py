"""
Chat-related API routes.
"""
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Body, Query, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.config import settings
from src.document_store.document_db import DocumentStore
from src.document_store.vector_db import VectorDBManager
from src.model_manager.rag_model import RAGModel
from src.model_manager.qa_model import QAModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
document_store = DocumentStore()
vector_db = VectorDBManager()
rag_model = RAGModel()
qa_model = QAModel(rag_model=rag_model)

# Pydantic models for request/response schemas
class ChatMessage(BaseModel):
    """Chat message schema."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Chat request schema."""
    messages: List[ChatMessage]
    document_ids: Optional[List[str]] = None
    temperature: float = 0.7
    stream: bool = False
    top_k: int = 5

class SourceInfo(BaseModel):
    """Source information schema."""
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    """Chat response schema."""
    id: str
    message: ChatMessage
    sources: List[SourceInfo] = Field(default_factory=list)

class StreamToken(BaseModel):
    """Streaming token schema."""
    text: str
    type: str = "token"

class StreamSource(BaseModel):
    """Streaming source schema."""
    document_id: str
    text: str
    score: float
    type: str = "source"

class StreamEnd(BaseModel):
    """Streaming end marker schema."""
    type: str = "end"

class QuestionRequest(BaseModel):
    """Question request schema."""
    question: str
    document_ids: Optional[List[str]] = None
    temperature: float = 0.7
    top_k: int = 5

class QuestionResponse(BaseModel):
    """Question response schema."""
    id: str
    question: str
    answer: str
    confidence: float
    sources: List[SourceInfo] = Field(default_factory=list)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with documents using RAG.
    
    Args:
        request: Chat request
        
    Returns:
        Chat response with sources
    """
    try:
        # Check if messages are provided
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="Messages cannot be empty"
            )
        
        # Get the latest user message
        latest_message = None
        for msg in reversed(request.messages):
            if msg.role.lower() == "user":
                latest_message = msg.content
                break
        
        if not latest_message:
            raise HTTPException(
                status_code=400,
                detail="No user message found"
            )
        
        # Convert the chat history to the format expected by the model
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages[:-1]  # Exclude the latest message
        ]
        
        # Generate response using RAG
        result = rag_model.answer(
            query=latest_message,
            chat_history=chat_history,
            top_k=request.top_k,
            temperature=request.temperature,
            stream=False
        )
        
        answer = result["answer"]
        sources = result["sources"]
        
        # Format sources
        formatted_sources = []
        for source in sources:
            # Get document metadata
            document = None
            try:
                document = document_store.get_document(source.get("document_id"))
            except Exception as e:
                logger.warning(f"Error getting document {source.get('document_id')}: {e}")
            
            # Create source object
            formatted_source = {
                "document_id": source.get("document_id"),
                "text": source.get("text", ""),
                "score": source.get("score", 0.0),
                "metadata": source.get("metadata", {})
            }
            
            # Add document metadata
            if document:
                formatted_source["metadata"] = document.get("metadata", {})
                
                # Add filename
                formatted_source["metadata"]["filename"] = document.get("filename", "")
            
            formatted_sources.append(formatted_source)
        
        # Create response
        response = {
            "id": str(uuid.uuid4()),
            "message": {
                "role": "assistant",
                "content": answer
            },
            "sources": formatted_sources
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating chat response: {str(e)}"
        )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat with documents using RAG.
    
    Args:
        request: Chat request
        
    Returns:
        Streaming response with sources
    """
    # Force streaming to be enabled
    request.stream = True
    
    async def generate():
        try:
            # Check if messages are provided
            if not request.messages:
                yield json.dumps({"error": "Messages cannot be empty"})
                return
            
            # Get the latest user message
            latest_message = None
            for msg in reversed(request.messages):
                if msg.role.lower() == "user":
                    latest_message = msg.content
                    break
            
            if not latest_message:
                yield json.dumps({"error": "No user message found"})
                return
            
            # Convert the chat history to the format expected by the model
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages[:-1]  # Exclude the latest message
            ]
            
            # Check if vLLM streaming is available
            if settings.VLLM_ENABLED and hasattr(rag_model.llm_client, "generate"):
                # First retrieve the context
                results = rag_model.retrieve(
                    query=latest_message,
                    top_k=request.top_k
                )
                
                # Format context
                context = rag_model._format_context(results)
                
                # Create prompt
                prompt = rag_model._create_chat_prompt(latest_message, context, chat_history)
                
                # Generate streaming response
                stream = rag_model.generate(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=request.temperature,
                    stream=True
                )
                
                # Stream tokens
                for chunk in stream:
                    if chunk:
                        token = chunk.outputs[0].text
                        yield json.dumps({"text": token, "type": "token"}) + "\n"
                
                # Stream sources at the end
                for source in results[:3]:  # Only include top 3 sources
                    doc_id = source.get("document_id", "")
                    text = source.get("text", "")[:200] + "..."  # Truncate for brevity
                    score = source.get("score", 0.0)
                    
                    yield json.dumps({
                        "document_id": doc_id,
                        "text": text,
                        "score": score,
                        "type": "source"
                    }) + "\n"
                
                # End marker
                yield json.dumps({"type": "end"}) + "\n"
                
            else:
                # Non-streaming fallback
                result = rag_model.answer(
                    query=latest_message,
                    chat_history=chat_history,
                    top_k=request.top_k,
                    temperature=request.temperature,
                    stream=False
                )
                
                answer = result["answer"]
                sources = result["sources"]
                
                # Simulate streaming by sending answer in chunks
                tokens = answer.split()
                for i in range(0, len(tokens), 3):
                    chunk = " ".join(tokens[i:i+3])
                    yield json.dumps({"text": chunk + " ", "type": "token"}) + "\n"
                    # In a real streaming implementation, you would add a small delay here
                
                # Stream sources
                for source in sources[:3]:  # Only include top 3 sources
                    doc_id = source.get("document_id", "")
                    text = source.get("text", "")[:200] + "..."  # Truncate for brevity
                    score = source.get("score", 0.0)
                    
                    yield json.dumps({
                        "document_id": doc_id,
                        "text": text,
                        "score": score,
                        "type": "source"
                    }) + "\n"
                
                # End marker
                yield json.dumps({"type": "end"}) + "\n"
                
        except Exception as e:
            logger.error(f"Error streaming chat response: {e}")
            yield json.dumps({"error": str(e), "type": "error"}) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )

@router.post("/question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a question using RAG.
    
    Args:
        request: Question request
        
    Returns:
        Question answer with sources and confidence
    """
    try:
        # Check if question is provided
        if not request.question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Answer question using QA model
        result = qa_model.answer_question(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature
        )
        
        answer = result["answer"]
        sources = result["sources"]
        confidence = result["confidence"]
        
        # Format sources
        formatted_sources = []
        for source in sources:
            # Get document metadata
            document = None
            try:
                document = document_store.get_document(source.get("document_id"))
            except Exception as e:
                logger.warning(f"Error getting document {source.get('document_id')}: {e}")
            
            # Create source object
            formatted_source = {
                "document_id": source.get("document_id"),
                "text": source.get("text", ""),
                "score": source.get("score", 0.0),
                "metadata": source.get("metadata", {})
            }
            
            # Add document metadata
            if document:
                formatted_source["metadata"] = document.get("metadata", {})
                
                # Add filename
                formatted_source["metadata"]["filename"] = document.get("filename", "")
            
            formatted_sources.append(formatted_source)
        
        # Create response
        response = {
            "id": str(uuid.uuid4()),
            "question": request.question,
            "answer": answer,
            "confidence": confidence,
            "sources": formatted_sources
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )

@router.post("/question/citations", response_model=Dict[str, Any])
async def answer_with_citations(request: QuestionRequest):
    """
    Answer a question with explicit citations.
    
    Args:
        request: Question request
        
    Returns:
        Answer with citations
    """
    try:
        # Check if question is provided
        if not request.question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Answer with citations
        result = qa_model.answer_with_citations(
            question=request.question
        )
        
        # Format citations
        formatted_citations = []
        for citation in result["citations"]:
            citation_id = citation.get("id")
            doc_id = citation.get("document_id")
            
            # Get document metadata
            document = None
            try:
                document = document_store.get_document(doc_id)
            except Exception as e:
                logger.warning(f"Error getting document {doc_id}: {e}")
            
            # Create citation object
            formatted_citation = {
                "id": citation_id,
                "document_id": doc_id,
                "text": citation.get("text", ""),
                "metadata": citation.get("metadata", {})
            }
            
            # Add document metadata
            if document:
                formatted_citation["metadata"] = document.get("metadata", {})
                
                # Add filename
                formatted_citation["metadata"]["filename"] = document.get("filename", "")
            
            formatted_citations.append(formatted_citation)
        
        # Create response
        response = {
            "id": str(uuid.uuid4()),
            "question": request.question,
            "answer": result["answer"],
            "citations": formatted_citations
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error answering with citations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error answering with citations: {str(e)}"
        )
