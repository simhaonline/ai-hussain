"""
FastAPI application for the AI Conversational PDF Processing System.
"""
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any

from config.config import settings
from src.api.routes import documents, search, chat

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="AI Conversational PDF Processing System",
        description="API for processing PDFs and enabling conversational interaction with document content",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
    app.include_router(search.router, prefix="/api/search", tags=["Search"])
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    
    # API key authentication if enabled
    if settings.API_KEY_ENABLED:
        from fastapi.security.api_key import APIKeyHeader
        from fastapi import Security
        
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        
        async def get_api_key(api_key: str = Security(api_key_header)):
            if api_key is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key is missing"
                )
            if api_key != settings.API_KEY:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid API key"
                )
            return api_key
        
        # Dependency to check API key
        app.dependency_overrides[api_key_header] = get_api_key
    
    # Add health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok"}
    
    # Add system info endpoint
    @app.get("/api/system/info", tags=["System"])
    async def system_info():
        """Get system information."""
        # Import optional components only when needed
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if gpu_available else None
            } if gpu_available else {}
        except ImportError:
            gpu_available = False
            gpu_info = {}
        
        # Gather system information
        from src.model_manager.rag_model import RAGModel
        try:
            rag_model = RAGModel()
            llm_info = rag_model.get_llm_info()
        except:
            llm_info = {
                "model_name": settings.LLM_MODEL,
                "vllm_enabled": settings.VLLM_ENABLED
            }
        
        return {
            "api_version": "1.0.0",
            "gpu": {
                "available": gpu_available,
                **gpu_info
            },
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "llm": llm_info,
            "vector_db": settings.VECTOR_DB_TYPE
        }
    
    # Add error handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        logger.exception(f"Unhandled exception: {exc}")
        return {"detail": "An internal server error occurred."}
    
    return app
