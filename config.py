"""
Configuration settings for the AI Conversational PDF Processing System.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    RAW_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "raw")
    PROCESSED_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "processed")
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    
    # PDF Processing
    OCR_ENABLED: bool = True
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 300
    PDF_EXTRACTION_TIMEOUT: int = 60  # seconds
    
    # Text Analysis
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-m3"  # BGE-M3 embedding model
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Vector DB
    VECTOR_DB_TYPE: str = "qdrant"  # One of: "faiss", "qdrant", "milvus"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "pdf_documents"
    
    # LLM Settings
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"  # Default LLM model
    LLM_HOST: Optional[str] = None  # For hosted models
    LLM_PORT: Optional[int] = None  # For hosted models
    VLLM_ENABLED: bool = False      # Whether to use vLLM for inference
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    DEBUG_MODE: bool = False
    
    # Document Store
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "pdf_processor"
    MONGODB_COLLECTION: str = "documents"
    
    # Authentication (if needed)
    API_KEY_ENABLED: bool = False
    API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
# Create an instance of the settings
settings = Settings()

# Create necessary directories
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
