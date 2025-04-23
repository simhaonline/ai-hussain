1. PDF Processing

PDFExtractor: Extracts text from both native PDFs and scanned documents using PyMuPDF and Tesseract OCR
OCR Module: Handles scanned pages with preprocessing for better text recognition

2. Text Analysis

TextChunker: Splits documents into semantic chunks with appropriate overlap
TextEmbedder: Generates vector representations using BGE-M3 model
KeywordExtractor: Extracts keywords and keyphrases from documents

3. Vector Database

VectorDBManager: Abstract interface to multiple vector databases
Support for FAISS, Qdrant, and Milvus backends
Storing and retrieving vector embeddings with metadata

4. Document Storage

DocumentStore: Manages document metadata and content
Support for MongoDB and simple JSON file-based storage
Document CRUD operations with metadata management

5. Model Management

RAGModel: Core implementation of Retrieval-Augmented Generation
QAModel: Specialized question-answering functionality
Summarizer: Document summarization with executive summaries and TOC generation
Support for both local transformers and vLLM for high-performance inference

6. API Layer

FastAPI Application: RESTful endpoints for all functionality
Document upload, processing, and management
Semantic search with filtering
Conversational interface with streaming responses
Question answering with citations

All components are designed to work together while being modular for easy troubleshooting. The system provides a complete end-to-end solution for:

Ingesting PDF documents (including OCR for scanned pages)
Preprocessing and analyzing text
Generating embeddings with BGE-M3
Storing vectors in a vector database
Providing a conversational interface for document interaction
Answering questions based on document content

The code follows best practices with proper error handling, logging, and configuration management. It's designed to be scalable and adaptable for different deployment scenarios.
