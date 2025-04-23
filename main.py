"""
Main entry point for the AI Conversational PDF Processing System.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.app import create_app
from config.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG_MODE else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_pdf_processor")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="AI Conversational PDF Processing System")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", type=str, default=settings.API_HOST, help="Host to run the API on")
    api_parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port to run the API on")
    api_parser.add_argument("--workers", type=int, default=settings.API_WORKERS, help="Number of workers")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDF documents")
    process_parser.add_argument("--input", type=str, required=True, help="Input PDF file or directory")
    process_parser.add_argument("--output", type=str, help="Output directory for processed files")
    process_parser.add_argument("--ocr", action="store_true", help="Force OCR processing")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for processed documents")
    embed_parser.add_argument("--input", type=str, required=True, help="Input processed documents directory")
    embed_parser.add_argument("--model", type=str, default=settings.EMBEDDING_MODEL, help="Embedding model to use")
    
    args = parser.parse_args()
    
    if args.command == "api":
        from uvicorn import run
        
        app = create_app()
        logger.info(f"Starting API server on {args.host}:{args.port}")
        run(app, host=args.host, port=args.port, workers=args.workers)
    
    elif args.command == "process":
        from src.pdf_processor.extractor import PDFExtractor
        
        extractor = PDFExtractor(ocr_enabled=args.ocr or settings.OCR_ENABLED)
        input_path = Path(args.input)
        output_dir = Path(args.output) if args.output else settings.PROCESSED_DATA_DIR
        
        if input_path.is_file():
            logger.info(f"Processing single PDF: {input_path}")
            extractor.process_pdf(input_path, output_dir)
        elif input_path.is_dir():
            logger.info(f"Processing directory of PDFs: {input_path}")
            pdf_files = list(input_path.glob("**/*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files:
                try:
                    logger.info(f"Processing {pdf_file}")
                    extractor.process_pdf(pdf_file, output_dir)
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
        else:
            logger.error(f"Input path {input_path} does not exist")
            sys.exit(1)
    
    elif args.command == "embed":
        from src.text_analyzer.embedder import TextEmbedder
        from src.document_store.vector_db import VectorDBManager
        
        embedder = TextEmbedder(model_name=args.model)
        vector_db = VectorDBManager()
        
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input path {input_path} does not exist")
            sys.exit(1)
        
        processed_files = list(input_path.glob("**/*.json"))
        logger.info(f"Found {len(processed_files)} processed document files")
        
        for file in processed_files:
            try:
                logger.info(f"Generating embeddings for {file}")
                # Load the processed document
                import json
                with open(file, "r", encoding="utf-8") as f:
                    document = json.load(f)
                
                # Generate embeddings for chunks
                document_id = document.get("id", str(file.stem))
                chunks = document.get("chunks", [])
                
                if not chunks:
                    logger.warning(f"No chunks found in {file}")
                    continue
                
                # Embed chunks
                texts = [chunk["text"] for chunk in chunks]
                embeddings = embedder.embed_texts(texts)
                
                # Store in vector database
                vector_db.add_documents(document_id, texts, embeddings, 
                                       {**document.get("metadata", {}), "source": str(file)})
                
                logger.info(f"Successfully embedded {len(texts)} chunks from {file}")
                
            except Exception as e:
                logger.error(f"Error embedding {file}: {e}")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
