"""
PDF text extraction module using PyMuPDF with OCR capabilities for scanned pages.
"""
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io

from config.config import settings
from src.pdf_processor.ocr import perform_ocr
from src.text_analyzer.chunker import TextChunker

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Class to extract text from PDF documents with OCR capabilities."""
    
    def __init__(self, ocr_enabled: bool = True, language: str = settings.OCR_LANGUAGE):
        """
        Initialize PDF extractor.
        
        Args:
            ocr_enabled: Whether to use OCR for scanned pages
            language: OCR language
        """
        self.ocr_enabled = ocr_enabled
        self.language = language
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    def _extract_text_from_page(self, page: fitz.Page) -> Tuple[str, bool]:
        """
        Extract text from a PDF page, detecting if OCR is needed.
        
        Args:
            page: PDF page object
            
        Returns:
            Tuple of (extracted text, whether OCR was used)
        """
        # Try direct text extraction first
        text = page.get_text()
        
        # Check if the page might be scanned (little or no text)
        if len(text.strip()) < 50 and self.ocr_enabled:
            logger.debug(f"Page {page.number} appears to be scanned or has little text. Using OCR.")
            
            # Render the page to an image
            pix = page.get_pixmap(dpi=settings.OCR_DPI)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # Perform OCR
            text = perform_ocr(img, self.language)
            return text, True
        
        return text, False
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text from a PDF document, with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text, metadata, and page information
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        result = {
            "id": str(uuid.uuid4()),
            "filename": pdf_path.name,
            "pages": [],
            "metadata": {},
            "text": "",
            "chunks": []
        }
        
        try:
            doc = fitz.open(str(pdf_path))
            
            # Extract document metadata
            metadata = doc.metadata
            result["metadata"] = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": len(doc)
            }
            
            all_text = []
            
            # Process each page
            for page_num, page in enumerate(doc):
                text, ocr_used = self._extract_text_from_page(page)
                
                # Clean and normalize text
                text = self._clean_text(text)
                
                page_info = {
                    "page_num": page_num + 1,
                    "text": text,
                    "ocr_used": ocr_used
                }
                
                result["pages"].append(page_info)
                all_text.append(text)
            
            # Combine all text
            full_text = " ".join(all_text)
            result["text"] = full_text
            
            # Create chunks
            result["chunks"] = self._create_chunks(full_text, result["metadata"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with a single space
        text = " ".join(text.split())
        
        # Remove excessive newlines, keeping paragraph structure
        text = text.replace("\n\n\n", "\n\n")
        
        return text
    
    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks from the extracted text.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with text and metadata
        """
        chunk_texts = self.chunker.chunk_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = {
                "chunk_id": i,
                "text": chunk_text,
                "metadata": {**metadata, "chunk_id": i}
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_pdf(self, pdf_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process a PDF document and save the extracted text and metadata.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save the processed output
            
        Returns:
            Dictionary with extracted text, metadata, and page information
        """
        if output_dir is None:
            output_dir = settings.PROCESSED_DATA_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract text and metadata
        result = self.extract_text_from_pdf(pdf_path)
        
        # Save the result
        output_path = output_dir / f"{pdf_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed PDF saved to {output_path}")
        
        return result
