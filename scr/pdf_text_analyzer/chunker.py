"""
Text chunking module for splitting documents into manageable pieces.
"""
import logging
import re
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class TextChunker:
    """Class to split text into chunks of manageable size."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of this chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Look for a natural break point (paragraph or sentence)
            chunk_end = self._find_break_point(text, end)
            
            # Add the chunk
            chunks.append(text[start:chunk_end])
            
            # Move the start pointer for the next chunk, considering overlap
            start = chunk_end - self.chunk_overlap
            
            # Ensure we're making forward progress
            if start >= chunk_end:
                start = chunk_end
        
        return chunks
    
    def _find_break_point(self, text: str, position: int) -> int:
        """
        Find a natural break point near the given position.
        Looks for paragraph breaks, then sentence breaks, then word breaks.
        
        Args:
            text: The text to search within
            position: Approximate position to find a break point near
            
        Returns:
            Index of the break point
        """
        # Search window for finding break points
        search_window = 200  # characters
        start_pos = max(0, position - search_window)
        end_pos = min(len(text), position + search_window)
        
        search_text = text[start_pos:end_pos]
        
        # Try to find paragraph breaks first (double newline)
        paragraph_breaks = [m.start() + start_pos for m in re.finditer(r'\n\s*\n', search_text)]
        if paragraph_breaks:
            # Find the closest paragraph break to the position
            closest_break = min(paragraph_breaks, key=lambda x: abs(x - position))
            return closest_break
        
        # Try to find sentence breaks (., !, ?)
        sentence_breaks = [m.start() + start_pos for m in re.finditer(r'[.!?]\s+', search_text)]
        if sentence_breaks:
            # Find the closest sentence break to the position
            closest_break = min(sentence_breaks, key=lambda x: abs(x - position))
            # Include the punctuation and space
            return closest_break + 2
        
        # Try to find word breaks (space)
        word_breaks = [m.start() + start_pos for m in re.finditer(r'\s+', search_text)]
        if word_breaks:
            # Find the closest word break to the position
            closest_break = min(word_breaks, key=lambda x: abs(x - position))
            # Include the space
            return closest_break + 1
        
        # If no natural breaks found, just use the position
        return position
    
    def chunk_by_semantic_units(self, text: str, max_tokens: int = 500) -> List[str]:
        """
        Alternative chunking method that tries to preserve semantic units like
        paragraphs, sentences, etc. Up to max_tokens.
        
        Args:
            text: Text to split into chunks
            max_tokens: Approximate maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Estimate: 1 token is roughly 4 characters in English
        char_limit = max_tokens * 4
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            paragraph_length = len(paragraph)
            
            # If this paragraph alone exceeds the limit, we need to split it further
            if paragraph_length > char_limit:
                # If we have content in the current chunk, finish it first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                
                sentence_chunk = []
                sentence_length = 0
                
                for sentence in sentences:
                    sentence_length_chars = len(sentence)
                    
                    # If this sentence alone exceeds the limit, we'll add it as its own chunk
                    if sentence_length_chars > char_limit:
                        # If we have content in the sentence chunk, finish it first
                        if sentence_chunk:
                            chunks.append(' '.join(sentence_chunk))
                            sentence_chunk = []
                            sentence_length = 0
                        
                        # Add the long sentence as its own chunk
                        chunks.append(sentence)
                    else:
                        # Check if adding this sentence would exceed the limit
                        if sentence_length + sentence_length_chars > char_limit:
                            # Finish the current sentence chunk
                            chunks.append(' '.join(sentence_chunk))
                            sentence_chunk = [sentence]
                            sentence_length = sentence_length_chars
                        else:
                            # Add the sentence to the current chunk
                            sentence_chunk.append(sentence)
                            sentence_length += sentence_length_chars
                
                # Add any remaining sentences
                if sentence_chunk:
                    chunks.append(' '.join(sentence_chunk))
            
            # If adding this paragraph would exceed the limit, start a new chunk
            elif current_length + paragraph_length > char_limit:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                # Add paragraph to the current chunk
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
