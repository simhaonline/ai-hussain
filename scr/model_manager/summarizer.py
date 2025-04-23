"""
Text summarization module.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from config.config import settings
from src.text_analyzer.chunker import TextChunker
from src.model_manager.rag_model import RAGModel

logger = logging.getLogger(__name__)

class Summarizer:
    """
    Text summarization model implementation.
    
    This class provides capabilities to summarize documents and chunks
    using extractive and abstractive summarization techniques.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the summarizer.
        
        Args:
            llm_client: Language model client (if None, uses RAGModel's client)
        """
        rag_model = RAGModel()
        self.llm_client = llm_client or rag_model.llm_client
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE * 2,  # Larger chunks for summarization
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    def _generate_summary(self, text: str, max_length: int = 200, min_length: Optional[int] = None) -> str:
        """
        Generate a summary using the language model.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            min_length: Minimum summary length in words
            
        Returns:
            Generated summary
        """
        min_length_str = f"at least {min_length} words" if min_length else "concise"
        
        # Create a prompt for summarization
        prompt = f"""
Please summarize the following text in {min_length_str} but no more than {max_length} words, 
capturing the main points and key information:

{text}

Summary:
"""
        
        # Generate summary
        if settings.VLLM_ENABLED and hasattr(self.llm_client, "generate"):
            # vLLM generate
            params = {
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.3,
                "stream": False
            }
            
            completion = self.llm_client.generate(**params)
            summary = completion.outputs[0].text
            
        else:
            # Transformers generate
            generate_params = {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "do_sample": True,
                "top_p": 0.9,
                "num_return_sequences": 1
            }
            
            outputs = self.llm_client(
                prompt,
                **generate_params
            )
            
            summary = outputs[0]["generated_text"][len(prompt):]
        
        # Clean up the summary
        summary = summary.strip()
        
        # Remove any prompt echoing
        if summary.startswith("Summary:"):
            summary = summary[len("Summary:"):].strip()
        
        return summary
    
    def summarize_text(self, text: str, max_length: int = 200, min_length: Optional[int] = None) -> str:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            min_length: Minimum summary length in words
            
        Returns:
            Generated summary
        """
        # Check if text is short enough to summarize directly
        if len(text.split()) <= 4000:
            return self._generate_summary(text, max_length, min_length)
        
        # For longer text, use recursive summarization
        return self.summarize_long_text(text, max_length, min_length)
    
    def summarize_long_text(self, text: str, max_length: int = 200, min_length: Optional[int] = None) -> str:
        """
        Summarize long text using a recursive approach.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            min_length: Minimum summary length in words
            
        Returns:
            Generated summary
        """
        # Split into chunks
        chunks = self.chunker.chunk_text(text)
        
        if not chunks:
            return "No text to summarize."
        
        # If there's only one chunk, summarize directly
        if len(chunks) == 1:
            return self._generate_summary(chunks[0], max_length, min_length)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_max_length = max(100, max_length // 2)
                summary = self._generate_summary(chunk, chunk_max_length)
                chunk_summaries.append(summary)
                logger.debug(f"Summarized chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {e}")
                # Use a truncated version of the original chunk as a fallback
                truncated = chunk[:500] + "..." if len(chunk) > 500 else chunk
                chunk_summaries.append(truncated)
        
        # Combine the chunk summaries
        combined_text = "\n\n".join(chunk_summaries)
        
        # Generate a final summary from the combined summaries
        return self._generate_summary(combined_text, max_length, min_length)
    
    def summarize_document(self, document: Dict[str, Any], max_length: int = 300) -> Dict[str, Any]:
        """
        Summarize a document from the document store.
        
        Args:
            document: Document dictionary
            max_length: Maximum summary length in words
            
        Returns:
            Dictionary with summary and metadata
        """
        text = document.get("text", "")
        
        if not text:
            if "pages" in document:
                # Combine text from pages
                text = "\n\n".join([page.get("text", "") for page in document.get("pages", [])])
            elif "chunks" in document:
                # Combine text from chunks
                text = "\n\n".join([chunk.get("text", "") for chunk in document.get("chunks", [])])
        
        if not text:
            return {
                "document_id": document.get("id"),
                "summary": "No text available to summarize.",
                "metadata": document.get("metadata", {})
            }
        
        # Generate summary
        summary = self.summarize_text(text, max_length)
        
        return {
            "document_id": document.get("id"),
            "summary": summary,
            "metadata": document.get("metadata", {})
        }
    
    def generate_executive_summary(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an executive summary of a document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Dictionary with executive summary, key points, and metadata
        """
        # Get the regular summary first
        summary_result = self.summarize_document(document, max_length=400)
        summary = summary_result["summary"]
        
        # Extract key points
        key_points_prompt = f"""
Based on the following summary of a document, extract 3-5 key takeaway points.
List each point on a new line starting with a dash (-).

Summary:
{summary}

Key points:
"""
        
        key_points_text = ""
        try:
            if settings.VLLM_ENABLED and hasattr(self.llm_client, "generate"):
                # vLLM generate
                params = {
                    "prompt": key_points_prompt,
                    "max_tokens": 256,
                    "temperature": 0.3,
                    "stream": False
                }
                
                completion = self.llm_client.generate(**params)
                key_points_text = completion.outputs[0].text
                
            else:
                # Transformers generate
                generate_params = {
                    "max_new_tokens": 256,
                    "temperature": 0.3,
                    "do_sample": True,
                    "top_p": 0.9,
                    "num_return_sequences": 1
                }
                
                outputs = self.llm_client(
                    key_points_prompt,
                    **generate_params
                )
                
                key_points_text = outputs[0]["generated_text"][len(key_points_prompt):]
        except Exception as e:
            logger.error(f"Error generating TOC: {e}")
            return []
        
        # Parse TOC
        toc = []
        try:
            # Extract JSON part (it might be wrapped in ```json ``` or other text)
            json_match = re.search(r'\[\s*\{.*\}\s*\]', toc_text, re.DOTALL)
            if json_match:
                import json
                toc = json.loads(json_match.group(0))
            else:
                # Try to find JSON with different delimiters
                json_match = re.search(r'\{.*"sections"\s*:\s*\[.*\]\s*\}', toc_text, re.DOTALL)
                if json_match:
                    import json
                    data = json.loads(json_match.group(0))
                    toc = data.get("sections", [])
        except Exception as e:
            logger.error(f"Error parsing TOC JSON: {e}")
            
            # Fallback: Try to extract TOC in plain text format
            try:
                sections = []
                current_section = {}
                
                for line in toc_text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this looks like a section title
                    if re.match(r'^[0-9.]+\s+', line) or re.match(r'^[IVXLCDM]+\.\s+', line):
                        # Save previous section if exists
                        if current_section and "title" in current_section:
                            sections.append(current_section)
                        
                        # Start new section
                        title_match = re.search(r'^(?:[0-9.]+|[IVXLCDM]+\.)\s+(.*)', line)
                        title = title_match.group(1) if title_match else line
                        current_section = {"title": title, "description": ""}
                    elif current_section:
                        # Add to description
                        current_section["description"] = current_section.get("description", "") + " " + line
                
                # Add the last section
                if current_section and "title" in current_section:
                    sections.append(current_section)
                
                toc = sections
            except Exception as e2:
                logger.error(f"Error parsing TOC as plain text: {e2}")
        
        return toc
    
    def compare_documents(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two documents and highlight key similarities and differences.
        
        Args:
            doc1: First document dictionary
            doc2: Second document dictionary
            
        Returns:
            Dictionary with comparison results
        """
        # Get summaries of both documents
        summary1 = self.summarize_document(doc1)["summary"]
        summary2 = self.summarize_document(doc2)["summary"]
        
        # Create comparison prompt
        comparison_prompt = f"""
Compare the following two document summaries and identify key similarities and differences.
List at least 3 similarities and 3 differences in a structured format.

Document 1 Summary:
{summary1}

Document 2 Summary:
{summary2}

Comparison:
Similarities:
- 
Differences:
- 
"""
        
        # Generate comparison
        comparison_text = ""
        try:
            if settings.VLLM_ENABLED and hasattr(self.llm_client, "generate"):
                # vLLM generate
                params = {
                    "prompt": comparison_prompt,
                    "max_tokens": 512,
                    "temperature": 0.3,
                    "stream": False
                }
                
                completion = self.llm_client.generate(**params)
                comparison_text = completion.outputs[0].text
                
            else:
                # Transformers generate
                generate_params = {
                    "max_new_tokens": 512,
                    "temperature": 0.3,
                    "do_sample": True,
                    "top_p": 0.9,
                    "num_return_sequences": 1
                }
                
                outputs = self.llm_client(
                    comparison_prompt,
                    **generate_params
                )
                
                comparison_text = outputs[0]["generated_text"][len(comparison_prompt):]
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            comparison_text = "Failed to generate comparison."
        
        # Parse similarities and differences
        similarities = []
        differences = []
        
        current_section = None
        for line in comparison_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.lower() == "similarities:":
                current_section = "similarities"
            elif line.lower() == "differences:":
                current_section = "differences"
            elif line.startswith("-") or line.startswith("*") or re.match(r"^[0-9]+\.", line):
                point = line.lstrip("-*0123456789. ").strip()
                if current_section == "similarities":
                    similarities.append(point)
                elif current_section == "differences":
                    differences.append(point)
        
        return {
            "document1_id": doc1.get("id"),
            "document2_id": doc2.get("id"),
            "document1_title": doc1.get("metadata", {}).get("title", "Document 1"),
            "document2_title": doc2.get("metadata", {}).get("title", "Document 2"),
            "similarities": similarities,
            "differences": differences,
            "document1_summary": summary1,
            "document2_summary": summary2
        } generating key points: {e}")
            key_points_text = "- Failed to extract key points"
        
        # Parse key points
        key_points = [
            point.strip().lstrip("- ") 
            for point in key_points_text.split("\n") 
            if point.strip() and point.strip().startswith("-")
        ]
        
        # If no key points were extracted or parsing failed, try again with a different format
        if not key_points:
            key_points = [point.strip() for point in key_points_text.strip().split("\n") if point.strip()]
        
        return {
            "document_id": document.get("id"),
            "executive_summary": summary,
            "key_points": key_points,
            "metadata": document.get("metadata", {})
        }
    
    def generate_table_of_contents(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a table of contents for a document.
        
        Args:
            document: Document dictionary
            
        Returns:
            List of section dictionaries with titles and summaries
        """
        text = document.get("text", "")
        
        if not text:
            if "pages" in document:
                # Combine text from pages
                text = "\n\n".join([page.get("text", "") for page in document.get("pages", [])])
            elif "chunks" in document:
                # Combine text from chunks
                text = "\n\n".join([chunk.get("text", "") for chunk in document.get("chunks", [])])
        
        if not text:
            return []
        
        # Create a prompt for TOC generation
        toc_prompt = f"""
Please analyze the following document and generate a table of contents with sections and subsections.
For each section, provide a brief 1-2 sentence description.
Format your response as JSON with sections, each having a "title" and "description".

Document:
{text[:8000]}...

Table of Contents (JSON format):
"""
        
        # Generate TOC
        toc_text = ""
        try:
            if settings.VLLM_ENABLED and hasattr(self.llm_client, "generate"):
                # vLLM generate
                params = {
                    "prompt": toc_prompt,
                    "max_tokens": 1024,
                    "temperature": 0.2,
                    "stream": False
                }
                
                completion = self.llm_client.generate(**params)
                toc_text = completion.outputs[0].text
                
            else:
                # Transformers generate
                generate_params = {
                    "max_new_tokens": 1024,
                    "temperature": 0.2,
                    "do_sample": True,
                    "top_p": 0.9,
                    "num_return_sequences": 1
                }
                
                outputs = self.llm_client(
                    toc_prompt,
                    **generate_params
                )
                
                toc_text = outputs[0]["generated_text"][len(toc_prompt):]
        except Exception as e:
            logger.error(f"Error
