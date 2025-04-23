"""
Retrieval-Augmented Generation (RAG) model implementation.
"""
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from config.config import settings
from src.text_analyzer.embedder import TextEmbedder, BGEEmbedder
from src.document_store.vector_db import VectorDBManager

logger = logging.getLogger(__name__)

class RAGModel:
    """
    Retrieval-Augmented Generation model implementation.
    
    This class combines a vector store retrieval system with a language model
    to provide context-aware responses to queries.
    """
    
    def __init__(self, 
                vector_db: Optional[VectorDBManager] = None,
                llm_client: Optional[Any] = None,
                embedder: Optional[TextEmbedder] = None,
                max_context_length: int = 4000,
                top_k: int = 5,
                min_similarity_score: float = 0.2):
        """
        Initialize the RAG model.
        
        Args:
            vector_db: Vector database manager
            llm_client: Language model client
            embedder: Text embedder for queries
            max_context_length: Maximum context length for the LLM
            top_k: Number of top documents to retrieve
            min_similarity_score: Minimum similarity score for retrieval
        """
        self.vector_db = vector_db or VectorDBManager()
        self.llm_client = llm_client or self._initialize_llm_client()
        
        # Initialize embedder
        if embedder is None:
            if "bge" in settings.EMBEDDING_MODEL.lower():
                self.embedder = BGEEmbedder(model_name=settings.EMBEDDING_MODEL)
            else:
                self.embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL)
        else:
            self.embedder = embedder
        
        self.max_context_length = max_context_length
        self.top_k = top_k
        self.min_similarity_score = min_similarity_score
    
    def _initialize_llm_client(self) -> Any:
        """
        Initialize the language model client based on configuration.
        
        Returns:
            Language model client
        """
        if settings.VLLM_ENABLED:
            return self._initialize_vllm_client()
        else:
            return self._initialize_transformers_client()
    
    def _initialize_vllm_client(self) -> Any:
        """
        Initialize vLLM client for high-performance LLM inference.
        
        Returns:
            vLLM client
        """
        try:
            from vllm.client import SyncClient
            
            client = SyncClient(
                host=settings.LLM_HOST or "localhost",
                port=settings.LLM_PORT or 8000
            )
            
            logger.info(f"Initialized vLLM client for model: {settings.LLM_MODEL}")
            return client
            
        except ImportError:
            logger.warning("vLLM not installed. Install with 'pip install vllm'")
            logger.warning("Falling back to Transformers client")
            return self._initialize_transformers_client()
        except Exception as e:
            logger.error(f"Error initializing vLLM client: {e}")
            logger.warning("Falling back to Transformers client")
            return self._initialize_transformers_client()
    
    def _initialize_transformers_client(self) -> Any:
        """
        Initialize HuggingFace Transformers client.
        
        Returns:
            Transformers pipeline
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(torch, 'has_mps') and torch.has_mps:
                device = "mps"  # For Apple Silicon
            
            logger.info(f"Loading LLM model: {settings.LLM_MODEL} on {device}")
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            
            tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL)
            
            # Create pipeline
            client = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            logger.info(f"Initialized Transformers client for model: {settings.LLM_MODEL}")
            return client
            
        except Exception as e:
            logger.error(f"Error initializing Transformers client: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of retrieved documents
        """
        # Embed query
        query_embedding = None
        if isinstance(self.embedder, BGEEmbedder):
            query_embedding = self.embedder.embed_query(query)
        else:
            query_embedding = self.embedder.embed_text(query)
        
        # Search for relevant documents
        k = top_k or self.top_k
        results = self.vector_db.search(query_embedding, k)
        
        # Filter by score if needed
        filtered_results = [
            result for result in results 
            if result.get("score", 0) >= self.min_similarity_score
        ]
        
        logger.info(f"Retrieved {len(filtered_results)} documents for query: {query}")
        return filtered_results
    
    def generate(self, prompt: str, 
                max_tokens: int = 512, 
                temperature: float = 0.7,
                stream: bool = False) -> Union[str, Any]:
        """
        Generate text using the language model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the output
            
        Returns:
            Generated text or streaming object
        """
        # Check if using vLLM or Transformers
        if settings.VLLM_ENABLED and hasattr(self.llm_client, "generate"):
            # vLLM generate
            params = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            
            completion = self.llm_client.generate(**params)
            
            if stream:
                return completion
            else:
                return completion.outputs[0].text
                
        else:
            # Transformers generate
            generate_params = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.0,
                "top_p": 0.9,
                "num_return_sequences": 1
            }
            
            if stream:
                # Set up streaming parameters
                generate_params["streamer"] = "transformer_streamer"
                raise NotImplementedError("Streaming not implemented for Transformers pipeline")
            
            try:
                outputs = self.llm_client(
                    prompt,
                    **generate_params
                )
                
                return outputs[0]["generated_text"][len(prompt):]
                
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                return "I'm sorry, I encountered an error while generating a response."
    
    def _create_chat_prompt(self, query: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Create a formatted prompt for the language model.
        
        Args:
            query: User query
            context: Retrieved context
            chat_history: Previous chat messages
            
        Returns:
            Formatted prompt string
        """
        # Default prompt template for Mistral-like models
        sys_prompt = "You are a helpful, accurate, and friendly assistant that answers questions based on the provided context. If the question cannot be answered using the context, say that you don't know the answer. Always cite your sources in the answer."
        
        prompt = f"<s>[INST] {sys_prompt}\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
        
        return prompt
    
    def _format_context(self, results: List[Dict[str, Any]], max_length: Optional[int] = None) -> str:
        """
        Format retrieved documents into context.
        
        Args:
            results: Retrieved documents
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        # Limit total context size
        max_len = max_length or self.max_context_length
        current_len = 0
        formatted_chunks = []
        
        for i, doc in enumerate(results):
            text = doc.get("text", "").strip()
            doc_id = doc.get("document_id", "unknown")
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            
            # Format chunk with reference info
            formatted_chunk = f"[Document {i+1} - {filename}]\n{text}\n"
            
            # Check if adding this chunk would exceed max length
            if current_len + len(formatted_chunk) > max_len:
                # Add truncated version if this is the first chunk
                if i == 0:
                    truncated = text[:max_len - 100] + "..."
                    formatted_chunks.append(f"[Document {i+1} - {filename}]\n{truncated}\n")
                break
            
            formatted_chunks.append(formatted_chunk)
            current_len += len(formatted_chunk)
        
        return "\n".join(formatted_chunks)
    
    def answer(self, query: str, 
              chat_history: Optional[List[Dict[str, str]]] = None,
              top_k: Optional[int] = None,
              max_tokens: int = 512,
              temperature: float = 0.7,
              stream: bool = False) -> Union[str, Dict[str, Any], Any]:
        """
        Answer a query using RAG.
        
        Args:
            query: User query
            chat_history: Previous chat messages
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the output
            
        Returns:
            Answer string, or dictionary with answer and sources, or streaming object
        """
        # Retrieve relevant documents
        results = self.retrieve(query, top_k)
        
        # Format context from retrieved documents
        context = self._format_context(results)
        
        # Create prompt
        prompt = self._create_chat_prompt(query, context, chat_history)
        
        # Generate answer
        if stream:
            return self.generate(prompt, max_tokens, temperature, stream=True)
        else:
            answer = self.generate(prompt, max_tokens, temperature)
            
            # Return answer with sources
            return {
                "answer": answer,
                "sources": [
                    {
                        "document_id": doc.get("document_id"),
                        "text": doc.get("text", "")[:200] + "...",  # Truncate for brevity
                        "score": doc.get("score"),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in results[:3]  # Include top 3 sources
                ]
            }
    
    def get_llm_info(self) -> Dict[str, Any]:
        """
        Get information about the language model.
        
        Returns:
            Dictionary with LLM information
        """
        return {
            "model_name": settings.LLM_MODEL,
            "vllm_enabled": settings.VLLM_ENABLED,
            "max_context_length": self.max_context_length
        }
