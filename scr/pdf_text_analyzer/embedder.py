"""
Text embedding module for generating vector representations of text chunks.
"""
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config.config import settings

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Class to generate text embeddings using transformer models."""
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL, 
                 batch_size: int = settings.EMBEDDING_BATCH_SIZE,
                 device: Optional[str] = None):
        """
        Initialize the text embedder with a specific model.
        
        Args:
            model_name: Name or path of the embedding model
            batch_size: Batch size for embedding generation
            device: Device to run the model on (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(torch, 'has_mps') and torch.has_mps:
                self.device = "mps"  # For Apple Silicon
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device} for embeddings")
        
        # Load model and tokenizer
        logger.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenization
            
        Returns:
            Pooled embeddings
        """
        # First element of model_output contains token embeddings
        token_embeddings = model_output[0]
        
        # Expand attention mask from [batch_size, seq_length] to [batch_size, seq_length, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum the token embeddings * mask and divide by the sum of the mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def embed_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            NumPy array of embeddings, shape [len(texts), embedding_dim]
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            encoded_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_inputs)
            
            # Apply pooling
            embeddings = self._mean_pooling(model_output, encoded_inputs["attention_mask"])
            
            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Move to CPU and convert to NumPy
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize embedding to unit length
            
        Returns:
            NumPy array of embedding, shape [embedding_dim]
        """
        embeddings = self.embed_texts([text], normalize)
        return embeddings[0]

    def embed_documents(self, documents: List[Dict[str, Any]], text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document dictionaries.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to embed
            
        Returns:
            List of documents with 'embedding' field added
        """
        # Extract texts
        texts = [doc[text_field] for doc in documents if text_field in doc]
        
        if not texts:
            logger.warning("No texts found in documents")
            return documents
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to documents
        embedded_docs = []
        embedding_idx = 0
        
        for doc in documents:
            if text_field in doc:
                doc_copy = doc.copy()
                doc_copy["embedding"] = embeddings[embedding_idx].tolist()
                embedded_docs.append(doc_copy)
                embedding_idx += 1
            else:
                embedded_docs.append(doc)
        
        return embedded_docs

class BGEEmbedder(TextEmbedder):
    """Specialized embedder class for BGE models with instruction tuning support."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", 
                 batch_size: int = settings.EMBEDDING_BATCH_SIZE,
                 device: Optional[str] = None,
                 instruction: str = "Represent this document for retrieval:"):
        """
        Initialize the BGE embedder.
        
        Args:
            model_name: Name or path of the BGE model
            batch_size: Batch size for embedding generation
            device: Device to run the model on
            instruction: Instruction prefix for embedding generation
        """
        super().__init__(model_name, batch_size, device)
        self.instruction = instruction
    
    def embed_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts with instruction prefix.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            NumPy array of embeddings
        """
        # Add instruction prefix to each text
        instructed_texts = [f"{self.instruction} {text}" for text in texts]
        return super().embed_texts(instructed_texts, normalize)
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a query with a query-specific instruction.
        
        Args:
            query: Query text to embed
            normalize: Whether to normalize embedding to unit length
            
        Returns:
            NumPy array of embedding
        """
        # Use a different instruction for queries
        query_instruction = "Represent this question for retrieving relevant passages:"
        instructed_query = f"{query_instruction} {query}"
        
        # Tokenize
        encoded_input = self.tokenizer(
            instructed_query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Apply pooling
        embedding = self._mean_pooling(model_output, encoded_input["attention_mask"])
        
        # Normalize if requested
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Move to CPU and convert to NumPy
        return embedding.cpu().numpy()[0]
