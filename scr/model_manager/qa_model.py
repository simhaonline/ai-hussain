"""
Question-answering model implementation.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from config.config import settings
from src.model_manager.rag_model import RAGModel

logger = logging.getLogger(__name__)

class QAModel:
    """
    Question-answering model implementation.
    
    This class provides specialized question-answering capabilities
    building on top of the RAG model.
    """
    
    def __init__(self, rag_model: Optional[RAGModel] = None):
        """
        Initialize the QA model.
        
        Args:
            rag_model: RAG model instance
        """
        self.rag_model = rag_model or RAGModel()
    
    def answer_question(self, question: str,
                       top_k: int = 5,
                       max_tokens: int = 512,
                       temperature: float = 0.7) -> Dict[str, Any]:
        """
        Answer a question using retrieved context.
        
        Args:
            question: Question to answer
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer, confidence, and sources
        """
        # Use RAG model to get answer
        result = self.rag_model.answer(
            query=question,
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        answer = result["answer"]
        sources = result["sources"]
        
        # Estimate confidence based on source relevance
        confidence = self._estimate_confidence(question, answer, sources)
        
        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": sources
        }
    
    def _estimate_confidence(self, question: str, answer: str, sources: List[Dict[str, Any]]) -> float:
        """
        Estimate confidence in the answer based on sources.
        
        Args:
            question: Original question
            answer: Generated answer
            sources: Retrieved sources
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence estimation based on source scores
        if not sources:
            return 0.1
        
        # Calculate average source score
        avg_score = sum(source.get("score", 0) for source in sources) / len(sources)
        
        # Check if answer indicates uncertainty
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i am not sure", 
            "cannot answer", "unable to answer", "don't have enough information"
        ]
        
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            # If model expressed uncertainty, reduce confidence
            return min(0.3, avg_score)
        
        # Use source scores as base confidence and apply heuristics
        confidence = avg_score
        
        # Boost confidence if answer is concise and not too generic
        if 50 < len(answer) < 500 and "context" not in answer.lower():
            confidence = min(1.0, confidence * 1.2)
        
        return confidence
    
    def answer_multiple_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of answer dictionaries
        """
        answers = []
        
        for question in questions:
            try:
                answer = self.answer_question(question)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                answers.append({
                    "question": question,
                    "answer": "Failed to generate an answer.",
                    "confidence": 0.0,
                    "sources": []
                })
        
        return answers
    
    def answer_with_citations(self, question: str) -> Dict[str, Any]:
        """
        Answer a question with explicit citations to sources.
        
        Args:
            question: Question to answer
            
        Returns:
            Dictionary with answer and cited sources
        """
        # Get answer using RAG
        result = self.rag_model.answer(query=question)
        answer = result["answer"]
        sources = result["sources"]
        
        # Extract or insert citations in the answer
        cited_answer, citations = self._format_citations(answer, sources)
        
        return {
            "question": question,
            "answer": cited_answer,
            "citations": citations
        }
    
    def _format_citations(self, answer: str, sources: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format answer with citations.
        
        Args:
            answer: Generated answer
            sources: Retrieved sources
            
        Returns:
            Tuple of (answer with citations, citation list)
        """
        # Check if answer already contains citations like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        existing_citations = re.findall(citation_pattern, answer)
        
        citations = []
        
        if existing_citations:
            # Extract existing citations
            for i, source in enumerate(sources[:len(existing_citations)]):
                citation_num = int(existing_citations[i]) if i < len(existing_citations) else i + 1
                
                citations.append({
                    "id": citation_num,
                    "document_id": source.get("document_id"),
                    "text": source.get("text", ""),
                    "metadata": source.get("metadata", {})
                })
        else:
            # Add citations if not present
            cited_answer = answer
            
            for i, source in enumerate(sources[:3]):  # Use top 3 sources
                citation_num = i + 1
                doc_id = source.get("document_id", "unknown")
                
                citation = f"[{citation_num}]"
                
                # Add citation to end of answer if not already cited
                if citation not in cited_answer:
                    cited_answer += f" {citation}"
                
                citations.append({
                    "id": citation_num,
                    "document_id": doc_id,
                    "text": source.get("text", ""),
                    "metadata": source.get("metadata", {})
                })
            
            answer = cited_answer
        
        return answer, citations
    
    def extract_facts(self, text: str) -> List[str]:
        """
        Extract factual statements from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted facts
        """
        # Create a prompt for fact extraction
        prompt = f"""
Extract the key factual statements from the following text. 
Return only the facts, one per line.

Text:
{text}

Facts:
"""
        
        # Generate facts
        facts_text = self.rag_model.generate(prompt, max_tokens=512, temperature=0.0)
        
        # Parse facts (one per line)
        facts = [line.strip() for line in facts_text.split('\n') if line.strip()]
        
        return facts
