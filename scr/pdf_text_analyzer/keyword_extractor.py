"""
Keyword extraction module for identifying important terms in documents.
"""
import logging
import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Class to extract keywords and keyphrases from text."""
    
    def __init__(self, language: str = "en", 
                 min_word_length: int = 3,
                 use_spacy: bool = True,
                 spacy_model: str = "en_core_web_sm",
                 max_keywords: int = 20):
        """
        Initialize the keyword extractor.
        
        Args:
            language: Language code (en, es, fr, etc.)
            min_word_length: Minimum length of words to consider
            use_spacy: Whether to use spaCy for NER and noun phrase extraction
            spacy_model: spaCy model to use
            max_keywords: Maximum number of keywords to extract
        """
        self.language = language
        self.min_word_length = min_word_length
        self.max_keywords = max_keywords
        self.use_spacy = use_spacy
        
        # Common English stopwords + common PDF terms
        self.stopwords = set(SPACY_STOP_WORDS).union({
            "page", "section", "figure", "table", "chapter", "pdf", "document", 
            "copyright", "all", "rights", "reserved", "et", "al", "etc"
        })
        
        # Initialize spaCy if needed
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.use_spacy = False
    
    def extract_keywords_tfidf(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF scoring.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        # Clean text
        clean_text = self._clean_text(text)
        if not clean_text:
            return []
        
        # Create a single-document corpus 
        corpus = [clean_text]
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(
            max_df=0.85,  # Ignore words that appear in >85% of documents
            min_df=1,     # Keep words that appear in at least 1 document
            stop_words=list(self.stopwords),
            use_idf=True,
            ngram_range=(1, 1),  # Use single words only
            max_features=5000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray().flatten()
            
            # Create (keyword, score) tuples and sort by score
            keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out keywords that are too short
            keyword_scores = [(k, s) for k, s in keyword_scores if len(k) >= self.min_word_length]
            
            return keyword_scores[:top_n]
        
        except Exception as e:
            logger.error(f"Error in TF-IDF keyword extraction: {e}")
            return []
    
    def extract_keyphrases(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract multi-word keyphrases using TF-IDF and n-grams.
        
        Args:
            text: Text to extract keyphrases from
            top_n: Number of top keyphrases to return
            
        Returns:
            List of (keyphrase, score) tuples
        """
        # Clean text
        clean_text = self._clean_text(text)
        if not clean_text:
            return []
        
        # Create a single-document corpus 
        corpus = [clean_text]
        
        # Apply TF-IDF with n-grams
        vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=1,
            stop_words=list(self.stopwords),
            use_idf=True,
            ngram_range=(2, 3),  # Use 2-3 word phrases
            max_features=5000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray().flatten()
            
            # Create (keyphrase, score) tuples and sort by score
            keyphrase_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keyphrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyphrase_scores[:top_n]
        
        except Exception as e:
            logger.error(f"Error in keyphrase extraction: {e}")
            return []
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities using spaCy NER.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of (entity_text, entity_type) tuples
        """
        if not self.use_spacy or not self.nlp:
            logger.warning("spaCy not available for named entity extraction")
            return []
        
        try:
            # Process with spaCy
            doc = self.nlp(text[:1000000])  # Limit size to avoid memory issues
            
            # Extract entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            return entities
        
        except Exception as e:
            logger.error(f"Error in named entity extraction: {e}")
            return []
    
    def extract_noun_phrases(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract noun phrases using spaCy.
        
        Args:
            text: Text to extract noun phrases from
            top_n: Number of top noun phrases to return
            
        Returns:
            List of noun phrases
        """
        if not self.use_spacy or not self.nlp:
            logger.warning("spaCy not available for noun phrase extraction")
            return []
        
        try:
            # Process with spaCy
            doc = self.nlp(text[:1000000])  # Limit size to avoid memory issues
            
            # Extract noun phrases
            noun_phrases = []
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                # Filter out short phrases and those with stopwords
                if (len(phrase) >= self.min_word_length and 
                    not any(word in self.stopwords for word in phrase.split())):
                    noun_phrases.append(phrase)
            
            # Count frequencies
            phrase_counts = Counter(noun_phrases)
            
            # Get most common phrases
            return [phrase for phrase, _ in phrase_counts.most_common(top_n)]
        
        except Exception as e:
            logger.error(f"Error in noun phrase extraction: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing punctuation, extra whitespace, etc.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def extract_all_keywords(self, text: str) -> Dict[str, Any]:
        """
        Extract keywords, keyphrases, entities, and noun phrases from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            Dictionary with all extracted information
        """
        result = {
            "keywords": [],
            "keyphrases": [],
            "entities": [],
            "noun_phrases": []
        }
        
        # Extract keywords
        keywords = self.extract_keywords_tfidf(text, top_n=self.max_keywords)
        result["keywords"] = [{"text": k, "score": float(s)} for k, s in keywords]
        
        # Extract keyphrases
        keyphrases = self.extract_keyphrases(text, top_n=self.max_keywords)
        result["keyphrases"] = [{"text": k, "score": float(s)} for k, s in keyphrases]
        
        # Extract entities if spaCy is available
        if self.use_spacy:
            entities = self.extract_named_entities(text)
            result["entities"] = [{"text": e, "type": t} for e, t in entities]
            
            # Extract noun phrases
            noun_phrases = self.extract_noun_phrases(text, top_n=self.max_keywords)
            result["noun_phrases"] = noun_phrases
        
        return result
