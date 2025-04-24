import spacy
import re
from typing import List, Optional

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with spaCy model."""
        self.nlp = spacy.load("en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize the input text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using spaCy.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    def get_pos_tags(self, text: str) -> List[tuple]:
        """
        Get part-of-speech tags for the input text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[tuple]: List of (token, pos_tag) tuples
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_entities(self, text: str) -> List[tuple]:
        """
        Extract named entities from the input text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[tuple]: List of (entity_text, entity_label) tuples
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def preprocess_essay(self, text: str) -> dict:
        """
        Perform comprehensive preprocessing on an essay.
        
        Args:
            text (str): Input essay text
            
        Returns:
            dict: Dictionary containing preprocessed components
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        pos_tags = self.get_pos_tags(cleaned_text)
        entities = self.get_entities(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'entities': entities,
            'word_count': len(tokens),
            'sentence_count': len(list(self.nlp(text).sents))
        } 