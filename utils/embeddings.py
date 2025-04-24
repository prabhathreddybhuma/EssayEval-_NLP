import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "roberta-base"):
        """
        Initialize the embedding generator with RoBERTa model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def generate_embeddings(self, texts: Union[str, List[str]], max_length: int = 512) -> np.ndarray:
        """
        Generate embeddings for input text(s) using RoBERTa.
        
        Args:
            texts (Union[str, List[str]]): Input text or list of texts
            max_length (int): Maximum sequence length
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize and prepare input
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def get_essay_embedding(self, essay: str) -> np.ndarray:
        """
        Generate embedding for a single essay.
        
        Args:
            essay (str): Input essay text
            
        Returns:
            np.ndarray: Essay embedding
        """
        return self.generate_embeddings(essay)[0]
    
    def get_batch_embeddings(self, essays: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of essays.
        
        Args:
            essays (List[str]): List of essay texts
            
        Returns:
            np.ndarray: Array of essay embeddings
        """
        return self.generate_embeddings(essays) 