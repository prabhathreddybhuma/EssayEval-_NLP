import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib
import os

from utils.embeddings import EmbeddingGenerator

class EssayScorer:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the essay scorer.
        
        Args:
            model_path (Optional[str]): Path to saved model file
        """
        self.embedding_generator = EmbeddingGenerator()
        self.scaler = StandardScaler()
        
        # Initialize with better default parameters
        self.model = Ridge(
            alpha=0.1,  # Reduced regularization
            fit_intercept=True,
            copy_X=True,
            max_iter=1000,
            tol=0.001,
            solver='auto'
        )
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with some basic training data
            self._initialize_with_basic_data()
    
    def _initialize_with_basic_data(self):
        """Initialize the model with some basic training examples."""
        # Basic training examples with varying quality
        essays = [
            "This is a very basic essay with simple sentences and limited vocabulary.",
            "A well-structured essay with clear arguments and good vocabulary usage.",
            "An excellent essay demonstrating deep analysis and sophisticated language.",
            "A poor essay with many grammatical errors and unclear arguments.",
            "A mediocre essay with some good points but lacking depth and coherence."
        ]
        scores = [3.0, 6.0, 9.0, 2.0, 5.0]
        
        # Generate embeddings
        embeddings = self.embedding_generator.get_batch_embeddings(essays)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(embeddings)
        
        # Train model
        self.model.fit(scaled_features, scores)
    
    def train(self, essays: list, scores: list) -> None:
        """
        Train the scoring model.
        
        Args:
            essays (list): List of essay texts
            scores (list): List of corresponding scores
        """
        # Generate embeddings
        embeddings = self.embedding_generator.get_batch_embeddings(essays)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(embeddings)
        
        # Train model
        self.model.fit(scaled_features, scores)
    
    def predict(self, essay: str) -> float:
        """
        Predict score for a single essay.
        
        Args:
            essay (str): Input essay text
            
        Returns:
            float: Predicted score
        """
        # Generate embedding
        embedding = self.embedding_generator.get_essay_embedding(essay)
        
        # Scale features
        scaled_features = self.scaler.transform(embedding.reshape(1, -1))
        
        # Predict score
        score = self.model.predict(scaled_features)[0]
        
        # Ensure score is within [0, 10] range and add some variance
        score = max(0, min(10, score))
        
        # Add some basic scoring factors
        word_count = len(essay.split())
        sentence_count = len([s for s in essay.split('.') if s.strip()])
        
        # Adjust score based on length and complexity
        if word_count < 100:
            score = max(0, score - 2)  # Penalize very short essays
        elif word_count > 500:
            score = min(10, score + 1)  # Reward longer essays
        
        if sentence_count < 5:
            score = max(0, score - 1)  # Penalize essays with few sentences
        
        return score
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model and scaler.
        
        Args:
            path (str): Path to the saved model
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
    
    def evaluate(self, essays: list, true_scores: list) -> Tuple[float, float]:
        """
        Evaluate the model on a test set.
        
        Args:
            essays (list): List of test essay texts
            true_scores (list): List of true scores
            
        Returns:
            Tuple[float, float]: Mean squared error and R² score
        """
        predictions = [self.predict(essay) for essay in essays]
        mse = np.mean((np.array(predictions) - np.array(true_scores)) ** 2)
        r2 = self.model.score(
            self.scaler.transform(self.embedding_generator.get_batch_embeddings(essays)),
            true_scores
        )
        return mse, r2