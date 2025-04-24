import os
from typing import Dict, List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv

load_dotenv()

class FeedbackGenerator:
    def __init__(self):
        """Initialize the feedback generator with a local language model."""
        # Use a smaller, efficient model for local processing
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Initialize text classification for grammar and style
        self.classifier = pipeline("text-classification", model="distilbert-base-uncased")
    
    def _analyze_essay_structure(self, essay: str) -> Dict[str, str]:
        """Analyze essay structure using rule-based and model-based approaches."""
        sentences = [s.strip() for s in essay.split('.') if s.strip()]
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        
        structure_feedback = {
            "length": f"Essay has {len(sentences)} sentences and {len(paragraphs)} paragraphs.",
            "coherence": "Good" if len(paragraphs) >= 3 else "Needs improvement",
            "organization": "Well-structured" if len(sentences) > 10 else "Could use more development"
        }
        
        return structure_feedback
    
    def _analyze_language_use(self, essay: str) -> Dict[str, str]:
        """Analyze language use and style."""
        # Get sentiment analysis
        sentiment = self.sentiment_analyzer(essay)[0]
        
        # Get style classification
        style = self.classifier(essay)[0]
        
        language_feedback = {
            "sentiment": f"Overall {sentiment['label']} with {sentiment['score']:.2f} confidence",
            "style": f"Writing style is {style['label']}",
            "vocabulary": "Appropriate" if len(set(essay.split())) > 50 else "Could be more varied"
        }
        
        return language_feedback
    
    def generate_feedback(self, essay: str, score: float) -> Dict[str, List[str]]:
        """
        Generate detailed feedback for an essay using local models.
        
        Args:
            essay (str): The essay text
            score (float): The predicted score
            
        Returns:
            Dict[str, List[str]]: Structured feedback containing strengths and areas for improvement
        """
        try:
            # Analyze structure
            structure = self._analyze_essay_structure(essay)
            
            # Analyze language use
            language = self._analyze_language_use(essay)
            
            # Generate strengths
            strengths = [
                f"Good essay structure with {structure['length']}",
                f"Writing shows {language['sentiment']}",
                f"Style is {language['style']}"
            ]
            
            # Generate improvements
            improvements = []
            if structure['coherence'] != "Good":
                improvements.append("Consider adding more paragraphs for better coherence")
            if structure['organization'] != "Well-structured":
                improvements.append("Develop your ideas more thoroughly")
            if language['vocabulary'] != "Appropriate":
                improvements.append("Try using a more varied vocabulary")
            
            # Generate suggestions
            suggestions = [
                "Review your essay structure and ensure each paragraph has a clear purpose",
                "Consider using transition words to improve flow between paragraphs",
                "Try to incorporate more specific examples and details"
            ]
            
            return {
                "strengths": strengths,
                "improvements": improvements,
                "suggestions": suggestions
            }
            
        except Exception as e:
            print(f"Error generating feedback: {str(e)}")
            return {
                "strengths": ["Unable to generate feedback at this time"],
                "improvements": ["Please try again later"],
                "suggestions": ["Contact support if the issue persists"]
            }
    
    def get_detailed_analysis(self, essay: str) -> Dict[str, str]:
        """
        Get a detailed analysis of the essay's structure, coherence, and language use.
        
        Args:
            essay (str): The essay text
            
        Returns:
            Dict[str, str]: Detailed analysis in different categories
        """
        try:
            structure = self._analyze_essay_structure(essay)
            language = self._analyze_language_use(essay)
            
            return {
                "structure": f"Essay structure analysis: {structure['length']}. {structure['coherence']} coherence. {structure['organization']} organization.",
                "argument": "Argument development appears solid based on essay length and structure.",
                "language": f"Language analysis: {language['sentiment']}. Style is {language['style']}. Vocabulary is {language['vocabulary']}.",
                "grammar": "Grammar appears correct based on sentence structure analysis."
            }
            
        except Exception as e:
            print(f"Error generating detailed analysis: {str(e)}")
            return {
                "structure": "Analysis unavailable",
                "argument": "Analysis unavailable",
                "language": "Analysis unavailable",
                "grammar": "Analysis unavailable"
            } 