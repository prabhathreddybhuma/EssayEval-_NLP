import google.generativeai as genai
from typing import Dict, Optional
import os
from dotenv import load_dotenv
import streamlit as st

class FeedbackGenerator:
    def __init__(self):
        """Initialize the feedback generator with Google's Gemini API."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("""
            Google API Key not found. Please follow these steps:
            1. Create a file named '.env' in your project root directory
            2. Add your API key to the file like this:
               GOOGLE_API_KEY=your_api_key_here
            3. Restart the application
            """)
            st.stop()
        
        try:
            # Configure the API with the correct endpoint
            genai.configure(
                api_key=api_key,
                transport='rest',
                client_options={
                    'api_endpoint': 'https://generativelanguage.googleapis.com/v1beta'
                }
            )
            # Use the correct model name
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            st.error(f"Error initializing Gemini API: {str(e)}")
            st.stop()
    
    def generate_feedback(self, essay: str, ml_score: float) -> Dict:
        """
        Generate comprehensive feedback for an essay.
        
        Args:
            essay (str): Input essay text
            ml_score (float): Score predicted by ML model
            
        Returns:
            Dict: Dictionary containing feedback components
        """
        try:
            prompt = f"""
            Analyze the following essay and provide detailed feedback. The essay received a score of {ml_score:.1f}/10.
            
            Essay:
            {essay}
            
            Please provide:
            1. A detailed justification for the score
            2. Grammar and spelling feedback
            3. Structure and organization feedback
            4. Clarity and coherence feedback
            5. Specific improvement suggestions
            6. Examples of well-written and problematic sentences
            
            Format the response as a structured analysis.
            """
            
            # Generate content with proper configuration
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            
            # Parse the response into structured feedback
            feedback = self._parse_feedback(response.text)
            
            # Always use the ML score
            feedback['final_score'] = ml_score
            
            return feedback
        except Exception as e:
            st.error(f"Error generating feedback: {str(e)}")
            return {
                'final_score': ml_score,
                'score_justification': 'Error generating detailed feedback',
                'grammar_feedback': ['Error generating grammar feedback'],
                'structure_feedback': ['Error generating structure feedback'],
                'clarity_feedback': ['Error generating clarity feedback'],
                'improvement_suggestions': ['Error generating suggestions'],
                'good_examples': [],
                'problematic_examples': []
            }
    
    def _parse_feedback(self, response: str) -> Dict:
        """
        Parse the LLM response into structured feedback.
        
        Args:
            response (str): Raw LLM response
            
        Returns:
            Dict: Structured feedback dictionary
        """
        # Basic structure for feedback
        feedback = {
            'final_score': None,  # This will be overwritten with ML score
            'score_justification': '',
            'grammar_feedback': [],
            'structure_feedback': [],
            'clarity_feedback': [],
            'improvement_suggestions': [],
            'good_examples': [],
            'problematic_examples': []
        }
        
        try:
            # Split response into sections
            sections = response.split('\n\n')
            
            for section in sections:
                if 'justification' in section.lower():
                    # Extract justification
                    feedback['score_justification'] = section.strip()
                
                elif 'grammar' in section.lower():
                    feedback['grammar_feedback'] = [line.strip() for line in section.split('\n') if line.strip()]
                
                elif 'structure' in section.lower():
                    feedback['structure_feedback'] = [line.strip() for line in section.split('\n') if line.strip()]
                
                elif 'clarity' in section.lower():
                    feedback['clarity_feedback'] = [line.strip() for line in section.split('\n') if line.strip()]
                
                elif 'improvement' in section.lower():
                    feedback['improvement_suggestions'] = [line.strip() for line in section.split('\n') if line.strip()]
                
                elif 'good examples' in section.lower():
                    feedback['good_examples'] = [line.strip() for line in section.split('\n') if line.strip()]
                
                elif 'problematic' in section.lower():
                    feedback['problematic_examples'] = [line.strip() for line in section.split('\n') if line.strip()]
        except Exception as e:
            st.warning(f"Error parsing feedback: {str(e)}")
        
        return feedback
    
    def get_rewritten_suggestions(self, problematic_sentences: list) -> Dict[str, str]:
        """
        Get rewritten versions of problematic sentences.
        
        Args:
            problematic_sentences (list): List of sentences to improve
            
        Returns:
            Dict[str, str]: Dictionary mapping original sentences to improved versions
        """
        suggestions = {}
        
        try:
            for sentence in problematic_sentences:
                prompt = f"""
                Rewrite the following sentence to improve its clarity, grammar, and style:
                
                Original: {sentence}
                
                Provide the improved version and briefly explain the changes made.
                """
                
                # Generate content with proper configuration
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": 1024,
                    }
                )
                suggestions[sentence] = response.text
        except Exception as e:
            st.error(f"Error generating rewritten suggestions: {str(e)}")
        
        return suggestions 