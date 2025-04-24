import streamlit as st
import pandas as pd
from models.scorer import EssayScorer
from models.feedback import FeedbackGenerator
from utils.preprocessor import TextPreprocessor
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key="AIzaSyDeQICgWF5tdNELnZypWSrP98MIwrFFKnk")

# Set page config
st.set_page_config(
    page_title="EssayEval+",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("EssayEval+ 📝")
st.markdown("""
A hybrid AI-based essay scoring system that combines machine learning with advanced language models
to provide comprehensive feedback on your writing.
""")

# Initialize session state
if 'feedback' not in st.session_state:
    st.session_state.feedback = None

# Initialize components
@st.cache_resource
def load_components():
    try:
        return {
            'scorer': EssayScorer(),
            'feedback_generator': FeedbackGenerator(),
            'preprocessor': TextPreprocessor()
        }
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.stop()

# Load components
try:
    components = load_components()
except Exception as e:
    st.error(f"Error loading components: {str(e)}")
    st.stop()

# Input section
st.header("Input Your Essay")
input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])

essay_text = ""
if input_method == "Text Input":
    essay_text = st.text_area("Enter your essay here:", height=300)
else:
    uploaded_file = st.file_uploader("Upload your essay (PDF or TXT)", type=['pdf', 'txt'])
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            essay_text = uploaded_file.getvalue().decode("utf-8")
        else:
            st.error("Please upload a text file (.txt)")

# Process essay
if essay_text and st.button("Evaluate Essay"):
    with st.spinner("Analyzing your essay..."):
        try:
            # Preprocess essay
            preprocessed = components['preprocessor'].preprocess_essay(essay_text)
            
            # Get ML score
            ml_score = components['scorer'].predict(essay_text)
            
            # Generate feedback
            feedback = components['feedback_generator'].generate_feedback(essay_text, ml_score)
            
            # Store feedback in session state
            st.session_state.feedback = feedback
        except Exception as e:
            st.error(f"Error processing essay: {str(e)}")

# Display results
if st.session_state.feedback:
    feedback = st.session_state.feedback
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Score and Analysis")
        
        # Display final score with error handling
        if feedback.get('final_score') is not None:
            st.metric(
                "Final Score",
                f"{feedback['final_score']:.1f}/10",
                delta=f"{feedback['final_score'] - ml_score:.1f}" if ml_score is not None else None
            )
        else:
            st.error("Unable to calculate final score")
        
        # Score justification with error handling
        st.subheader("Score Justification")
        if feedback.get('score_justification'):
            st.write(feedback['score_justification'])
        else:
            st.write("No score justification available")
        
        # Grammar feedback with error handling
        st.subheader("Grammar and Spelling")
        if feedback.get('grammar_feedback'):
            for item in feedback['grammar_feedback']:
                st.write(f"• {item}")
        else:
            st.write("No grammar feedback available")
        
        # Structure feedback with error handling
        st.subheader("Structure and Organization")
        if feedback.get('structure_feedback'):
            for item in feedback['structure_feedback']:
                st.write(f"• {item}")
        else:
            st.write("No structure feedback available")
    
    with col2:
        st.header("Improvement Suggestions")
        
        # Clarity feedback with error handling
        st.subheader("Clarity and Coherence")
        if feedback.get('clarity_feedback'):
            for item in feedback['clarity_feedback']:
                st.write(f"• {item}")
        else:
            st.write("No clarity feedback available")
        
        # Improvement suggestions with error handling
        st.subheader("Specific Suggestions")
        if feedback.get('improvement_suggestions'):
            for item in feedback['improvement_suggestions']:
                st.write(f"• {item}")
        else:
            st.write("No improvement suggestions available")
        
        # Examples with error handling
        st.subheader("Examples")
        
        if feedback.get('good_examples'):
            st.write("**Well-written sentences:**")
            for example in feedback['good_examples']:
                st.write(f"✓ {example}")
        else:
            st.write("No examples of well-written sentences available")
        
        if feedback.get('problematic_examples'):
            st.write("**Sentences that need improvement:**")
            for example in feedback['problematic_examples']:
                st.write(f"⚠️ {example}")
            
            # Get rewritten suggestions with error handling
            if st.button("Show Improved Versions"):
                try:
                    suggestions = components['feedback_generator'].get_rewritten_suggestions(
                        feedback['problematic_examples']
                    )
                    
                    st.write("**Improved versions:**")
                    for original, improved in suggestions.items():
                        st.write(f"Original: {original}")
                        st.write(f"Improved: {improved}")
                        st.write("---")
                except Exception as e:
                    st.error(f"Error generating improved versions: {str(e)}")
        else:
            st.write("No problematic examples available")

# Footer
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)