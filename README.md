# EssayEval+ — Hybrid AI-Based Automated Essay Scorer

A sophisticated essay scoring system that combines traditional NLP-based scoring with Google's Gemini API for comprehensive evaluation and feedback.

## Features

- Automated essay scoring using RoBERTa + Ridge Regression
- LLM-powered feedback using Google's Gemini API
- Real-time evaluation and writing suggestions
- User-friendly Streamlit interface
- Support for text input and file uploads

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/essay-eval-plus.git
cd essay-eval-plus
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Set up environment variables:
Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

3. Input your essay or upload a file

4. Receive comprehensive scoring and feedback

## Project Structure

```
essay-eval-plus/
├── app.py                 # Main Streamlit application
├── models/
│   ├── scorer.py         # ML-based scoring model
│   └── feedback.py       # LLM-based feedback generation
├── utils/
│   ├── preprocessor.py   # Text preprocessing utilities
│   └── embeddings.py     # Text embedding generation
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Model Architecture

The system uses a hybrid approach:
1. Initial scoring using RoBERTa embeddings and Ridge Regression
2. Score refinement and feedback generation using Google's Gemini API
3. Final score and detailed feedback presentation

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 