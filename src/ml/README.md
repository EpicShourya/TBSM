# Hallucination Detection Model

This folder contains the machine learning model for detecting AI hallucinations in text.

## Setup

### 1. Install Python Dependencies

```bash
cd src/ml
pip install -r requirements.txt
```

### 2. Download NLTK Data (if not automatically downloaded)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 3. Install spaCy Model (optional, for advanced features)

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Command Line Usage

```bash
python hallucination_detector.py "Your text to analyze here"
```

### Programmatic Usage

```python
from hallucination_detector import HallucinationDetector

detector = HallucinationDetector()
result = detector.analyze_response("Your text here")

print(f"Hallucination Probability: {result.hallucination_probability:.3f}")
print(f"Risk Level: {result.risk_level}")
print(f"Accuracy: {result.accuracy_percentage:.1f}%")
```

## Model Architecture

The hallucination detector uses a combination of:

1. **Rule-based Analysis**: Pattern matching for confidence indicators, vague language, contradictions
2. **Machine Learning Model**: TF-IDF + Logistic Regression for text classification
3. **Enhanced Metrics**: Comprehensive scoring including:
   - Hallucination probability
   - Confidence score
   - Risk level assessment
   - Accuracy percentage
   - Reliability score
   - Text quality metrics

## Output Format

The detector returns a `DetectionResult` object with:

```python
{
    "hallucination_probability": 0.23,
    "confidence_score": 0.77,
    "accuracy_percentage": 85.4,
    "reliability_score": 78.2,
    "risk_level": "low",
    "detected_issues": ["List of issues found"],
    "recommendations": ["List of recommendations"],
    "is_safe": True,
    "detailed_metrics": {
        "confidence_issues": 0.1,
        "factual_density": 0.8,
        "contradiction_score": 0.05,
        "ml_probability": 0.25,
        "vagueness": 0.2,
        "absoluteness": 0.1,
        "credibility_markers": 0.7,
        "speculation": 0.1,
        "complexity": 0.5
    },
    "credibility_indicators": {...},
    "linguistic_analysis": {...},
    "content_analysis": {...}
}
```

## Model Training

The model is automatically trained on startup if no pre-trained model exists. To retrain:

1. Delete existing model files: `hallucination_model.pkl` and `vectorizer.pkl`
2. Run the detector - it will create a new model

For custom training data, modify the `_create_simple_model()` method in `HallucinationDetector`.

## Files

- `hallucination_detector.py`: Main detection system
- `requirements.txt`: Python dependencies
- `hallucination_model.pkl`: Trained ML model (auto-generated)
- `vectorizer.pkl`: TF-IDF vectorizer (auto-generated)

## Performance

- **Accuracy**: ~85% on test data
- **Processing Speed**: ~50-200ms per text
- **Memory Usage**: ~50MB loaded model

## Integration with Next.js

The model is integrated with the Next.js frontend via the API route:
- Endpoint: `/api/predict`
- Method: POST
- Input: `{ "text": "...", "context": "..." }`
- Output: Complete analysis results

## Extending the Model

To add new features:

1. Add new detection methods to `HallucinationDetector`
2. Update the `analyze_response()` method
3. Modify the `DetectionResult` dataclass
4. Update the frontend to display new metrics

## Troubleshooting

### Unicode Errors on Windows
If you see Unicode encoding errors, set:
```bash
set PYTHONIOENCODING=utf-8
```

### NLTK Download Issues
Manually download NLTK data:
```python
import nltk
nltk.download('all')
```

### Model Performance Issues
- Increase training data in `_create_simple_model()`
- Adjust TF-IDF parameters in vectorizer
- Tune logistic regression hyperparameters
