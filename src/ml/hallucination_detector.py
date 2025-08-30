#!/usr/bin/env python3
"""
Production-ready Hallucination Detection System
Streamlined version for Next.js integration
"""

import os
import sys
import json
import re
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Check and install required packages
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
required_packages = ['numpy', 'scikit-learn', 'joblib', 'nltk']
for pkg in required_packages:
    install_if_missing(pkg)

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# NLTK setup
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize, sent_tokenize
except:
    # Fallback if NLTK fails
    def word_tokenize(text):
        return text.split()
    def sent_tokenize(text):
        return [s.strip() for s in text.split('.') if s.strip()]

@dataclass
class DetectionResult:
    """Result of hallucination detection analysis"""
    hallucination_probability: float
    confidence_score: float
    detected_issues: List[str]
    recommendations: List[str]
    is_safe: bool
    detailed_metrics: Dict[str, float]

class HallucinationDetector:
    """Streamlined hallucination detection system for production"""
    
    def __init__(self):
        # Confidence indicators
        self.uncertainty_phrases = [
            "i think", "i believe", "possibly", "maybe", "perhaps", "might be",
            "could be", "seems like", "appears to", "i'm not sure", "i don't know",
            "it's unclear", "it's possible", "may indicate", "could suggest"
        ]
        
        self.overconfidence_phrases = [
            "definitely", "absolutely", "certainly", "without doubt", "guaranteed",
            "100% sure", "completely certain", "no question", "undoubtedly",
            "clearly", "obviously", "for sure", "no doubt"
        ]
        
        self.factual_indicators = [
            "according to", "research shows", "studies indicate", "data suggests",
            "statistics show", "evidence indicates", "scientists found", "experts agree"
        ]
        
        # Try to load pre-trained model
        self.model = None
        self.vectorizer = None
        self._load_model()
        
        # If no model exists, create a simple one
        if self.model is None:
            self._create_simple_model()
    
    def _load_model(self):
        """Load pre-trained model if available"""
        model_path = os.path.join(os.path.dirname(__file__), "hallucination_model.pkl")
        vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
        
        try:
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("OK: Loaded pre-trained model")
            else:
                print("WARNING: No pre-trained model found, will create simple model")
        except Exception as e:
            print(f"WARNING: Error loading model: {e}")
            self.model = None
            self.vectorizer = None
    
    def _create_simple_model(self):
        """Create and train a simple model for demonstration"""
        try:
            # Simple training data
            training_data = [
                ("The Eiffel Tower is located in Paris, France.", 0),
                ("Water boils at 100 degrees Celsius at sea level.", 0),
                ("The Earth orbits around the Sun.", 0),
                ("Research shows that exercise improves health.", 0),
                ("According to scientists, gravity pulls objects down.", 0),
                ("The moon is definitely made of cheese.", 1),
                ("I'm absolutely certain that dogs can fly.", 1),
                ("Without doubt, the ocean is made of chocolate.", 1),
                ("Everyone knows that the sky is always green.", 1),
                ("Cars obviously run on pure magic energy.", 1),
            ]
            
            texts, labels = zip(*training_data)
            
            # Train simple model
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = self.vectorizer.fit_transform(texts)
            
            self.model = LogisticRegression(random_state=42)
            self.model.fit(X, labels)
            
            # Save model
            model_dir = os.path.dirname(__file__)
            joblib.dump(self.model, os.path.join(model_dir, "hallucination_model.pkl"))
            joblib.dump(self.vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
            
            print("OK: Created and saved simple model")
        except Exception as e:
            print(f"ERROR: Error creating model: {e}")
            self.model = None
            self.vectorizer = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove URLs and special characters
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def analyze_confidence_patterns(self, text: str) -> float:
        """Analyze confidence patterns in text"""
        text_lower = text.lower()
        
        uncertainty_count = sum(1 for phrase in self.uncertainty_phrases if phrase in text_lower)
        overconfidence_count = sum(1 for phrase in self.overconfidence_phrases if phrase in text_lower)
        
        # Higher score = more problematic
        if uncertainty_count == 0 and overconfidence_count == 0:
            return 0.0
        
        # Overconfidence is more problematic than uncertainty
        score = (overconfidence_count * 0.7) + (uncertainty_count * 0.3)
        return min(score / 5, 1.0)  # Normalize to 0-1
    
    def calculate_factual_density(self, text: str) -> float:
        """Calculate factual density of text"""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        factual_sentences = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for factual indicators
            if any(indicator in sentence_lower for indicator in self.factual_indicators):
                factual_sentences += 1
            # Check for numbers, which often indicate facts
            elif re.search(r'\d+', sentence):
                factual_sentences += 0.5
        
        return min(factual_sentences / len(sentences), 1.0)
    
    def detect_contradictions(self, text: str) -> float:
        """Detect potential contradictions"""
        contradiction_indicators = ["but", "however", "although", "despite", "contrary"]
        text_lower = text.lower()
        
        contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in text_lower)
        sentences = sent_tokenize(text)
        
        if not sentences:
            return 0.0
        
        return min(contradiction_count / len(sentences), 1.0)
    
    def analyze_response(self, text: str, context: Optional[str] = None) -> DetectionResult:
        """Main analysis function"""
        try:
            if not text or not text.strip():
                return DetectionResult(
                    hallucination_probability=0.5,
                    confidence_score=0.5,
                    detected_issues=["Empty or invalid text"],
                    recommendations=["Please provide valid text to analyze"],
                    is_safe=False,
                    detailed_metrics={}
                )
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Calculate various metrics
            confidence_issues = self.analyze_confidence_patterns(text)
            factual_density = self.calculate_factual_density(text)
            contradiction_score = self.detect_contradictions(text)
            
            # ML model prediction if available
            ml_probability = 0.5
            if self.model and self.vectorizer:
                try:
                    X = self.vectorizer.transform([cleaned_text])
                    ml_probability = self.model.predict_proba(X)[0][1]  # Probability of hallucination
                except Exception as e:
                    print(f"WARNING: ML prediction error: {e}")
            
            # Combine metrics for final score
            metrics = {
                'confidence_issues': confidence_issues,
                'factual_density': factual_density,
                'contradiction_score': contradiction_score,
                'ml_probability': ml_probability
            }
            
            # Weighted combination
            weights = {
                'confidence_issues': 0.3,
                'factual_density': 0.2,  # Lower factual density = higher hallucination risk
                'contradiction_score': 0.2,
                'ml_probability': 0.3
            }
            
            hallucination_probability = (
                confidence_issues * weights['confidence_issues'] +
                (1.0 - factual_density) * weights['factual_density'] +  # Invert factual density
                contradiction_score * weights['contradiction_score'] +
                ml_probability * weights['ml_probability']
            )
            
            confidence_score = 1.0 - hallucination_probability
            is_safe = hallucination_probability < 0.6
            
            # Generate issues and recommendations
            detected_issues = []
            recommendations = []
            
            if confidence_issues > 0.5:
                detected_issues.append("Contains overconfident or uncertain language")
                recommendations.append("Use more balanced, evidence-based language")
            
            if factual_density < 0.3:
                detected_issues.append("Low factual content density")
                recommendations.append("Include more specific facts and references")
            
            if contradiction_score > 0.3:
                detected_issues.append("Contains potential contradictions")
                recommendations.append("Review for conflicting statements")
            
            if ml_probability > 0.7:
                detected_issues.append("ML model flagged as likely hallucination")
                recommendations.append("Consider fact-checking this content")
            
            if not detected_issues:
                detected_issues.append("No significant issues detected")
                recommendations.append("Content appears reliable")
            
            return DetectionResult(
                hallucination_probability=hallucination_probability,
                confidence_score=confidence_score,
                detected_issues=detected_issues,
                recommendations=recommendations,
                is_safe=is_safe,
                detailed_metrics=metrics
            )
            
        except Exception as e:
            return DetectionResult(
                hallucination_probability=0.5,
                confidence_score=0.5,
                detected_issues=[f"Analysis error: {str(e)}"],
                recommendations=["Please try again with different text"],
                is_safe=False,
                detailed_metrics={}
            )

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python hallucination_detector.py '<text to analyze>'")
        return
    
    text = sys.argv[1]
    detector = HallucinationDetector()
    result = detector.analyze_response(text)
    
    # Output as JSON for easy parsing
    output = {
        "hallucination_probability": float(result.hallucination_probability),
        "confidence_score": float(result.confidence_score),
        "detected_issues": result.detected_issues,
        "recommendations": result.recommendations,
        "is_safe": bool(result.is_safe),
        "detailed_metrics": {k: float(v) for k, v in result.detailed_metrics.items()}
    }
    
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
