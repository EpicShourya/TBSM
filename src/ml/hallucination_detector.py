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
    """Enhanced result of hallucination detection analysis"""
    hallucination_probability: float
    confidence_score: float
    detected_issues: List[str]
    recommendations: List[str]
    is_safe: bool
    detailed_metrics: Dict[str, float]
    hallucination_ratio: float
    risk_level: str
    text_quality_score: float
    credibility_indicators: Dict[str, Any]
    linguistic_analysis: Dict[str, float]
    content_analysis: Dict[str, float]
    accuracy_percentage: float
    prediction_confidence: float
    reliability_score: float

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
        
        # Enhanced detection patterns
        self.vague_terms = [
            "some", "many", "often", "usually", "generally", "typically", "sometimes",
            "might", "could", "would", "should", "probably", "likely"
        ]
        
        self.absolute_terms = [
            "always", "never", "all", "none", "every", "everything", "nothing",
            "completely", "totally", "entirely", "perfectly", "impossible"
        ]
        
        self.credibility_markers = [
            "peer-reviewed", "published", "verified", "confirmed", "documented",
            "official", "authenticated", "validated", "certified", "endorsed"
        ]
        
        self.speculation_markers = [
            "rumor", "allegedly", "supposedly", "claimed", "reported", "said to be",
            "believed to be", "thought to be", "appears to be", "seems to be"
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
    
    def analyze_vagueness(self, text: str) -> float:
        """Analyze the vagueness of language used"""
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        if not words:
            return 0.0
        
        vague_count = sum(1 for word in words if word in self.vague_terms)
        return min(vague_count / len(words) * 5, 1.0)  # Scale up for visibility
    
    def analyze_absoluteness(self, text: str) -> float:
        """Analyze use of absolute terms which can indicate overconfidence"""
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        if not words:
            return 0.0
        
        absolute_count = sum(1 for word in words if word in self.absolute_terms)
        return min(absolute_count / len(words) * 10, 1.0)  # Scale up for visibility
    
    def analyze_credibility_markers(self, text: str) -> float:
        """Analyze presence of credibility markers"""
        text_lower = text.lower()
        credibility_count = sum(1 for marker in self.credibility_markers if marker in text_lower)
        sentences = sent_tokenize(text)
        
        if not sentences:
            return 0.0
        
        return min(credibility_count / len(sentences), 1.0)
    
    def analyze_speculation(self, text: str) -> float:
        """Analyze speculative language"""
        text_lower = text.lower()
        speculation_count = sum(1 for marker in self.speculation_markers if marker in text_lower)
        sentences = sent_tokenize(text)
        
        if not sentences:
            return 0.0
        
        return min(speculation_count / len(sentences), 1.0)
    
    def calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity based on sentence and word length"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Normalize complexity score (higher = more complex)
        complexity = min((avg_sentence_length / 20) + (avg_word_length / 10), 1.0)
        return complexity
    
    def calculate_hallucination_ratio(self, text: str, metrics: Dict[str, float]) -> float:
        """Calculate a comprehensive hallucination ratio"""
        # Weight different factors
        weights = {
            'overconfidence': 0.25,
            'vagueness': 0.15,
            'speculation': 0.20,
            'contradiction': 0.20,
            'low_credibility': 0.15,
            'ml_probability': 0.05
        }
        
        overconfidence = metrics.get('confidence_issues', 0) + metrics.get('absoluteness', 0)
        vagueness = metrics.get('vagueness', 0)
        speculation = metrics.get('speculation', 0)
        contradiction = metrics.get('contradiction_score', 0)
        low_credibility = 1.0 - metrics.get('credibility_markers', 0)
        ml_prob = metrics.get('ml_probability', 0.5)
        
        ratio = (
            overconfidence * weights['overconfidence'] +
            vagueness * weights['vagueness'] +
            speculation * weights['speculation'] +
            contradiction * weights['contradiction'] +
            low_credibility * weights['low_credibility'] +
            ml_prob * weights['ml_probability']
        )
        
        return min(ratio, 1.0)
    
    def determine_risk_level(self, hallucination_probability: float) -> str:
        """Determine risk level based on probability"""
        if hallucination_probability < 0.2:
            return "Very Low"
        elif hallucination_probability < 0.4:
            return "Low"
        elif hallucination_probability < 0.6:
            return "Medium"
        elif hallucination_probability < 0.8:
            return "High"
        else:
            return "Very High"
    
    def calculate_text_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall text quality score"""
        quality_factors = {
            'factual_density': metrics.get('factual_density', 0) * 0.3,
            'credibility': metrics.get('credibility_markers', 0) * 0.3,
            'clarity': (1.0 - metrics.get('vagueness', 0)) * 0.2,
            'consistency': (1.0 - metrics.get('contradiction_score', 0)) * 0.2
        }
        
        return sum(quality_factors.values())
    
    def calculate_accuracy_percentage(self, metrics: Dict[str, float]) -> float:
        """Calculate accuracy percentage based on multiple reliability factors"""
        # Factors that contribute to accuracy
        factual_accuracy = metrics.get('factual_density', 0) * 0.35
        source_reliability = metrics.get('credibility_markers', 0) * 0.25
        logical_consistency = (1.0 - metrics.get('contradiction_score', 0)) * 0.20
        language_precision = (1.0 - metrics.get('vagueness', 0)) * 0.10
        balanced_confidence = (1.0 - metrics.get('confidence_issues', 0)) * 0.10
        
        accuracy = (factual_accuracy + source_reliability + logical_consistency + 
                   language_precision + balanced_confidence)
        
        return min(accuracy * 100, 100.0)  # Convert to percentage and cap at 100%
    
    def calculate_prediction_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence in the prediction itself"""
        # How confident we are in our analysis
        ml_confidence = 1.0 - abs(0.5 - metrics.get('ml_probability', 0.5)) * 2  # Distance from 50/50
        feature_consistency = 1.0 - np.std([
            metrics.get('confidence_issues', 0),
            metrics.get('contradiction_score', 0),
            metrics.get('vagueness', 0),
            metrics.get('speculation', 0)
        ])
        
        text_length_factor = min(len(self.current_text.split()) / 50, 1.0) if hasattr(self, 'current_text') else 0.5
        
        confidence = (ml_confidence * 0.5 + feature_consistency * 0.3 + text_length_factor * 0.2)
        return min(confidence * 100, 100.0)
    
    def calculate_reliability_score(self, metrics: Dict[str, float], accuracy: float) -> float:
        """Calculate overall reliability score combining accuracy and other factors"""
        # Base reliability on accuracy
        base_reliability = accuracy / 100
        
        # Adjust for risk factors
        risk_adjustment = 1.0 - (
            metrics.get('speculation', 0) * 0.3 +
            metrics.get('absoluteness', 0) * 0.2 +
            metrics.get('contradiction_score', 0) * 0.3 +
            (1.0 - metrics.get('credibility_markers', 0)) * 0.2
        )
        
        reliability = (base_reliability * 0.7 + risk_adjustment * 0.3)
        return min(reliability * 100, 100.0)
    
    def analyze_response(self, text: str, context: Optional[str] = None) -> DetectionResult:
        """Enhanced main analysis function with comprehensive metrics"""
        try:
            if not text or not text.strip():
                return DetectionResult(
                    hallucination_probability=0.5,
                    confidence_score=0.5,
                    detected_issues=["Empty or invalid text"],
                    recommendations=["Please provide valid text to analyze"],
                    is_safe=False,
                    detailed_metrics={},
                    hallucination_ratio=0.5,
                    risk_level="Medium",
                    text_quality_score=0.0,
                    credibility_indicators={},
                    linguistic_analysis={},
                    content_analysis={},
                    accuracy_percentage=0.0,
                    prediction_confidence=0.0,
                    reliability_score=0.0
                )
            
            # Store current text for analysis
            self.current_text = text
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Calculate all metrics
            confidence_issues = self.analyze_confidence_patterns(text)
            factual_density = self.calculate_factual_density(text)
            contradiction_score = self.detect_contradictions(text)
            vagueness = self.analyze_vagueness(text)
            absoluteness = self.analyze_absoluteness(text)
            credibility_markers = self.analyze_credibility_markers(text)
            speculation = self.analyze_speculation(text)
            complexity = self.calculate_text_complexity(text)
            
            # ML model prediction if available
            ml_probability = 0.5
            if self.model and self.vectorizer:
                try:
                    X = self.vectorizer.transform([cleaned_text])
                    ml_probability = self.model.predict_proba(X)[0][1]
                except Exception as e:
                    print(f"WARNING: ML prediction error: {e}")
            
            # Comprehensive metrics
            detailed_metrics = {
                'confidence_issues': confidence_issues,
                'factual_density': factual_density,
                'contradiction_score': contradiction_score,
                'ml_probability': ml_probability,
                'vagueness': vagueness,
                'absoluteness': absoluteness,
                'credibility_markers': credibility_markers,
                'speculation': speculation,
                'complexity': complexity
            }
            
            # Calculate new accuracy metrics
            accuracy_percentage = self.calculate_accuracy_percentage(detailed_metrics)
            prediction_confidence = self.calculate_prediction_confidence(detailed_metrics)
            
            # Calculate hallucination ratio
            hallucination_ratio = self.calculate_hallucination_ratio(text, detailed_metrics)
            
            # Enhanced weighted combination for main probability
            weights = {
                'confidence_issues': 0.20,
                'factual_density': 0.15,
                'contradiction_score': 0.15,
                'ml_probability': 0.20,
                'vagueness': 0.10,
                'absoluteness': 0.10,
                'speculation': 0.10
            }
            
            hallucination_probability = (
                confidence_issues * weights['confidence_issues'] +
                (1.0 - factual_density) * weights['factual_density'] +
                contradiction_score * weights['contradiction_score'] +
                ml_probability * weights['ml_probability'] +
                vagueness * weights['vagueness'] +
                absoluteness * weights['absoluteness'] +
                speculation * weights['speculation']
            )
            
            confidence_score = 1.0 - hallucination_probability
            is_safe = hallucination_probability < 0.6
            risk_level = self.determine_risk_level(hallucination_probability)
            text_quality_score = self.calculate_text_quality_score(detailed_metrics)
            reliability_score = self.calculate_reliability_score(detailed_metrics, accuracy_percentage)
            
            # Enhanced issue detection
            detected_issues = []
            recommendations = []
            
            if accuracy_percentage < 60:
                detected_issues.append(f"Low accuracy score: {accuracy_percentage:.1f}%")
                recommendations.append("Improve factual content and source credibility")
            
            if confidence_issues > 0.4:
                detected_issues.append("High confidence inconsistency detected")
                recommendations.append("Balance uncertain and confident statements")
            
            if factual_density < 0.3:
                detected_issues.append("Low factual content density")
                recommendations.append("Include more specific facts and references")
            
            if contradiction_score > 0.3:
                detected_issues.append("Contradictory statements found")
                recommendations.append("Review for logical consistency")
            
            if vagueness > 0.4:
                detected_issues.append("Excessive use of vague language")
                recommendations.append("Be more specific and precise")
            
            if absoluteness > 0.3:
                detected_issues.append("Overuse of absolute terms")
                recommendations.append("Consider using more nuanced language")
            
            if speculation > 0.4:
                detected_issues.append("High speculation markers detected")
                recommendations.append("Verify claims with reliable sources")
            
            if credibility_markers < 0.1:
                detected_issues.append("Lack of credibility indicators")
                recommendations.append("Include references to authoritative sources")
            
            if ml_probability > 0.7:
                detected_issues.append("ML model flagged as likely hallucination")
                recommendations.append("Consider comprehensive fact-checking")
            
            if prediction_confidence < 70:
                detected_issues.append(f"Low prediction confidence: {prediction_confidence:.1f}%")
                recommendations.append("Consider providing more context or longer text")
            
            if not detected_issues:
                detected_issues.append("No significant issues detected")
                recommendations.append("Content appears reliable and well-structured")
            
            # Credibility indicators analysis
            credibility_indicators = {
                'has_sources': credibility_markers > 0.1,
                'factual_content': factual_density > 0.5,
                'balanced_confidence': confidence_issues < 0.3,
                'low_speculation': speculation < 0.3,
                'consistent_logic': contradiction_score < 0.2,
                'high_accuracy': accuracy_percentage > 70,
                'reliable_prediction': prediction_confidence > 70
            }
            
            # Linguistic analysis
            linguistic_analysis = {
                'vagueness_score': vagueness,
                'absoluteness_score': absoluteness,
                'speculation_score': speculation,
                'complexity_score': complexity,
                'confidence_pattern_score': confidence_issues
            }
            
            # Content analysis
            content_analysis = {
                'factual_density': factual_density,
                'credibility_markers': credibility_markers,
                'contradiction_level': contradiction_score,
                'overall_quality': text_quality_score,
                'accuracy_level': accuracy_percentage / 100,
                'reliability_level': reliability_score / 100
            }
            
            return DetectionResult(
                hallucination_probability=hallucination_probability,
                confidence_score=confidence_score,
                detected_issues=detected_issues,
                recommendations=recommendations,
                is_safe=is_safe,
                detailed_metrics=detailed_metrics,
                hallucination_ratio=hallucination_ratio,
                risk_level=risk_level,
                text_quality_score=text_quality_score,
                credibility_indicators=credibility_indicators,
                linguistic_analysis=linguistic_analysis,
                content_analysis=content_analysis,
                accuracy_percentage=accuracy_percentage,
                prediction_confidence=prediction_confidence,
                reliability_score=reliability_score
            )
            
        except Exception as e:
            return DetectionResult(
                hallucination_probability=0.5,
                confidence_score=0.5,
                detected_issues=[f"Analysis error: {str(e)}"],
                recommendations=["Please try again with different text"],
                is_safe=False,
                detailed_metrics={},
                hallucination_ratio=0.5,
                risk_level="Medium",
                text_quality_score=0.0,
                credibility_indicators={},
                linguistic_analysis={},
                content_analysis={},
                accuracy_percentage=0.0,
                prediction_confidence=0.0,
                reliability_score=0.0
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
        "detailed_metrics": {k: float(v) for k, v in result.detailed_metrics.items()},
        "hallucination_ratio": float(result.hallucination_ratio),
        "risk_level": result.risk_level,
        "text_quality_score": float(result.text_quality_score),
        "credibility_indicators": {k: bool(v) if isinstance(v, bool) else float(v) for k, v in result.credibility_indicators.items()},
        "linguistic_analysis": {k: float(v) for k, v in result.linguistic_analysis.items()},
        "content_analysis": {k: float(v) for k, v in result.content_analysis.items()},
        "accuracy_percentage": float(result.accuracy_percentage),
        "prediction_confidence": float(result.prediction_confidence),
        "reliability_score": float(result.reliability_score)
    }
    
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
