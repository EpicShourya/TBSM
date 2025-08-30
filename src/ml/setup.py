#!/usr/bin/env python3
"""
Setup script for the Hallucination Detection Model
Installs dependencies and downloads required data
"""

import os
import sys
import subprocess

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False

def download_spacy_model():
    """Download spaCy English model (optional)"""
    print("Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Could not download spaCy model: {e}")
        print("  This is optional and won't affect basic functionality")
        return False

def test_installation():
    """Test if the hallucination detector works"""
    print("Testing hallucination detector...")
    try:
        from hallucination_detector import HallucinationDetector
        detector = HallucinationDetector()
        result = detector.analyze_response("The Eiffel Tower is located in Paris.")
        
        if result and hasattr(result, 'hallucination_probability'):
            print("✓ Hallucination detector is working correctly")
            print(f"  Test result: {result.hallucination_probability:.3f} hallucination probability")
            return True
        else:
            print("✗ Error: Detector returned invalid result")
            return False
            
    except Exception as e:
        print(f"✗ Error testing detector: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("Hallucination Detection Model Setup")
    print("=" * 60)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    print()
    
    # Download NLTK data
    if not download_nltk_data():
        success = False
    
    print()
    
    # Download spaCy model (optional)
    download_spacy_model()
    
    print()
    
    # Test installation
    if success and test_installation():
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("=" * 60)
        print("\nYou can now use the hallucination detector:")
        print("  python hallucination_detector.py \"Your text here\"")
        print("\nOr import it in your Python code:")
        print("  from hallucination_detector import HallucinationDetector")
    else:
        print("\n" + "=" * 60)
        print("❌ Setup encountered some issues")
        print("=" * 60)
        print("\nPlease check the error messages above and try again.")
        print("You may need to install dependencies manually:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
