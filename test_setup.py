#!/usr/bin/env python3
"""
Test Setup Script for Healthcare Chatbot
Comprehensive testing of all dependencies, imports, and data files
"""

import sys
import os
import importlib
import warnings
warnings.filterwarnings('ignore')

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10+")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("\n📦 Testing Dependencies...")
    dependencies = [
        'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
        'joblib', 'spacy', 'pyttsx3', 'requests'
    ]
    
    failed_deps = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            failed_deps.append(dep)
    
    return len(failed_deps) == 0

def test_project_imports():
    """Test project-specific imports"""
    print("\n🔧 Testing Project Imports...")
    
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    project_modules = [
        'config', 'data_preprocessing_fixed', 'chatbot_interface',
        'train_with_ai_augmented', 'diagnose_model'
    ]
    
    failed_imports = []
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_data_files():
    """Test if all required data files exist"""
    print("\n📁 Testing Data Files...")
    
    data_files = [
        "Data/Training.csv",
        "Data/Training_safe_augmented.csv", 
        "Data/Training_ai_augmented.csv",
        "Data/symptom_Description.csv",
        "Data/symptom_precaution.csv",
        "Data/Symptom_severity.csv"
    ]
    
    missing_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_model_files():
    """Test if model files exist"""
    print("\n🤖 Testing Model Files...")
    
    model_files = [
        "models/enhanced_ai_augmented_model.joblib",
        "models/enhanced_ai_augmented_label_encoder.joblib"
    ]
    
    missing_models = []
    for model_path in model_files:
        if os.path.exists(model_path):
            print(f"✅ {model_path}")
        else:
            print(f"⚠️  {model_path} - Missing (will be created during training)")
            missing_models.append(model_path)
    
    return len(missing_models) == 0

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing Data Loading...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data_preprocessing_fixed import DataPreprocessorFixed
        
        preprocessor = DataPreprocessorFixed()
        preprocessor.load_datasets(use_augmented=False, use_ai_augmented=False, use_safe_augmented=True)
        print("✅ Data loading successful")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n🤖 Testing Model Loading...")
    
    model_path = "models/enhanced_ai_augmented_model.joblib"
    encoder_path = "models/enhanced_ai_augmented_label_encoder.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("⚠️  Model files not found - skipping model loading test")
        return True
    
    try:
        import joblib
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        print("✅ Model loading successful")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_spacy_model():
    """Test spacy language model"""
    print("\n🗣️  Testing Spacy Language Model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ Spacy language model loaded successfully")
        return True
    except OSError:
        print("❌ Spacy language model not found")
        print("💡 Install with: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"❌ Spacy error: {e}")
        return False

def test_text_to_speech():
    """Test text-to-speech functionality"""
    print("\n🔊 Testing Text-to-Speech...")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("✅ Text-to-Speech initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️  Text-to-Speech initialization failed: {e}")
        print("💡 This is optional and won't affect core functionality")
        return True  # Don't fail the test for TTS issues

def main():
    """Run all tests"""
    print("🧪 HEALTHCARE CHATBOT SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Project Imports", test_project_imports),
        ("Data Files", test_data_files),
        ("Model Files", test_model_files),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Spacy Model", test_spacy_model),
        ("Text-to-Speech", test_text_to_speech)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📋 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The chatbot is ready to use.")
        print("\n🚀 To start the chatbot, run:")
        print("   python main.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("\n💡 Common solutions:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Install spacy model: python -m spacy download en_core_web_sm")
        print("   3. Check that all data files are in the Data/ directory")
        print("   4. Train the model first: python train_with_ai_augmented.py")

if __name__ == "__main__":
    main() 