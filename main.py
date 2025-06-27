#!/usr/bin/env python3
"""
Main Application - ENHANCED VERSION
Symptom-Based Disease Prediction Chatbot with AI and Machine Learning
"""

import os
import sys
import warnings
import argparse
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing_fixed import DataPreprocessorFixed
from train_with_ai_augmented import EnhancedAITrainingPipeline
from chatbot_interface import haChatbotInterface
from config import (
    TRAINING_DATA_PATH, 
    TRAINING_AI_AUGMENTED_DATA_PATH,
    MODELS_DIR,
    REPORTS_DIR
)

def create_directories():
    """Create necessary directories"""
    directories = [MODELS_DIR, REPORTS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def check_model_files():
    """Check if enhanced model files exist"""
    model_file = os.path.join(MODELS_DIR, "enhanced_ai_augmented_model.joblib")
    encoder_file = os.path.join(MODELS_DIR, "enhanced_ai_augmented_label_encoder.joblib")
    
    if os.path.exists(model_file) and os.path.exists(encoder_file):
        print("✅ Enhanced model files found")
        return True
    else:
        print("⚠️  Enhanced model files not found")
        return False

def train_enhanced_model():
    """Train the enhanced model if not already trained"""
    print("\n🚀 TRAINING ENHANCED MODEL")
    print("=" * 50)
    
    try:
        # Initialize and run enhanced training pipeline
        pipeline = EnhancedAITrainingPipeline()
        pipeline.load_and_preprocess()
        pipeline.train_enhanced_model()
        results = pipeline.evaluate_enhanced_model()
        pipeline.save_enhanced_model()
        pipeline.generate_enhanced_report(results)
        
        print("✅ Enhanced model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during enhanced model training: {e}")
        return False

def load_enhanced_model():
    """Load the enhanced trained model"""
    print("\n📂 LOADING ENHANCED MODEL")
    print("=" * 30)
    
    try:
        import joblib
        
        # Load enhanced model and encoder
        model_path = os.path.join(MODELS_DIR, "enhanced_ai_augmented_model.joblib")
        encoder_path = os.path.join(MODELS_DIR, "enhanced_ai_augmented_label_encoder.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            print("❌ Enhanced model files not found. Please train the model first.")
            return None, None
        
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        
        print("✅ Enhanced model loaded successfully!")
        return model, label_encoder
        
    except Exception as e:
        print(f"❌ Error loading enhanced model: {e}")
        return None, None

def create_model_trainer(model, label_encoder):
    """Create a model trainer object with the loaded model"""
    class ModelTrainer:
        def __init__(self, model, label_encoder):
            self.clf = model
            self.label_encoder = label_encoder
    
    return ModelTrainer(model, label_encoder)

def main():
    """Main application function - ENHANCED VERSION"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Healthcare Chatbot - Disease Prediction System')
    args = parser.parse_args()
    
    print("🏥 ENHANCED SYMPTOM-BASED DISEASE PREDICTION CHATBOT")
    print("=" * 60)
    print("🤖 AI-Powered • Machine Learning • Medical Validation • USER MODE")
    print("=" * 60)
    
    try:
        # Create necessary directories
        create_directories()
        
        # Check if enhanced model exists
        if not check_model_files():
            print("\n🔄 Enhanced model not found. Starting training...")
            if not train_enhanced_model():
                print("❌ Failed to train enhanced model. Exiting.")
                return
        else:
            print("✅ Using existing enhanced model")
        
        # Load enhanced model
        model, label_encoder = load_enhanced_model()
        if model is None:
            print("❌ Failed to load enhanced model. Exiting.")
            return
        
        # Initialize data preprocessor
        print("\n📊 Initializing data preprocessor...")
        data_preprocessor = DataPreprocessorFixed()
        data_preprocessor.initialize_all(use_augmented=False, use_ai_augmented=True)
        
        # Create model trainer
        model_trainer = create_model_trainer(model, label_encoder)
        
        # Initialize enhanced chatbot interface
        print("\n🤖 Initializing enhanced chatbot interface...")
        chatbot = ChatbotInterface(data_preprocessor, model_trainer)
        
        # Start the enhanced chatbot
        print("\n🎯 Starting Enhanced Healthcare Chatbot...")
        print("=" * 60)
        chatbot.start_chatbot()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thank you for using the Enhanced Healthcare Chatbot!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()