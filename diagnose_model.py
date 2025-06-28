#!/usr/bin/env python3
"""
Model Diagnostic Script
Analyzes the model's performance and identifies issues with low confidence
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing_fixed import DataPreprocessorFixed

def analyze_model_performance():
    """Analyze the model's performance and identify issues"""
    print("🔍 MODEL DIAGNOSTIC ANALYSIS")
    print("=" * 50)
    
    # Load data and model
    preprocessor = DataPreprocessorFixed()
    preprocessor.initialize_all(use_augmented=False, use_ai_augmented=True)
    
    model = joblib.load("models/enhanced_ai_augmented_model.joblib")
    le = joblib.load("models/enhanced_ai_augmented_label_encoder.joblib")
    
    # Test on test set
    X_test = preprocessor.x_test
    y_test = preprocessor.y_test
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate confidence scores
    max_probs = np.max(y_pred_proba, axis=1)
    confidence_stats = {
        'mean': np.mean(max_probs),
        'std': np.std(max_probs),
        'min': np.min(max_probs),
        'max': np.max(max_probs),
        'low_confidence_count': np.sum(max_probs < 0.5),
        'very_low_confidence_count': np.sum(max_probs < 0.3)
    }
    
    print(f"📊 CONFIDENCE ANALYSIS:")
    print(f"Mean confidence: {confidence_stats['mean']:.3f}")
    print(f"Std confidence: {confidence_stats['std']:.3f}")
    print(f"Min confidence: {confidence_stats['min']:.3f}")
    print(f"Max confidence: {confidence_stats['max']:.3f}")
    print(f"Low confidence predictions (<50%): {confidence_stats['low_confidence_count']}/{len(y_test)} ({confidence_stats['low_confidence_count']/len(y_test)*100:.1f}%)")
    print(f"Very low confidence predictions (<30%): {confidence_stats['very_low_confidence_count']}/{len(y_test)} ({confidence_stats['very_low_confidence_count']/len(y_test)*100:.1f}%)")
    
    # Analyze feature importance
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    # Get feature importance from the underlying estimator
    if hasattr(model, 'base_estimator'):
        # For CalibratedClassifierCV
        base_model = model.base_estimator
    elif hasattr(model, 'estimator'):
        # For other wrapped models
        base_model = model.estimator
    else:
        base_model = model
    
    feature_importance = pd.DataFrame({
        'symptom': preprocessor.cols,
        'importance': base_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important symptoms:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['symptom']:<30} {row['importance']:.4f}")
    
    # Analyze low confidence cases
    low_conf_indices = np.where(max_probs < 0.5)[0]
    if len(low_conf_indices) > 0:
        print(f"\n⚠️  LOW CONFIDENCE CASES ANALYSIS:")
        print(f"Found {len(low_conf_indices)} cases with low confidence")
        
        # Show a few examples
        for i, idx in enumerate(low_conf_indices[:5]):
            true_disease = le.inverse_transform([y_test[idx]])[0]
            pred_disease = le.inverse_transform([y_pred[idx]])[0]
            confidence = max_probs[idx]
            
            # Get symptoms for this case
            symptoms = []
            for j, col in enumerate(preprocessor.cols):
                if X_test.iloc[idx, j] == 1:
                    symptoms.append(col)
            
            print(f"\nCase {i+1}:")
            print(f"  True disease: {true_disease}")
            print(f"  Predicted: {pred_disease}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Symptoms: {', '.join(symptoms[:5])}...")
    
    # Test specific symptom combinations
    print(f"\n🧪 TESTING SPECIFIC SYMPTOM COMBINATIONS:")
    test_cases = [
        (['headache', 'mild_fever', 'nausea'], "Common migraine symptoms"),
        (['itching', 'skin_rash'], "Common skin infection symptoms"),
        (['cough', 'fever'], "Common respiratory symptoms"),
        (['abdominal_pain', 'nausea', 'vomiting'], "Common gastrointestinal symptoms")
    ]
    
    for symptoms, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"Symptoms: {', '.join(symptoms)}")
        
        # Create input vector
        input_vector = [1 if symptom in symptoms else 0 for symptom in preprocessor.cols]
        
        # Get prediction
        proba = model.predict_proba([input_vector])[0]
        top_3_idx = np.argsort(proba)[::-1][:3]
        top_3_diseases = le.inverse_transform(top_3_idx)
        top_3_confidences = proba[top_3_idx]
        
        print(f"Top 3 predictions:")
        for i, (disease, conf) in enumerate(zip(top_3_diseases, top_3_confidences)):
            print(f"  {i+1}. {disease}: {conf:.3f}")
    
    return confidence_stats, preprocessor

def suggest_improvements(confidence_stats, preprocessor):
    """Suggest improvements based on analysis"""
    print(f"\n💡 SUGGESTIONS FOR IMPROVEMENT:")
    
    if confidence_stats['mean'] < 0.7:
        print("1. ⚠️  Model confidence is too low. Consider:")
        print("   - Retraining with more diverse data")
        print("   - Using ensemble methods")
        print("   - Adjusting model hyperparameters")
    
    if confidence_stats['low_confidence_count'] > len(preprocessor.y_test) * 0.3:
        print("2. ⚠️  Too many low confidence predictions. Consider:")
        print("   - Improving feature engineering")
        print("   - Adding more training data")
        print("   - Using different algorithms")
    
    print("3. 🔧 Technical improvements:")
    print("   - Implement confidence thresholds")
    print("   - Add fallback to multiple predictions")
    print("   - Improve symptom matching logic")
    print("   - Add medical validation rules")

if __name__ == "__main__":
    confidence_stats, preprocessor = analyze_model_performance()
    suggest_improvements(confidence_stats, preprocessor) 