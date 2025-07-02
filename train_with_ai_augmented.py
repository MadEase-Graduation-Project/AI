#!/usr/bin/env python3
"""
Training Script with Safe Augmented Dataset - ENHANCED VERSION
Uses the safe medically validated augmented dataset for training with improved confidence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE
from data_preprocessing_fixed import DataPreprocessorFixed

class EnhancedAITrainingPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessorFixed()
        self.model = None
        self.calibrated_model = None
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess(self):
        """Load and preprocess the data"""
        print("🔄 Initializing data preprocessor...")
        self.preprocessor.initialize_all(use_augmented=False, use_ai_augmented=False, use_safe_augmented=True)
        
        # Get features and target, using severity scores for present symptoms
        X_raw = self.preprocessor.training.drop(['prognosis', 'Medical Specialties'], axis=1)
        y = self.preprocessor.training['prognosis']
        
        # Map binary symptom presence to severity score
        severity_dict = self.preprocessor.severityDictionary
        X = X_raw.copy()
        for col in X.columns:
            sev = severity_dict.get(col, 1)
            X[col] = X[col].apply(lambda v: sev if v == 1 else 0)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Features: {len(X.columns)} symptoms")
        print(f"Classes: {len(np.unique(y_encoded))} diseases")
        
    def train_enhanced_model(self):
        """Train the enhanced Random Forest model with better parameters"""
        print("\n🚀 Training Enhanced Random Forest model...")
        
        # Initialize model with optimized parameters for better confidence
        self.model = RandomForestClassifier(
            n_estimators=500,  # Increased from 200
            max_depth=20,      # Increased from 15
            min_samples_split=3,  # Reduced from 5
            min_samples_leaf=1,   # Reduced from 2
            max_features='sqrt',  # Better feature selection
            bootstrap=True,
            oob_score=True,       # Out-of-bag scoring
            class_weight='balanced',  # Handle class imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Calibrate the model for better probability estimates
        print("🔄 Calibrating model probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, 
            cv=5, 
            method='isotonic'
        )
        self.calibrated_model.fit(self.X_train, self.y_train)
        
        print("✅ Enhanced model training completed!")
        
    def evaluate_enhanced_model(self):
        """Comprehensive model evaluation with confidence analysis"""
        print("\n📊 ENHANCED MODEL EVALUATION")
        print("=" * 50)
        
        # Use calibrated model for better probability estimates
        model_to_evaluate = self.calibrated_model if self.calibrated_model else self.model
        
        # Basic predictions
        y_pred = model_to_evaluate.predict(self.X_test)
        y_pred_proba = model_to_evaluate.predict_proba(self.X_test if isinstance(self.X_test, pd.DataFrame) else pd.DataFrame(self.X_test, columns=self.X_train.columns))
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Cross-validation
        print("\n🔄 Cross-Validation Results:")
        cv_scores = cross_val_score(
            model_to_evaluate, self.X_train, self.y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='accuracy'
        )
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print("\n📋 Classification Report:")
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        # Enhanced confidence analysis
        self._analyze_confidence_distribution(y_pred_proba)
        
        # Feature importance
        self._analyze_feature_importance()
        
        # Confusion matrix
        self._plot_confusion_matrix(y_test_decoded, y_pred_decoded)
        
        return {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': model_to_evaluate
        }
    
    def _analyze_confidence_distribution(self, y_pred_proba):
        """Analyze confidence distribution"""
        print("\n🎯 CONFIDENCE DISTRIBUTION ANALYSIS")
        print("-" * 35)
        
        # Get max probability for each prediction
        max_probs = np.max(y_pred_proba, axis=1)
        
        print(f"Mean confidence: {np.mean(max_probs):.4f}")
        print(f"Median confidence: {np.median(max_probs):.4f}")
        print(f"Std confidence: {np.std(max_probs):.4f}")
        print(f"Min confidence: {np.min(max_probs):.4f}")
        print(f"Max confidence: {np.max(max_probs):.4f}")
        
        # Confidence thresholds
        high_conf = np.sum(max_probs >= 0.7)
        medium_conf = np.sum((max_probs >= 0.5) & (max_probs < 0.7))
        low_conf = np.sum(max_probs < 0.5)
        
        print(f"\nConfidence breakdown:")
        print(f"High confidence (≥70%): {high_conf}/{len(max_probs)} ({high_conf/len(max_probs)*100:.1f}%)")
        print(f"Medium confidence (50-70%): {medium_conf}/{len(max_probs)} ({medium_conf/len(max_probs)*100:.1f}%)")
        print(f"Low confidence (<50%): {low_conf}/{len(max_probs)} ({low_conf/len(max_probs)*100:.1f}%)")
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
        plt.axvline(0.7, color='green', linestyle='--', label='High confidence threshold')
        plt.axvline(0.5, color='orange', linestyle='--', label='Medium confidence threshold')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Enhanced Model Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('enhanced_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Enhanced confidence distribution saved as 'enhanced_confidence_distribution.png'")
    
    def _analyze_feature_importance(self):
        """Analyze and display feature importance"""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 30)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'symptom': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 15 Most Important Symptoms:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"{i+1:2d}. {row['symptom']:<30} {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['symptom'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Symptoms (Enhanced Model)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Enhanced feature importance plot saved as 'enhanced_feature_importance.png'")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        print("\n📈 CONFUSION MATRIX")
        print("-" * 20)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Enhanced Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Enhanced confusion matrix saved as 'enhanced_confusion_matrix.png'")
        
        # Print some statistics
        print(f"Total predictions: {len(y_pred)}")
        print(f"Correct predictions: {np.sum(y_true == y_pred)}")
        print(f"Incorrect predictions: {np.sum(y_true != y_pred)}")
    
    def save_enhanced_model(self):
        """Save the enhanced model and label encoder"""
        print("\n💾 Saving Enhanced Model...")
        
        # Save the calibrated model if available, otherwise save the base model
        model_to_save = self.calibrated_model if self.calibrated_model else self.model
        
        import joblib
        joblib.dump(model_to_save, "models/enhanced_ai_augmented_model.joblib")
        joblib.dump(self.label_encoder, "models/enhanced_ai_augmented_label_encoder.joblib")
        
        print("✅ Enhanced model saved as 'enhanced_ai_augmented_model.joblib'")
        print("✅ Enhanced label encoder saved as 'enhanced_ai_augmented_label_encoder.joblib'")
    
    def generate_enhanced_report(self, results):
        """Generate comprehensive report"""
        print("\n📋 GENERATING ENHANCED REPORT")
        print("-" * 30)
        
        report_content = f"""# Enhanced Safe Augmented Model Training Report

## Model Configuration
- **Algorithm**: Random Forest (Enhanced)
- **Estimators**: 500
- **Max Depth**: 20
- **Min Samples Split**: 3
- **Min Samples Leaf**: 1
- **Class Weight**: Balanced
- **Calibration**: Isotonic
- **Dataset**: Safe Medically Validated Augmented Data

## Performance Metrics
- **Test Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- **CV Accuracy**: {results['cv_scores'].mean():.4f} (+/- {results['cv_scores'].std() * 2:.4f})

## Confidence Analysis
- **Mean Confidence**: {np.mean(np.max(results['probabilities'], axis=1)):.4f}
- **High Confidence (≥70%)**: {np.sum(np.max(results['probabilities'], axis=1) >= 0.7)}/{len(results['probabilities'])} ({np.sum(np.max(results['probabilities'], axis=1) >= 0.7)/len(results['probabilities'])*100:.1f}%)
- **Medium Confidence (50-70%)**: {np.sum((np.max(results['probabilities'], axis=1) >= 0.5) & (np.max(results['probabilities'], axis=1) < 0.7))}/{len(results['probabilities'])} ({np.sum((np.max(results['probabilities'], axis=1) >= 0.5) & (np.max(results['probabilities'], axis=1) < 0.7))/len(results['probabilities'])*100:.1f}%)
- **Low Confidence (<50%)**: {np.sum(np.max(results['probabilities'], axis=1) < 0.5)}/{len(results['probabilities'])} ({np.sum(np.max(results['probabilities'], axis=1) < 0.5)/len(results['probabilities'])*100:.1f}%)

## Improvements
1. **Enhanced Parameters**: Increased estimators and depth for better learning
2. **Class Balancing**: Added class_weight to handle imbalanced data
3. **Probability Calibration**: Used isotonic calibration for better confidence estimates
4. **Out-of-bag Scoring**: Added OOB scoring for better validation
5. **Safe Medical Data**: Uses medically validated augmented dataset with strict rules

## Files Generated
- `enhanced_ai_augmented_model.joblib`: Enhanced trained model
- `enhanced_ai_augmented_label_encoder.joblib`: Label encoder
- `enhanced_confidence_distribution.png`: Confidence distribution plot
- `enhanced_feature_importance.png`: Feature importance plot
- `enhanced_confusion_matrix.png`: Confusion matrix

---
Generated with Enhanced Safe Augmented training pipeline
"""
        
        with open("reports/enhanced_ai_augmented_training_report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print("📄 Enhanced report saved as 'enhanced_ai_augmented_training_report.md'")

def main():
    """Main training function"""
    print("🚀 Starting Enhanced Safe Augmented Model Training...")
    
    # Initialize pipeline
    pipeline = EnhancedAITrainingPipeline()
    
    # Load and preprocess data
    pipeline.load_and_preprocess()
    
    # Train enhanced model
    pipeline.train_enhanced_model()
    
    # Evaluate model
    results = pipeline.evaluate_enhanced_model()
    
    # Save model
    pipeline.save_enhanced_model()
    
    # Generate report
    pipeline.generate_enhanced_report(results)
    
    print("\n🎉 Enhanced safe augmented training completed successfully!")
    print("📊 Check the reports/ directory for detailed analysis")

if __name__ == "__main__":
    main() 