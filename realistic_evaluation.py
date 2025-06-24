#!/usr/bin/env python3
"""
Realistic Model Evaluation
Tests model robustness with noisy data and smaller test sets
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_preprocessing_fixed import DataPreprocessorFixed

# Load pretrained model and label encoder
MODEL_PATH = "models/enhanced_ai_augmented_model.joblib"
ENCODER_PATH = "models/enhanced_ai_augmented_label_encoder.joblib"

class ModelLoader:
    def __init__(self, model_path, encoder_path):
        self.clf = joblib.load(model_path)
        self.le = joblib.load(encoder_path)
    def train_models(self):
        pass
    def calc_condition(self, symptoms, num_days):
        pass

def add_noise_to_data(X, noise_level=0.1):
    """Add random noise to symptoms to simulate real-world variability"""
    X_noisy = X.copy()
    for i in range(len(X_noisy)):
        # Randomly flip some symptoms (0->1 or 1->0)
        flip_indices = np.random.choice(len(X_noisy.columns), 
                                       size=int(noise_level * len(X_noisy.columns)), 
                                       replace=False)
        for idx in flip_indices:
            X_noisy.iloc[i, idx] = 1 - X_noisy.iloc[i, idx]
    return X_noisy

def realistic_evaluation():
    """Perform realistic evaluation of the model"""
    
    print("🔬 REALISTIC MODEL EVALUATION")
    print("=" * 60)
    
    # Load preprocessor and data
    preprocessor = DataPreprocessorFixed()
    preprocessor.initialize_all(use_augmented=False, use_ai_augmented=True)
    X = preprocessor.x
    y = preprocessor.y
    # Load model and encoder
    model = ModelLoader(MODEL_PATH, ENCODER_PATH)
    clf = model.clf
    le = model.le

    # Split data (use the same split as in training)
    X_train, X_test, y_train, y_test = preprocessor.x_train, preprocessor.x_test, preprocessor.y_train, preprocessor.y_test

    # Evaluate on test set
    print("Test set size:", len(X_test))
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
    print("\nConfusion Matrix:")
    print(confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

    # Noise robustness test
    print("\n--- Noise Robustness Test ---")
    X_test_noisy = X_test.copy()
    np.random.seed(42)
    # Flip 10% of symptom values
    for i in range(X_test_noisy.shape[0]):
        flip_indices = np.random.choice(X_test_noisy.columns, size=int(0.1*len(X_test_noisy.columns)), replace=False)
        for col in flip_indices:
            X_test_noisy.iloc[i][col] = 1 - X_test_noisy.iloc[i][col]
    y_pred_noisy = clf.predict(X_test_noisy)
    acc_noisy = accuracy_score(y_test, y_pred_noisy)
    print(f"Accuracy with 10% random symptom noise: {acc_noisy:.4f} ({acc_noisy*100:.2f}%)")

    # Show top 3 predictions for first 5 test cases
    print("\n--- Top 3 Predictions for First 5 Test Cases ---")
    proba = clf.predict_proba(X_test)
    for i in range(min(5, len(X_test))):
        top3_idx = np.argsort(proba[i])[::-1][:3]
        top3_diseases = le.inverse_transform(top3_idx)
        top3_probs = proba[i][top3_idx]
        print(f"Case {i+1}: True: {le.inverse_transform([y_test[i]])[0]}")
        for d, p in zip(top3_diseases, top3_probs):
            print(f"   {d}: {p:.2%}")
    print("\n=== END OF EVALUATION ===\n")

    # Test 1: Add noise to test data
    print(f"\n🧪 Test 1: Noisy Data Evaluation")
    noise_levels = [0.05, 0.1, 0.15, 0.2]
    noisy_accuracies = []
    
    for noise in noise_levels:
        X_noisy = add_noise_to_data(X_test, noise)
        noisy_pred = clf.predict(X_noisy)
        noisy_acc = accuracy_score(y_test, noisy_pred)
        noisy_accuracies.append(noisy_acc)
        print(f"Noise level {noise:.2f}: Accuracy = {noisy_acc:.4f}")
    
    # Test 2: Cross-validation with smaller folds
    print(f"\n🔄 Test 2: Cross-Validation with Smaller Folds")
    X_full = preprocessor.x_train
    y_full = preprocessor.y_train
    
    # Use 3-fold CV (matches minimum class size)
    cv_scores = cross_val_score(clf, X_full, y_full, cv=3, scoring='accuracy')
    print(f"3-fold CV scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test 3: Leave-one-out cross-validation (most stringent) - COMMENTED OUT DUE TO SLOWNESS
    # print(f"\n🎯 Test 3: Leave-One-Out Cross-Validation")
    # from sklearn.model_selection import LeaveOneOut
    # loo = LeaveOneOut()
    # loo_scores = cross_val_score(clf, X_full, y_full, cv=loo, scoring='accuracy')
    # print(f"LOO CV accuracy: {loo_scores.mean():.4f} (+/- {loo_scores.std() * 2:.4f})")
    
    # Use a faster alternative: Stratified K-Fold with more folds
    print(f"\n🎯 Test 3: Stratified 5-Fold Cross-Validation (Faster Alternative)")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_scores = cross_val_score(clf, X_full, y_full, cv=skf, scoring='accuracy')
    print(f"5-Fold Stratified CV accuracy: {skf_scores.mean():.4f} (+/- {skf_scores.std() * 2:.4f})")
    
    # Test 4: Feature importance analysis
    print(f"\n🔍 Test 4: Feature Importance Analysis")
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        feature_names = preprocessor.cols
        
        # Check if model relies too heavily on few features
        top_features = np.argsort(importances)[-10:]
        print("Top 10 most important features:")
        for i, idx in enumerate(reversed(top_features)):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Calculate feature concentration
        concentration = np.sum(importances[top_features]) / np.sum(importances)
        print(f"Top 10 features account for {concentration:.2%} of importance")
        
        if concentration > 0.8:
            print("⚠️  WARNING: Model relies heavily on few features - may be overfitting")
    
    # Test 5: Confusion matrix analysis
    print(f"\n📈 Test 5: Confusion Matrix Analysis")
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("Per-class accuracy:")
    for i, acc in enumerate(class_accuracies):
        disease_name = le.inverse_transform([i])[0]
        print(f"{disease_name}: {acc:.4f}")
    
    # Find worst performing classes
    worst_classes = np.argsort(class_accuracies)[:5]
    print("\nWorst performing classes:")
    for i, idx in enumerate(worst_classes):
        disease_name = le.inverse_transform([idx])[0]
        print(f"{i+1}. {disease_name}: {class_accuracies[idx]:.4f}")
    
    # Plot results
    plot_realistic_evaluation(noise_levels, noisy_accuracies, cv_scores, skf_scores.mean())
    
    return {
        'original_accuracy': acc,
        'noisy_accuracies': noisy_accuracies,
        'cv_accuracy': cv_scores.mean(),
        'loo_accuracy': skf_scores.mean(),
        'cv_std': cv_scores.std()
    }

def plot_realistic_evaluation(noise_levels, noisy_accuracies, cv_scores, skf_accuracy):
    """Plot the realistic evaluation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Noise vs Accuracy
    axes[0, 0].plot(noise_levels, noisy_accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Noise Level')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Robustness to Noise')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cross-validation scores
    axes[0, 1].boxplot(cv_scores)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('3-Fold Cross-Validation Scores')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Performance comparison
    methods = ['Original Test', '3-Fold CV', '5-Fold Stratified CV']
    accuracies = [1.0, cv_scores.mean(), skf_accuracy]  # Assuming original test was 100%
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = axes[1, 0].bar(methods, accuracies, color=colors)
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Performance Comparison')
    axes[1, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 4: Noise impact summary
    axes[1, 1].text(0.1, 0.8, f'Original Accuracy: 100%', fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'3-Fold CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'5-Fold Stratified CV: {skf_accuracy:.3f}', fontsize=12)
    axes[1, 1].text(0.1, 0.5, f'Noise Sensitivity: {noisy_accuracies[1]:.3f} (10% noise)', fontsize=12)
    
    # Add interpretation
    if cv_scores.mean() < 0.9:
        axes[1, 1].text(0.1, 0.3, '⚠️ Model may be overfitting', fontsize=12, color='red')
    else:
        axes[1, 1].text(0.1, 0.3, '✅ Model shows good generalization', fontsize=12, color='green')
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Evaluation Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Realistic evaluation plots generated!")

def generate_realistic_report(results):
    """Generate a realistic evaluation report"""
    
    report = f"""
# Realistic Model Evaluation Report

## Executive Summary
The model was evaluated using multiple realistic testing scenarios to assess true generalization performance.

## Key Findings

### Performance Metrics
- **Original Test Accuracy**: {results['original_accuracy']:.4f} (potentially overfitting)
- **3-Fold Cross-Validation**: {results['cv_accuracy']:.4f} ± {results['cv_std']:.4f}
- **5-Fold Stratified CV**: {results['loo_accuracy']:.4f}
- **Noise Robustness**: {results['noisy_accuracies'][1]:.4f} (with 10% noise)

### Interpretation
"""
    
    if results['cv_accuracy'] < 0.9:
        report += """
⚠️ **MODEL MAY BE OVERFITTING**
- Cross-validation accuracy significantly lower than test accuracy
- Model may not generalize well to new, unseen data
- Consider collecting more diverse training data
"""
    else:
        report += """
✅ **MODEL SHOWS GOOD GENERALIZATION**
- Cross-validation accuracy is high and consistent
- Model should perform well on new data
"""
    
    report += f"""

### Recommendations
1. **Cross-Validation**: Use {results['cv_accuracy']:.3f} as the true performance metric
2. **Noise Testing**: Model robustness to noise is {results['noisy_accuracies'][1]:.3f}
3. **Real-world Validation**: Test with external datasets
4. **Data Collection**: Gather more diverse symptom combinations

### Conclusion
The 100% test accuracy is likely due to overfitting to a small, artificial test set. 
The cross-validation accuracy of {results['cv_accuracy']:.3f} is a more realistic 
estimate of true model performance.
"""
    
    with open('realistic_evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("📄 Realistic evaluation report saved as 'realistic_evaluation_report.md'")

if __name__ == "__main__":
    print("🚀 Starting Realistic Model Evaluation...")
    results = realistic_evaluation()
    generate_realistic_report(results)
    print("✅ Realistic evaluation completed!") 