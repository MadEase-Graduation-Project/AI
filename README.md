# Symptom-Based Disease Prediction Chatbot

**Python 3.10+ required**

## 🏥 Overview 

A robust, AI-powered health diagnosis chatbot that leverages machine learning to interpret symptoms and predict potential medical conditions. Designed for safety, accessibility, and collaboration, this chatbot offers instant, reliable health insights, guiding users towards informed medical decisions.

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Medical Safety](#medical-safety)
- [Testing & QA](#testing--qa)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Credits](#credits)
- [Contact](#contact)

## ✨ Features

1. **🤖 AI-Powered Symptom Analysis**: Users can input symptoms in free text. The chatbot uses fuzzy/typo matching to interpret input, auto-corrects clear typos, and suggests corrections for ambiguous terms.

2. **🔍 Generic Term Handling**: If a user enters a generic term (e.g., "pain", "fever"), the chatbot prompts for clarification and lets the user select specific symptoms.

3. **✏️ Symptom Review/Edit**: Before diagnosis, users can review, add, or remove symptoms to ensure accuracy.

4. **🛡️ Rule-Based Post-Processing**: If all symptoms are mild/common and the top prediction is severe with low confidence, the bot suggests a mild disease instead.

5. **💡 Smart Recommendations**: The chatbot provides recommendations based on the identified diseases, including precautions and possible treatments.

6. **🎯 User-Friendly Interface**: Designed for clarity, error handling, and easy interaction.

7. **🔊 Text-to-Speech**: Optional voice output for accessibility.

8. **🌍 Multi-language Support**: Basic support for Arabic symptom synonyms.

9. **🛡️ Robust Input Handling**: All yes/no prompts accept both 'yes/y' and 'no/n'. Typo correction and generic term handling are applied in all chatbot flows and modes for a consistent, user-friendly experience.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd AI

# Activate virtual environment (if exists)
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate

# Test the setup
python3 test_setup.py

# Run the chatbot
python3 main.py
```

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd AI
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python3 -m venv chatbot_env
   source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Spacy Language Model**
   ```bash
   python3 -m spacy download en_core_web_sm
   ```

5. **Verify Installation**
   ```bash
   python3 test_setup.py
   ```

## 🎮 Usage

### Running the Chatbot
```bash
python3 main.py
```

### Example Session
```
🏥 ENHANCED SYMPTOM-BASED DISEASE PREDICTION CHATBOT
============================================================
🤖 AI-Powered • Machine Learning • Medical Validation • USER MODE
============================================================

Enter symptoms: i have pain, fever
You entered 'pain'. This could refer to multiple symptoms:
  1) joint_pain
  2) chest_pain
  3) muscle_pain
Select all that apply (comma-separated numbers, e.g. 1,2): 1

You entered 'fever'. This could refer to:
  1) high_fever
  2) mild_fever
Select all that apply (comma-separated numbers, e.g. 1,2): 2

Here are the symptoms I have: joint_pain, mild_fever
Would you like to add, remove, or edit any symptoms? (yes/y or no/n): n

Diagnosis: [disease prediction with confidence]
Precautions: [recommendations]
```

### Manual Training (Optional)
```bash
python3 train_with_ai_augmented.py
```

## 🧪 Testing

### Test Setup
Run the comprehensive test suite to verify everything is working:

```bash
python3 test_setup.py
```

This will test:
- ✅ All dependencies are installed correctly
- ✅ Project imports work properly
- ✅ Data files are accessible
- ✅ Model files are available
- ✅ Data loading functionality
- ✅ Model loading functionality

### Expected Output
```
🧪 HEALTHCARE CHATBOT SETUP TEST
==================================================
📋 Dependencies
✅ PASS Dependencies
✅ PASS Project Imports
✅ PASS Data Files
✅ PASS Model Files
✅ PASS Data Loading
✅ PASS Model Loading

Overall: 6/6 tests passed

🎉 All tests passed! The chatbot is ready to use.
```

## 🎬 Demo

### Interactive Demo
Run the demo script to see the chatbot in action:

```bash
python3 demo.py
```

This demonstrates:
- 🔍 **Symptom Analysis**: How the chatbot extracts and processes symptoms
- 🛡️ **Medical Validation**: Severity scoring and medical rule validation
- 🔍 **Feature Importance**: Most important symptoms for disease prediction
- 🎯 **Symptom Matching**: Fuzzy matching and typo correction

### Demo Output Example
```
🏥 HEALTHCARE CHATBOT DEMO
============================================================
🔍 DEMO: Symptom Analysis
==================================================

📋 Test Case 1: 'headache fever'
✅ Extracted symptoms: headache, mild_fever
🏥 Predicted disease: Migraine
📊 Confidence: 85.23%

🔍 DEMO: Medical Validation
==================================================

📋 Mild symptoms: headache, mild_fever
📊 Severity score: 8
📈 Severity level: Medium
🏥 Predicted: Migraine (85.23%)
✅ Medical validation: Passed
```

## 📊 Dataset

The project utilizes a comprehensive dataset containing symptom-disease mappings for disease prediction:

- **Training.csv**: Original symptom-disease training data
- **Training_ai_augmented.csv**: AI-augmented training data for improved accuracy
- **Symptom_severity.csv**: Symptom severity scores for medical validation
- **symptom_Description.csv**: Detailed symptom descriptions
- **symptom_precaution.csv**: Disease precautions

## 🧠 Model Architecture

The disease detection model is built using advanced machine learning techniques:

- **Algorithm**: Random Forest Classifier with optimized hyperparameters
- **Training Data**: Both original and AI-augmented datasets
- **Validation**: Cross-validation and hyperparameter tuning
- **Calibration**: Isotonic calibration for better probability estimates
- **Features**: 132 symptoms with severity scoring

### Model Performance
- **Accuracy**: High accuracy with cross-validation
- **Confidence**: Calibrated probability estimates
- **Robustness**: Tested against noisy data and edge cases

## 🛡️ Medical Safety

- **Rule-based logic** prevents suggesting severe diseases for mild/common symptoms with low confidence
- **Medical validation** ensures predictions align with medical knowledge
- **Severity scoring** helps prioritize recommendations
- **All data augmentation and model logic maintain medical plausibility**

## 🧪 Testing & QA

- **Comprehensive test cases** for typos, ambiguous symptoms, generic terms, and edge-case scenarios
- **Realistic evaluation** with noisy data testing
- **Model diagnostics** for performance analysis
- **Users are encouraged to test and verify chatbot behavior in realistic and edge-case scenarios**

## 📁 Project Structure

```
AI/
├── main.py                          # Main application entry point
├── chatbot_interface.py             # Chatbot user interface
├── data_preprocessing_fixed.py      # Data preprocessing utilities
├── train_with_ai_augmented.py      # AI-augmented training pipeline
├── ai_data_augmentation.py         # Data augmentation utilities
├── diagnose_model.py               # Model diagnostics
├── realistic_evaluation.py         # Model evaluation
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── test_setup.py                   # Setup verification script
├── demo.py                         # Demo script
├── Data/                           # Dataset files
│   ├── Training.csv               # Original training data
│   ├── Training_ai_augmented.csv  # AI-augmented training data
│   ├── Symptom_severity.csv       # Symptom severity mappings
│   ├── symptom_Description.csv    # Symptom descriptions
│   └── symptom_precaution.csv     # Disease precautions
├── models/                         # Trained model files
│   ├── enhanced_ai_augmented_model.joblib
│   └── enhanced_ai_augmented_label_encoder.joblib
├── reports/                        # Generated reports
└── plots/                          # Visualization outputs
```

## 🔧 Troubleshooting

### Common Issues

1. **Spacy Language Model Not Found**
   ```bash
   python3 -m spacy download en_core_web_sm
   ```

2. **Text-to-Speech Issues**
   - **macOS**: May require additional system permissions
   - **Linux**: May need to install `espeak` or `festival`
   - **Windows**: Should work out of the box

3. **Model Training Issues**
   - Ensure all data files are present in the `Data/` directory
   - Check that you have sufficient disk space for model files

4. **Import Errors**
   - Verify all dependencies are installed: `pip list`
   - Check Python version: `python3 --version`

5. **Virtual Environment Issues**
   - Make sure the virtual environment is activated: `source chatbot_env/bin/activate`
   - Reinstall dependencies if needed: `pip install -r requirements.txt`

6. **Test Failures**
   - Run `python3 test_setup.py` to identify specific issues
   - Check the error messages for guidance

## 🤝 Contributing

1. Fork the repository or create a new branch
2. Make your changes and add clear commit messages
3. Test your changes locally using `python3 test_setup.py`
4. Submit a pull request or merge request for review

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive error handling
- Include docstrings for new functions
- Test with various symptom combinations
- Ensure medical accuracy of changes
- Run the test suite before submitting changes

## 📈 Future Work

1. **Enhanced Accuracy**: Integrate more comprehensive symptom-disease mappings
2. **User History**: Track user history for personalized recommendations
3. **Web/Mobile App**: Deploy as a web or mobile application
4. **Hospital Integration**: Integrate hospital/doctor APIs for real-world help
5. **Multi-language**: Expand language support beyond English and Arabic
6. **Advanced NLP**: Implement more sophisticated natural language processing

## 🙏 Credits

- **Numpy and pandas** for mathematical operations
- **csv module** for reading dataset files
- **Regular expression** for pattern matching
- **sklearn** for preprocessing, building models, and evaluation
- **Seaborn and Matplotlib** for visualization
- **Spacy** for natural language processing
- **Pyttsx3** for text-to-speech functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Medical Disclaimer**: This chatbot is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers for medical concerns.
