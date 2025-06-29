# Symptom-Based Disease Prediction Chatbot

**Python 3.10+ required**

## 🏥 Overview 

A robust, AI-powered health diagnosis chatbot that leverages machine learning to interpret symptoms and predict potential medical conditions. Designed for safety, accessibility, and collaboration, this chatbot offers instant, reliable health insights, guiding users towards informed medical decisions.

**🆕 Phase 4 Complete**: Enhanced with code quality improvements, user-friendly symptom formatting, centralized configuration, and optimized maintainability while preserving all core functionality.

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
- [Code Quality](#code-quality)
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

**🆕 Phase 3 Enhancements:**

10. **📊 Data-Driven Prediction Explanations**: The chatbot now explains its predictions by showing which symptoms contributed most to the diagnosis, using feature importance analysis.

11. **🎯 Top-3 Display for Low Confidence**: When confidence is low, the chatbot shows the top 3 possible diseases with confidence scores and encourages users to add more symptoms.

12. **🔍 Intelligent Follow-up Questions**: Data-driven follow-up questions based on missing critical symptoms, not hardcoded rules.

13. **🏥 Doctor Recommendations**: Integration with medical API to recommend relevant specialists based on predicted disease.

14. **🛡️ Enhanced Medical Safety**: Comprehensive medical validation, severity scoring, and safety warnings for all predictions.

**🆕 Latest Features:**

15. **🚨 Emergency Hospital Recommendations**: For emergency conditions (like heart attacks), the chatbot automatically recommends nearby hospitals with contact information and URLs.

16. **🏥 Hospital Search & Booking**: Users can search for hospitals by location and book appointments directly through the chatbot.

17. **🌐 Hospital URL Integration**: Emergency hospital recommendations include direct URLs to hospital profiles for easy access.

18. **📞 Location-Based Emergency Numbers**: Automatic emergency number detection based on user location (supports 200+ countries and cities).

19. **💊 Disease-Specific Emergency Advice**: Tailored emergency advice for specific conditions (heart attack, drug reactions, etc.).

20. **📋 Appointment Booking System**: Complete hospital appointment booking with confirmation and reference numbers.

**🆕 Phase 4 Code Quality Improvements:**

21. **🎨 User-Friendly Symptom Display**: Symptoms now display in clean, readable format (e.g., "Acute Liver Failure" instead of "acute_liver_failure").

22. **⚙️ Centralized Configuration**: All magic numbers and thresholds moved to configurable constants for easy maintenance.

23. **🧹 Code Deduplication**: Eliminated duplicate rating functions and input validation patterns.

24. **🔧 Utility Methods**: Created reusable utility methods for common operations.

25. **📊 Optimized Performance**: Reduced code complexity and improved maintainability.

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

### Main Menu Options
The chatbot now offers a comprehensive menu system:

1. **Diagnosis (disease prediction)** - Core symptom analysis and disease prediction
2. **Find a doctor** - Specialist recommendations based on diagnosis
3. **Find a hospital** - Hospital search by location with ratings and contact info
4. **Book hospital appointment** - Direct appointment booking system
5. **Exit** - End the session

### Example Session
```
🏥 ENHANCED SYMPTOM-BASED DISEASE PREDICTION CHATBOT
============================================================
🤖 AI-Powered • Machine Learning • Medical Validation • USER MODE
============================================================

What would you like to do?
1) Diagnosis (disease prediction)
2) Find a doctor
3) Find a hospital
4) Book hospital appointment
5) I am done / Exit

Enter 1, 2, 3, 4, or 5: 1

Enter symptoms: i have chest pain, vomiting, breathlessness
You entered 'chest pain'. This could refer to:
  1) Chest Pain
Select all that apply (comma-separated numbers, e.g. 1): 1

Here are the symptoms I have: Chest Pain, Vomiting, Breathlessness
Would you like to add, remove, or edit any symptoms? (yes/y or no/n): n

🏥 PREDICTION RESULTS
==================================================
🎯 Primary Diagnosis: Heart attack (100.0% confidence)

📊 Prediction Explanation:
This prediction is based on your reported symptoms:
• Chest Pain (high importance)
• Vomiting (high importance)
• Breathlessness (high importance)

🚨 EMERGENCY ALERT: Heart attack detected!
⚠️  This is a medical emergency requiring immediate hospital care.

🏥 Emergency Hospitals in cairo:
1. Majdi Yakoub
   📍 Location: Cairo, Egypt
   📞 Emergency Phone: +20112257967
   ⭐ Rating: 4/5
   🏗️  Established: 30-07-2005
   🌐 Profile: http://localhost:5173/hospital/profile?name=Majdi-Yakoub
---------------------------

🚨 IMMEDIATE ACTION REQUIRED:
📞 Call emergency services: 122
1. Go to the nearest hospital emergency department
2. Do not delay seeking medical attention
3. Bring someone with you if possible

💔 HEART ATTACK SPECIFIC ADVICE:
- Call emergency services immediately: 122
- Sit down and rest, avoid any physical exertion
- Take aspirin if available (unless allergic)
- Loosen tight clothing
- Stay calm and wait for emergency responders
```

### Hospital Booking Example
```
🏥 HOSPITAL BOOKING MENU
==================================================
🏥 Found 2 hospitals in cairo:

1. Majdi Yakoub
   📍 Location: Cairo, Egypt
   ⭐ Rating: 4/5
   📞 Phone: +20112257967
   🏗️  Established: 30-07-2005

2. Cairo Medical Center
   📍 Location: Cairo, Egypt
   ⭐ Rating: N/A
   📞 Phone: +20212345678
   🏗️  Established: 20-05-1990

Select hospital (1-2) or 'back' to return: 1

📋 BOOKING DETAILS
Patient Name: John Doe
Patient Phone: +20123456789
Appointment Date: 2024-01-15
Appointment Time: 10:00 AM

✅ BOOKING CONFIRMED!
📅 Your appointment has been successfully booked.
📞 You will receive a confirmation call shortly.
🏥 Please arrive 15 minutes before your appointment time.
📋 Don't forget to bring your ID and insurance card.
🔢 Booking Reference: ABC12345
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
- 📊 **Prediction Explanations**: Data-driven explanations for predictions
- 🏥 **Doctor Recommendations**: Specialist recommendations based on diagnosis
- 🚨 **Emergency Handling**: Emergency hospital recommendations with URLs
- 📋 **Hospital Booking**: Complete appointment booking workflow

### Demo Output Example
```
🏥 HEALTHCARE CHATBOT DEMO
============================================================
🔍 DEMO: Symptom Analysis
==================================================

📋 Test Case 1: 'headache fever'
✅ Extracted symptoms: Headache, Mild Fever
🏥 Predicted disease: Migraine
📊 Confidence: 85.23%

📊 Prediction Explanation:
This prediction is based on your reported symptoms:
• Headache (high importance)
• Mild Fever (medium importance)

🔍 DEMO: Medical Validation
==================================================

📋 Mild symptoms: Headache, Mild Fever
📊 Severity score: 8
📈 Severity level: Medium
🏥 Predicted: Migraine (85.23%)
✅ Medical validation: Passed

🚨 DEMO: Emergency Handling
==================================================

📋 Emergency symptoms: Chest Pain, Vomiting, Breathlessness
🏥 Predicted: Heart attack (100.0%)
🚨 Emergency alert triggered
🏥 Emergency hospitals recommended with URLs
📞 Location-based emergency number: 122
```

## 📊 Dataset

The project utilizes a comprehensive dataset containing symptom-disease mappings for disease prediction:

- **Training_safe_augmented.csv**: **Primary dataset** - Safe, rule-based augmented data (1,695 samples) with strict medical plausibility
- **Training.csv**: Original symptom-disease training data (303 samples)
- **Symptom_severity.csv**: Symptom severity scores for medical validation
- **symptom_Description.csv**: Detailed symptom descriptions
- **symptom_precaution.csv**: Disease precautions

**🆕 Dataset Improvements:**
- **Safe Augmentation**: Rule-based data augmentation maintaining medical plausibility
- **Medical Validation**: All symptom-disease associations validated for medical accuracy
- **Balanced Coverage**: 828 samples used in training after deduplication and balancing
- **Emergency Mapping**: Diseases mapped to medical specialties including "Emergency" for critical conditions

## 🧠 Model Architecture

The disease detection model is built using advanced machine learning techniques:

- **Algorithm**: Random Forest Classifier with optimized hyperparameters
- **Training Data**: Safe augmented dataset with medical validation
- **Validation**: Cross-validation and hyperparameter tuning
- **Calibration**: Isotonic calibration for better probability estimates
- **Features**: 132 symptoms with severity scoring

### Model Performance
- **Accuracy**: High accuracy with cross-validation
- **Confidence**: Calibrated probability estimates
- **Robustness**: Tested against noisy data and edge cases
- **Medical Safety**: All predictions validated for medical plausibility

## 🛡️ Medical Safety

- **Rule-based logic** prevents suggesting severe diseases for mild/common symptoms with low confidence
- **Medical validation** ensures predictions align with medical knowledge
- **Severity scoring** helps prioritize recommendations
- **Safe data augmentation** maintains medical plausibility
- **Comprehensive disclaimers** and safety warnings
- **Doctor recommendations** based on medical specialties
- **Low confidence warnings** when predictions are uncertain
- **Emergency detection** automatically triggers hospital recommendations for critical conditions
- **Location-based emergency numbers** provide appropriate emergency contacts
- **Disease-specific emergency advice** offers tailored guidance for different conditions

## 🎨 Code Quality

**🆕 Phase 4 Improvements:**

### **Code Deduplication**
- ✅ Eliminated duplicate `get_rating_value` functions for doctors and hospitals
- ✅ Created `_extract_rating_value()` utility method
- ✅ Reduced code by ~20 lines while improving maintainability

### **Input Validation Optimization**
- ✅ Created `_get_numeric_input()` utility method for consistent validation
- ✅ Replaced multiple `while True` loops with reusable utility
- ✅ Reduced repetitive code in hospital booking menu by ~30 lines

### **Centralized Configuration**
- ✅ Moved 15+ hardcoded values to `config.py`
- ✅ Added configuration constants for UI limits, time slots, confidence thresholds
- ✅ Easy to modify limits and thresholds without code changes

### **User-Friendly Symptom Display**
- ✅ Created `_format_symptom_name()` and `_format_symptom_list()` methods
- ✅ Symptoms now display as "Acute Liver Failure" instead of "acute_liver_failure"
- ✅ Consistent formatting throughout the application

### **Performance Improvements**
- ✅ Reduced code complexity and improved maintainability
- ✅ Better separation of concerns
- ✅ More professional and readable codebase

## 🧪 Testing & QA

- **Comprehensive test cases** for typos, ambiguous symptoms, generic terms, and edge-case scenarios
- **Realistic evaluation** with noisy data testing
- **Model diagnostics** for performance analysis
- **Medical validation testing** for prediction accuracy
- **Emergency scenario testing** for critical condition handling
- **Hospital booking testing** for appointment system validation
- **Code quality testing** for maintainability and performance
- **Users are encouraged to test and verify chatbot behavior in realistic and edge-case scenarios**

## 📁 Project Structure

```
AI/
├── main.py                          # Main application entry point
├── chatbot_interface.py             # Chatbot user interface (Phase 4 enhanced)
├── data_preprocessing_fixed.py      # Data preprocessing utilities
├── train_with_ai_augmented.py      # Training pipeline
├── config.py                       # Configuration settings (Phase 4 enhanced)
├── requirements.txt                # Python dependencies
├── emergency_numbers.json          # Emergency contact numbers (200+ countries)
├── Data/                           # Dataset files
│   ├── Training.csv               # Original training data
│   ├── Training_safe_augmented.csv # Primary dataset (safe augmented)
│   ├── Symptom_severity.csv       # Symptom severity mappings
│   ├── symptom_Description.csv    # Symptom descriptions
│   └── symptom_precaution.csv     # Disease precautions
├── models/                         # Trained model files
│   ├── enhanced_ai_augmented_model.joblib
│   └── enhanced_ai_augmented_label_encoder.joblib
├── bookings/                       # Generated booking files
│   └── booking_*.json             # Appointment booking records
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

7. **Hospital API Issues**
   - Check internet connection for hospital/doctor API access
   - Verify API endpoints are accessible

8. **Emergency Number Issues**
   - Ensure `emergency_numbers.json` file is present
   - Check file permissions and encoding

9. **Configuration Issues**
   - Verify `config.py` file is present and properly formatted
   - Check that all required constants are defined

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
- Maintain data-driven approach (avoid hardcoding)
- Test emergency scenarios thoroughly
- Validate hospital booking functionality
- Use configuration constants instead of magic numbers
- Follow the established utility method patterns

## 📈 Future Work

1. **Enhanced Accuracy**: Integrate more comprehensive symptom-disease mappings
2. **User History**: Track user history for personalized recommendations
3. **Web/Mobile App**: Deploy as a web or mobile application
4. **Advanced Hospital Integration**: Real-time hospital availability and scheduling
5. **Multi-language**: Expand language support beyond English and Arabic
6. **Advanced NLP**: Implement more sophisticated natural language processing
7. **Real-time Learning**: Incorporate user feedback for continuous improvement
8. **Telemedicine Integration**: Direct video consultation booking
9. **Prescription Management**: Medication tracking and reminders
10. **Health Records**: Secure user health history management
11. **Advanced Configuration**: Web-based configuration interface
12. **Performance Monitoring**: Real-time performance metrics and alerts

## 🙏 Credits

- **Numpy and pandas** for mathematical operations
- **csv module** for reading dataset files
- **Regular expression** for pattern matching
- **sklearn** for preprocessing, building models, and evaluation
- **Seaborn and Matplotlib** for visualization
- **Spacy** for natural language processing
- **Pyttsx3** for text-to-speech functionality
- **Requests** for API integration
- **Medical APIs** for doctor and hospital data

## 📄 License
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MadEase-Graduation-Project/AI)
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Medical Disclaimer**: This chatbot is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers for medical concerns. In emergency situations, call emergency services immediately.
