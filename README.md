# Symptom-Based-Disease-Prediction-Chatbot-Using-NLP

**Python 3.10+ required**

## Overview 

A robust, AI-powered health diagnosis chatbot that leverages machine learning to interpret symptoms and predict potential medical conditions. Designed for safety, accessibility, and collaboration, this chatbot offers instant, reliable health insights, guiding users towards informed medical decisions.

## Table of Contents

Description\
Features\
Usage\
Dataset\
Model Architecture\
Medical Safety\
Testing & QA\
Future Work\
Troubleshooting\
Credits\
Contact Info

## Description

This project implements a HealthCare Chatbot for disease detection based on symptoms. The chatbot utilizes advanced machine learning algorithms (Random Forest, SVC) and a robust user interface to analyze user-reported symptoms, identify potential diseases, and provide relevant recommendations. The system is designed for medical safety and future integration with apps/web.

## Features

1. *Symptom Analysis*: Users can input symptoms in free text. The chatbot uses fuzzy/typo matching to interpret input, auto-corrects clear typos, and suggests corrections for ambiguous terms.
2. *Generic Term Handling*: If a user enters a generic term (e.g., "pain", "fever"), the chatbot prompts for clarification and lets the user select specific symptoms.
3. *Symptom Review/Edit*: Before diagnosis, users can review, add, or remove symptoms to ensure accuracy.
4. *Rule-Based Post-Processing*: If all symptoms are mild/common and the top prediction is severe with low confidence, the bot suggests a mild disease instead.
5. *Recommendations*: The chatbot provides recommendations based on the identified diseases, including precautions and possible treatments.
6. *User-Friendly Interface*: Designed for clarity, error handling, and easy interaction.

## Usage

1. Launch the chatbot application.
2. Enter the symptoms you are experiencing (free text, e.g., "i have pain, fever").
3. The chatbot will auto-correct typos, prompt for clarification on generic/ambiguous terms, and let you review your symptoms.
4. Confirm or edit your symptoms as needed.
5. Receive disease predictions and recommendations from the chatbot.

## Dataset

The project utilizes a dataset containing symptom-disease mappings for disease prediction. The dataset is included in this repository and can be obtained from sources like Kaggle. See `config.py` for data file paths.

## Model Architecture

The disease detection model is built using machine learning algorithms (Random Forest, SVC) and uses severity scores as input features. The model is trained on both original and AI-augmented data, with cross-validation and hyperparameter tuning for reliability.

## Medical Safety
- Rule-based logic prevents suggesting severe diseases for mild/common symptoms with low confidence.
- All data augmentation and model logic maintain medical plausibility.

## Testing & QA
- Comprehensive test cases for typos, ambiguous symptoms, generic terms, and edge-case scenarios.
- Users are encouraged to test and verify chatbot behavior in realistic and edge-case scenarios.

## Future Work
1. Enhance accuracy with more comprehensive symptom-disease mappings.
2. Integrate user history tracking and personalized recommendations.
3. Deploy as a web or mobile application.
4. Integrate hospital/doctor APIs for real-world help.

## Troubleshooting
If you encounter any issues:
1. Check that all dependencies are installed.
2. Ensure the dataset is accessible and formatted correctly.
3. Verify that the chatbot application is using the correct input/output channels.

## Credits
- Numpy and pandas for mathematical operations
- csv module for reading dataset files
- Regular expression for pattern matching
- sklearn for preprocessing, building models, and evaluation
- Seaborn and Matplotlib for visualization

## Contact Information
For questions, feedback, or contributions, please contact Saumya at saumya.rastogi.03@gmail.com

## Getting Started
1. Clone the repository.
2. Create a virtual environment: `python -m venv chatbot_env`
3. Activate the environment: `source chatbot_env/bin/activate` (Linux/Mac) or `chatbot_env\Scripts\activate` (Windows)
4. Install requirements: `pip install -r requirements.txt`
5. Run the chatbot: `python main.py`

## Example Session
```
$ python main.py
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
Would you like to add, remove, or edit any symptoms? (y/n)
-> n
Diagnosis: [disease prediction]
Precautions: [recommendations]
```

## Contributing
1. Fork the repository or create a new branch.
2. Make your changes and add clear commit messages.
3. Test your changes locally.
4. Submit a pull request or merge request for review.

## Configuration
*If you add new data files, update the paths in `config.py` accordingly.*
