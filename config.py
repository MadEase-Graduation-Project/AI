# config.py
import os

# Data paths
TRAINING_DATA_PATH = "Data/Training.csv"
TRAINING_AUGMENTED_DATA_PATH = "Data/Training_augmented.csv"
TRAINING_AI_AUGMENTED_DATA_PATH = "Data/Training_ai_augmented.csv"  # Updated path
TESTING_DATA_PATH = "Data/Testing.csv"  # Added missing constant

# Model paths
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Symptom severity and description paths
SYMPTOM_SEVERITY_PATH = "Data/Symptom_severity.csv"
SYMPTOM_DESCRIPTION_PATH = "Data/symptom_Description.csv"
SYMPTOM_PRECAUTION_PATH = "Data/symptom_precaution.csv"

# Text-to-Speech settings
TTS_RATE = 150
TTS_VOLUME = 0.9

# Search settings
MAX_SYMPTOM_SEARCH_RESULTS = 10

# Test size for data splitting
TEST_SIZE = 0.2  # Added missing constant

# Random state for reproducibility
RANDOM_STATE = 42
