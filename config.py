# config.py
import os

# Data paths
TRAINING_DATA_PATH = "Data/Training.csv"
# Note: TRAINING_AUGMENTED_DATA_PATH is not used in current implementation
# TRAINING_AUGMENTED_DATA_PATH = "Data/Training_augmented.csv"  # Commented out - file doesn't exist
TRAINING_SAFE_AUGMENTED_DATA_PATH = "Data/Training_safe_augmented.csv"  # Safe augmented data
TRAINING_AI_AUGMENTED_DATA_PATH = "Data/Training_ai_augmented.csv"  # Keep for reference
# Note: TESTING_DATA_PATH is not used in current implementation
# TESTING_DATA_PATH = "Data/Testing.csv"  # Commented out - file doesn't exist

# Model paths
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Symptom severity and description paths
SYMPTOM_SEVERITY_PATH = "Data/Symptom_severity.csv"
SYMPTOM_DESCRIPTION_PATH = "Data/symptom_Description.csv"
SYMPTOM_PRECAUTION_PATH = "Data/symptom_precaution.csv"

# Search settings
MAX_SYMPTOM_SEARCH_RESULTS = 10

# Test size for data splitting
TEST_SIZE = 0.2

# Random state for reproducibility
RANDOM_STATE = 42
