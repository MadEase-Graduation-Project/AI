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

# UI and Display Constants
MAX_FOLLOW_UP_QUESTIONS = 6
MAX_FOLLOW_UP_ROUNDS = 2
MAX_HOSPITAL_DISPLAY = 5
MAX_DOCTOR_DISPLAY = 10

# Time and Date Constants
APPOINTMENT_DAYS_AHEAD = 7
TIME_SLOTS = [
    "09:00 AM", "10:00 AM", "11:00 AM", "12:00 PM",
    "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM"
]

# Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    'high_risk': 0.6,
    'medium_risk': 0.5,
    'low_risk': 0.4,
    'min_confidence': 0.05,
    'max_penalty': 0.75,
    'symptom_penalty': 0.1,
    'critical_penalty': 0.05,
    'severity_penalty': 0.05
}

# Severity Levels
SEVERITY_LEVELS = {
    'critical': 20,
    'high': 15,
    'moderate': 10,
    'mild': 5
}

# Booking Constants
BOOKING_REF_LENGTH = 8
BOOKING_ARRIVAL_MINUTES = 15
