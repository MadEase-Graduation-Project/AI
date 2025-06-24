"""
Configuration file for Healthcare Chatbot
This file contains all file paths and configuration settings
"""

import os
from pathlib import Path

# Get the directory where this config.py file is located
BASE_DIR = Path(__file__).parent

# Data directory path
DATA_DIR = BASE_DIR / "Data"

# Model and report directories
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset file paths
TRAINING_DATA_PATH = DATA_DIR / "Training.csv"
TRAINING_AUGMENTED_DATA_PATH = DATA_DIR / "Training_augmented.csv"
TRAINING_AI_AUGMENTED_DATA_PATH = DATA_DIR / "Training_ai_augmented.csv"
TESTING_DATA_PATH = DATA_DIR / "Testing.csv"
DOCTORS_DATA_PATH = DATA_DIR / "doctors_dataset.csv"
SYMPTOM_SEVERITY_PATH = DATA_DIR / "Symptom_severity.csv"
SYMPTOM_DESCRIPTION_PATH = DATA_DIR / "symptom_Description.csv"
SYMPTOM_PRECAUTION_PATH = DATA_DIR / "symptom_precaution.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

# Text-to-Speech settings
TTS_RATE = 150
TTS_VOLUME = 0.9

# Chatbot settings
MAX_SYMPTOM_SEARCH_RESULTS = 10
SEVERITY_THRESHOLD = 13

# API configuration
API_BASE_URL = "https://api.freeapi.app/api/v1/public/instances?search="