import matplotlib
matplotlib.use('Agg')

# Numpy and pandas for mathematical operations
import numpy as np
import pandas as pd

# To read csv dataset files
import csv

# The preprocessing module provides functions for data preprocessing tasks such as scaling and handling missing data.
from sklearn import preprocessing

# For Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# train-test split and cross validation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Remove unecessary warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import configuration
from config import (
    TRAINING_DATA_PATH, 
    TESTING_DATA_PATH, 
    SYMPTOM_SEVERITY_PATH,
    SYMPTOM_DESCRIPTION_PATH,
    SYMPTOM_PRECAUTION_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TRAINING_AUGMENTED_DATA_PATH,
    TRAINING_AI_AUGMENTED_DATA_PATH
)

class DataPreprocessorFixed:
    def __init__(self):
        self.le = preprocessing.LabelEncoder()
        self.training = None
        self.testing = None
        self.doctors_df = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.cols = None
        self.x = None
        self.y = None
        
        # Initialize dictionaries to store symptom severity, description, and precautions
        self.severityDictionary = dict()
        self.description_list = dict()
        self.precautionDictionary = dict()
        self.symptoms_dict = {}

    def load_datasets(self, use_augmented=True, use_ai_augmented=False):
        """Load Datasets for training and testing"""
        try:
            if use_ai_augmented:
                self.training = pd.read_csv(TRAINING_AI_AUGMENTED_DATA_PATH)
                print("✓ Using AI-augmented dataset")
            elif use_augmented:
                self.training = pd.read_csv(TRAINING_AUGMENTED_DATA_PATH)
                print("✓ Using augmented dataset")
            else:
                self.training = pd.read_csv(TRAINING_DATA_PATH)
                print("✓ Using original dataset")
            
            print("✓ All datasets loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Error: Dataset file not found - {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            raise

    def explore_data(self):
        """Exploratory Data Analysis (EDA)"""
        # Number of rows and columns
        shape = self.training.shape
        print("Shape of Training dataset: ", shape)

        # Description about dataset
        description = self.training.describe()
        print(description)

        # Information about Dataset
        info_df = self.training.info()

        # To find total number of null values in dataset
        null_values_count = self.training.isnull().sum()
        print("Null values count:", null_values_count)

        # Print First eight rows of the Dataset
        print("First 8 rows:")
        print(self.training.head(8))

        # Figsize used to define size of the figure
        plt.figure(figsize=(10, 20))
        # Countplot from seaborn on the target varable and data accesed from Training dataset
        sns.countplot(y='prognosis', data=self.training)
        # Tile for title of the figur
        plt.title('Distribution of Target (Prognosis)')
        # Show used to display the figure on screen
        plt.show()

        # Analyze unique symptom combinations per disease
        self._analyze_symptom_diversity()

    def _analyze_symptom_diversity(self):
        """Analyze symptom diversity for each disease"""
        print("\n=== SYMPTOM DIVERSITY ANALYSIS ===")
        symptom_cols = [col for col in self.training.columns if col not in ['prognosis', 'Medical Specialties']]
        
        for disease in self.training['prognosis'].unique():
            disease_data = self.training[self.training['prognosis'] == disease]
            unique_combinations = len(disease_data[symptom_cols].drop_duplicates())
            total_cases = len(disease_data)
            print(f"{disease}: {unique_combinations} unique combinations out of {total_cases} total cases")

    def remove_duplicates_and_balance(self):
        """Remove duplicate symptom combinations and create balanced dataset"""
        print("\n🔄 Removing duplicate symptom combinations...")
        
        symptom_cols = [col for col in self.training.columns if col not in ['prognosis', 'Medical Specialties']]
        
        # Remove duplicates while keeping the first occurrence
        self.training = self.training.drop_duplicates(subset=symptom_cols)
        
        print(f"After removing duplicates: {len(self.training)} samples")
        
        # Analyze the new distribution
        print("\n=== NEW DISEASE DISTRIBUTION ===")
        disease_counts = self.training['prognosis'].value_counts()
        print(disease_counts)
        
        # Find the minimum number of samples per disease
        min_samples = disease_counts.min()
        print(f"\nMinimum samples per disease: {min_samples}")
        
        # Balance the dataset by sampling equal numbers from each disease
        balanced_data = []
        for disease in self.training['prognosis'].unique():
            disease_data = self.training[self.training['prognosis'] == disease]
            if len(disease_data) >= min_samples:
                # Sample min_samples from this disease
                sampled_data = disease_data.sample(n=min_samples, random_state=RANDOM_STATE)
                balanced_data.append(sampled_data)
            else:
                # If we don't have enough samples, use all available
                balanced_data.append(disease_data)
        
        # Combine all balanced data
        self.training = pd.concat(balanced_data, ignore_index=True)
        
        print(f"Final balanced dataset: {len(self.training)} samples")
        print(f"Final disease distribution:")
        print(self.training['prognosis'].value_counts())

    def preprocess_data(self):
        """Data Pre-processing with proper train/validation/test splits"""
        # First remove duplicates and balance the dataset
        self.remove_duplicates_and_balance()
        
        # Exclude 'prognosis' and 'Medical Specialties' columns
        self.cols = [col for col in self.training.columns if col not in ['prognosis', 'Medical Specialties']]
        self.x = self.training[self.cols]

        # y stores the target variable for disease prediction
        self.y = self.training['prognosis']

        # Encode labels
        self.le.fit(self.y)
        self.y = self.le.transform(self.y)

        # ✅ FIXED: Create proper train/validation/test splits
        # First split: 80% train+val, 20% test
        self.x_train_val, self.x_test, self.y_train_val, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=self.y
        )
        
        # Second split: 75% train, 25% validation (of the remaining 80%)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train_val,
            self.y_train_val,
            test_size=0.25,  # This gives us 20% validation of total data
            random_state=RANDOM_STATE,
            stratify=self.y_train_val
        )

        print(f"✓ Data split completed:")
        print(f"  Training set: {len(self.x_train)} samples")
        print(f"  Validation set: {len(self.x_val)} samples") 
        print(f"  Test set: {len(self.x_test)} samples")

        # Populate symptoms dictionary
        for index, symptom in enumerate(self.x):
            self.symptoms_dict[symptom] = index

        # Add reduced_data for chatbot compatibility
        # Structure: index=disease, columns=symptoms, values=0/1
        symptom_cols = self.cols
        reduced_df = self.training.groupby('prognosis')[symptom_cols].max()
        self.reduced_data = reduced_df

    def check_data_leakage(self):
        """Check for data leakage between train, validation, and test sets"""
        print("\n=== DATA LEAKAGE CHECK ===")
        
        # Check for exact duplicates between sets
        train_duplicates_in_val = 0
        train_duplicates_in_test = 0
        
        for idx, val_row in self.x_val.iterrows():
            for _, train_row in self.x_train.iterrows():
                if val_row.equals(train_row):
                    train_duplicates_in_val += 1
                    break
                    
        for idx, test_row in self.x_test.iterrows():
            for _, train_row in self.x_train.iterrows():
                if test_row.equals(train_row):
                    train_duplicates_in_test += 1
                    break
        
        print(f"Training samples found in validation set: {train_duplicates_in_val}")
        print(f"Training samples found in test set: {train_duplicates_in_test}")
        
        if train_duplicates_in_val > 0 or train_duplicates_in_test > 0:
            print("⚠️  WARNING: Data leakage detected!")
        else:
            print("✅ No data leakage detected")

    def getSeverityDict(self):
        """Function to read and store symptom severity information from a CSV file"""
        try:
            with open(SYMPTOM_SEVERITY_PATH, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if len(row) >= 2:  # Ensure row has at least 2 columns
                        try:
                            _diction = {row[0]: int(row[1])}
                            self.severityDictionary.update(_diction)
                        except ValueError:
                            # Skip rows where the second column is not a valid integer
                            continue
            print("✓ Severity dictionary loaded successfully")
        except FileNotFoundError:
            print(f"❌ Warning: Severity file not found at {SYMPTOM_SEVERITY_PATH}")
        except Exception as e:
            print(f"❌ Error loading severity dictionary: {e}")

    def getDescription(self):
        """Function to read and store symptom descriptions from a CSV file"""
        try:
            with open(SYMPTOM_DESCRIPTION_PATH, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if len(row) >= 2:  # Ensure row has at least 2 columns
                        _description = {row[0]: row[1]}
                        self.description_list.update(_description)
            print("✓ Description dictionary loaded successfully")
        except FileNotFoundError:
            print(f"❌ Warning: Description file not found at {SYMPTOM_DESCRIPTION_PATH}")
        except Exception as e:
            print(f"❌ Error loading description dictionary: {e}")

    def getprecautionDict(self):
        """Function to read and store symptom precaution information from a CSV file"""
        try:
            with open(SYMPTOM_PRECAUTION_PATH, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if len(row) >= 5:  # Ensure row has at least 5 columns
                        _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                        self.precautionDictionary.update(_prec)
            print("✓ Precaution dictionary loaded successfully")
        except FileNotFoundError:
            print(f"❌ Warning: Precaution file not found at {SYMPTOM_PRECAUTION_PATH}")
        except Exception as e:
            print(f"❌ Error loading precaution dictionary: {e}")

    def initialize_all(self, use_augmented=True, use_ai_augmented=False):
        """Initialize all data preprocessing steps"""
        print("🔄 Starting data preprocessing...")
        self.load_datasets(use_augmented, use_ai_augmented)
        self.explore_data()
        self.preprocess_data()
        self.check_data_leakage()
        self.getSeverityDict()
        self.getDescription()
        self.getprecautionDict()
        print("✅ Data preprocessing completed successfully!")
        return self 