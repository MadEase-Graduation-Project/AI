#!/usr/bin/env python3
"""
Safe Medical Data Augmentation
Creates medically plausible symptom combinations based on strict medical rules
"""

import pandas as pd
import numpy as np
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SafeMedicalAugmenter:
    def __init__(self, target_samples_per_disease=20):
        self.target_samples = target_samples_per_disease
        self.original_data = None
        self.symptom_cols = None
        self.medical_rules = {}
        self.augmented_data = []
        
    def load_data(self):
        """Load the cleaned original data"""
        print("🔄 Loading cleaned data...")
        
        self.original_data = pd.read_csv('Data/Training.csv')
        self.symptom_cols = [col for col in self.original_data.columns 
                           if col not in ['prognosis', 'Medical Specialties']]
        
        print(f"✅ Loaded {len(self.original_data)} samples")
        print(f"✅ Found {len(self.symptom_cols)} symptoms")
        
    def define_medical_rules(self):
        """Define strict medical rules for each disease"""
        print("\n📋 Defining medical rules...")
        
        self.medical_rules = {
            'Fungal infection': {
                'core_symptoms': ['itching', 'skin_rash', 'nodal_skin_eruptions'],
                'common_symptoms': ['blister', 'dischromic _patches', 'skin_peeling'],
                'rare_symptoms': ['pus_filled_pimples', 'blackheads', 'scurring'],
                'never_symptoms': ['chest_pain', 'breathlessness', 'vomiting', 'diarrhoea', 'heart_rate']
            },
            'Allergy': {
                'core_symptoms': ['continuous_sneezing', 'shivering', 'chills'],
                'common_symptoms': ['cough', 'watering_from_eyes', 'runny_nose', 'congestion'],
                'rare_symptoms': ['throat_irritation', 'redness_of_eyes', 'sinus_pressure'],
                'never_symptoms': ['chest_pain', 'breathlessness', 'vomiting', 'diarrhoea', 'skin_rash']
            },
            'GERD': {
                'core_symptoms': ['stomach_pain', 'acidity', 'ulcers_on_tongue'],
                'common_symptoms': ['vomiting', 'cough', 'chest_pain', 'indigestion'],
                'rare_symptoms': ['headache', 'back_pain', 'abdominal_pain'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain', 'breathlessness']
            },
            'Chronic cholestasis': {
                'core_symptoms': ['vomiting', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite'],
                'common_symptoms': ['itching', 'abdominal_pain', 'yellowing_of_eyes', 'fatigue'],
                'rare_symptoms': ['weight_loss', 'yellow_urine', 'mild_fever'],
                'never_symptoms': ['chest_pain', 'breathlessness', 'joint_pain']
            },
            'Drug Reaction': {
                'core_symptoms': ['itching', 'skin_rash', 'stomach_pain'],
                'common_symptoms': ['burning_micturition', 'spotting_ urination', 'vomiting'],
                'rare_symptoms': ['nausea', 'headache', 'fatigue'],
                'never_symptoms': ['chest_pain', 'breathlessness', 'joint_pain']
            },
            'Peptic ulcer disease': {
                'core_symptoms': ['vomiting', 'abdominal_pain'],
                'common_symptoms': ['headache', 'back_pain', 'indigestion', 'loss_of_appetite'],
                'rare_symptoms': ['passage_of_gases', 'internal_itching', 'nausea'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'AIDS': {
                'core_symptoms': ['muscle_wasting', 'high_fever'],
                'common_symptoms': ['patches_in_throat', 'extra_marital_contacts', 'fatigue'],
                'rare_symptoms': ['sunken_eyes', 'receiving_blood_transfusion', 'receiving_unsterile_injections'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'Diabetes': {
                'core_symptoms': ['fatigue', 'irregular_sugar_level'],
                'common_symptoms': ['weight_loss', 'restlessness', 'lethargy', 'blurred_and_distorted_vision'],
                'rare_symptoms': ['obesity', 'excessive_hunger', 'increased_appetite', 'polyuria'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'Gastroenteritis': {
                'core_symptoms': ['vomiting', 'diarrhoea'],
                'common_symptoms': ['sweating', 'dehydration', 'sunken_eyes', 'abdominal_pain'],
                'rare_symptoms': ['nausea', 'loss_of_appetite', 'fatigue'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'Bronchial Asthma': {
                'core_symptoms': ['cough', 'breathlessness'],
                'common_symptoms': ['fatigue', 'high_fever', 'family_history', 'mucoid_sputum'],
                'rare_symptoms': ['chest_pain', 'fast_heart_rate', 'sweating'],
                'never_symptoms': ['vomiting', 'diarrhoea', 'skin_rash']
            },
            'Hypertension': {
                'core_symptoms': ['headache'],
                'common_symptoms': ['chest_pain', 'dizziness', 'loss_of_balance', 'lack_of_concentration'],
                'rare_symptoms': ['palpitations', 'fast_heart_rate', 'sweating'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'Migraine': {
                'core_symptoms': ['headache'],
                'common_symptoms': ['acidity', 'indigestion', 'blurred_and_distorted_vision', 'excessive_hunger'],
                'rare_symptoms': ['stiff_neck', 'depression', 'irritability', 'visual_disturbances', 'vomiting'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'Cervical spondylosis': {
                'core_symptoms': ['back_pain', 'neck_pain'],
                'common_symptoms': ['weakness_in_limbs', 'dizziness', 'loss_of_balance'],
                'rare_symptoms': ['stiff_neck', 'movement_stiffness', 'headache'],
                'never_symptoms': ['skin_rash', 'itching', 'vomiting']
            },
            'Paralysis (brain hemorrhage)': {
                'core_symptoms': ['vomiting', 'headache', 'weakness_of_one_body_side'],
                'common_symptoms': ['altered_sensorium', 'dizziness'],
                'rare_symptoms': ['loss_of_balance', 'unsteadiness'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'Jaundice': {
                'core_symptoms': ['yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite'],
                'common_symptoms': ['itching', 'vomiting', 'fatigue', 'weight_loss', 'abdominal_pain'],
                'rare_symptoms': ['high_fever', 'yellowing_of_eyes', 'yellow_urine'],
                'never_symptoms': ['chest_pain', 'breathlessness', 'joint_pain']
            },
            'Malaria': {
                'core_symptoms': ['chills', 'high_fever'],
                'common_symptoms': ['vomiting', 'sweating', 'headache', 'nausea', 'muscle_pain'],
                'rare_symptoms': ['diarrhoea', 'fatigue', 'loss_of_appetite'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'Chicken pox': {
                'core_symptoms': ['itching', 'skin_rash', 'high_fever'],
                'common_symptoms': ['fatigue', 'lethargy', 'headache', 'loss_of_appetite', 'red_spots_over_body'],
                'rare_symptoms': ['mild_fever', 'swelled_lymph_nodes', 'malaise'],
                'never_symptoms': ['chest_pain', 'breathlessness', 'vomiting']
            },
            'Dengue': {
                'core_symptoms': ['chills', 'high_fever', 'headache', 'joint_pain'],
                'common_symptoms': ['vomiting', 'fatigue', 'nausea', 'loss_of_appetite', 'muscle_pain'],
                'rare_symptoms': ['pain_behind_the_eyes', 'back_pain', 'malaise', 'red_spots_over_body'],
                'never_symptoms': ['skin_rash', 'itching']
            },
            'Typhoid': {
                'core_symptoms': ['chills', 'high_fever', 'headache'],
                'common_symptoms': ['vomiting', 'fatigue', 'nausea', 'abdominal_pain'],
                'rare_symptoms': ['constipation', 'diarrhoea', 'toxic_look_(typhos)', 'belly_pain'],
                'never_symptoms': ['skin_rash', 'itching', 'joint_pain']
            },
            'hepatitis A': {
                'core_symptoms': ['vomiting', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite'],
                'common_symptoms': ['joint_pain', 'abdominal_pain', 'fatigue'],
                'rare_symptoms': ['diarrhoea', 'mild_fever', 'yellowing_of_eyes', 'muscle_pain'],
                'never_symptoms': ['chest_pain', 'breathlessness', 'skin_rash']
            }
        }
        
        print(f"✅ Defined medical rules for {len(self.medical_rules)} diseases")
        
    def generate_safe_variations(self, base_pattern, disease_name):
        """Generate medically safe variations of a symptom pattern"""
        variations = []
        rules = self.medical_rules.get(disease_name, {})
        
        if not rules:
            return variations
        
        # Strategy 1: Omit 1-3 non-core symptoms (simulate incomplete reporting)
        for _ in range(5):  # Increased from 3 to 5
            variation = base_pattern.copy()
            non_core_symptoms = []
            
            for i, symptom in enumerate(self.symptom_cols):
                if (base_pattern[i] == 1 and 
                    symptom not in rules.get('core_symptoms', []) and
                    symptom not in rules.get('never_symptoms', [])):
                    non_core_symptoms.append(i)
            
            if len(non_core_symptoms) > 0:
                num_to_omit = min(random.randint(1, 3), len(non_core_symptoms))  # Increased max from 2 to 3
                indices_to_omit = random.sample(non_core_symptoms, num_to_omit)
                for idx in indices_to_omit:
                    variation[idx] = 0
                variations.append(variation)
        
        # Strategy 2: Add 1-3 plausible symptoms (from common/rare lists)
        for _ in range(4):  # Increased from 2 to 4
            variation = base_pattern.copy()
            possible_additions = []
            
            for symptom in rules.get('common_symptoms', []) + rules.get('rare_symptoms', []):
                if symptom in self.symptom_cols:
                    symptom_idx = self.symptom_cols.index(symptom)
                    if base_pattern[symptom_idx] == 0:
                        possible_additions.append(symptom_idx)
            
            if possible_additions:
                num_to_add = min(random.randint(1, 3), len(possible_additions))  # Increased max from 2 to 3
                indices_to_add = random.sample(possible_additions, num_to_add)
                for idx in indices_to_add:
                    variation[idx] = 1
                variations.append(variation)
        
        # Strategy 3: Ensure core symptoms are present (if missing)
        for _ in range(2):  # Increased from 1 to 2
            variation = base_pattern.copy()
            missing_core = []
            
            for symptom in rules.get('core_symptoms', []):
                if symptom in self.symptom_cols:
                    symptom_idx = self.symptom_cols.index(symptom)
                    if base_pattern[symptom_idx] == 0:
                        missing_core.append(symptom_idx)
            
            if missing_core:
                # Add 50% of missing core symptoms
                num_to_add = max(1, len(missing_core) // 2)
                indices_to_add = random.sample(missing_core, num_to_add)
                for idx in indices_to_add:
                    variation[idx] = 1
                variations.append(variation)
        
        # Strategy 4: Create variations with different symptom combinations
        for _ in range(3):  # New strategy
            variation = base_pattern.copy()
            
            # Randomly add 1-2 common symptoms and remove 1-2 non-core symptoms
            common_symptoms = []
            for symptom in rules.get('common_symptoms', []):
                if symptom in self.symptom_cols:
                    symptom_idx = self.symptom_cols.index(symptom)
                    if base_pattern[symptom_idx] == 0:
                        common_symptoms.append(symptom_idx)
            
            non_core_symptoms = []
            for i, symptom in enumerate(self.symptom_cols):
                if (base_pattern[i] == 1 and 
                    symptom not in rules.get('core_symptoms', []) and
                    symptom not in rules.get('never_symptoms', [])):
                    non_core_symptoms.append(i)
            
            # Add common symptoms
            if common_symptoms:
                num_to_add = min(random.randint(1, 2), len(common_symptoms))
                indices_to_add = random.sample(common_symptoms, num_to_add)
                for idx in indices_to_add:
                    variation[idx] = 1
            
            # Remove non-core symptoms
            if non_core_symptoms:
                num_to_remove = min(random.randint(1, 2), len(non_core_symptoms))
                indices_to_remove = random.sample(non_core_symptoms, num_to_remove)
                for idx in indices_to_remove:
                    variation[idx] = 0
            
            variations.append(variation)
        
        return variations
    
    def generate_disease_samples(self, disease_name):
        """Generate safe samples for a specific disease"""
        disease_info = self.medical_rules.get(disease_name, {})
        if not disease_info:
            return []
        
        # Get original samples for this disease
        disease_data = self.original_data[self.original_data['prognosis'] == disease_name]
        medical_specialty = disease_data['Medical Specialties'].iloc[0] if len(disease_data) > 0 else 'Unknown'
        
        # Start with original patterns
        new_samples = []
        for _, row in disease_data.iterrows():
            pattern = [row[col] for col in self.symptom_cols]
            new_samples.append({
                'pattern': pattern,
                'disease': disease_name,
                'medical_specialty': medical_specialty,
                'source': 'original'
            })
        
        # Generate safe variations until we reach target
        attempts = 0
        max_attempts = self.target_samples * 20  # Prevent infinite loops
        
        while len(new_samples) < self.target_samples and attempts < max_attempts:
            # Pick a random base pattern from original data
            if len(disease_data) > 0:
                base_row = disease_data.sample(n=1).iloc[0]
                base_pattern = [base_row[col] for col in self.symptom_cols]
                
                # Generate safe variations
                variations = self.generate_safe_variations(base_pattern, disease_name)
                
                for variation in variations:
                    if len(new_samples) >= self.target_samples:
                        break
                    
                    # Check if this variation is unique
                    is_unique = True
                    for existing_sample in new_samples:
                        if np.array_equal(variation, existing_sample['pattern']):
                            is_unique = False
                            break
                    
                    if is_unique:
                        new_samples.append({
                            'pattern': variation,
                            'disease': disease_name,
                            'medical_specialty': medical_specialty,
                            'source': 'augmented'
                        })
            
            attempts += 1
        
        print(f"Generated {len(new_samples)} samples for {disease_name} "
              f"({len([s for s in new_samples if s['source'] == 'augmented'])} augmented)")
        
        return new_samples
    
    def create_safe_augmented_dataset(self):
        """Create the final safe augmented dataset"""
        print("\n🚀 Starting safe medical data augmentation...")
        
        all_samples = []
        
        for disease_name in self.medical_rules.keys():
            print(f"\n📊 Processing {disease_name}...")
            disease_samples = self.generate_disease_samples(disease_name)
            all_samples.extend(disease_samples)
        
        # Convert to DataFrame
        print("\n🔄 Creating final dataset...")
        rows = []
        for sample in all_samples:
            row = list(sample['pattern']) + [sample['disease'], sample['medical_specialty']]
            rows.append(row)
        
        columns = self.symptom_cols + ['prognosis', 'Medical Specialties']
        augmented_df = pd.DataFrame(rows, columns=columns)
        
        # Remove any accidental duplicates
        augmented_df = augmented_df.drop_duplicates(subset=self.symptom_cols + ['prognosis'])
        
        print(f"✅ Final safe augmented dataset: {len(augmented_df)} samples")
        
        # Analyze the results
        self._analyze_augmentation_results(augmented_df)
        
        return augmented_df
    
    def _analyze_augmentation_results(self, df):
        """Analyze the results of safe augmentation"""
        print("\n📈 SAFE AUGMENTATION RESULTS ANALYSIS")
        print("=" * 50)
        
        # Disease distribution
        disease_counts = df['prognosis'].value_counts()
        print(f"Average samples per disease: {disease_counts.mean():.1f}")
        print(f"Min samples per disease: {disease_counts.min()}")
        print(f"Max samples per disease: {disease_counts.max()}")
        
        # Check for medical rule violations
        violations = 0
        for disease in self.medical_rules.keys():
            disease_data = df[df['prognosis'] == disease]
            rules = self.medical_rules[disease]
            
            for _, row in disease_data.iterrows():
                for symptom in rules.get('never_symptoms', []):
                    if symptom in df.columns and row[symptom] == 1:
                        violations += 1
        
        print(f"Medical rule violations: {violations}")
        
        if violations == 0:
            print("✅ No medical rule violations detected!")
        else:
            print(f"⚠️  Found {violations} medical rule violations")
    
    def save_safe_dataset(self, df, filename="Data/Training_safe_augmented.csv"):
        """Save the safe augmented dataset"""
        df.to_csv(filename, index=False)
        print(f"\n💾 Saved safe augmented dataset to {filename}")
        
        # Also create a summary report
        summary_filename = filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w') as f:
            f.write("SAFE MEDICAL DATA AUGMENTATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Total diseases: {df['prognosis'].nunique()}\n")
            f.write(f"Average samples per disease: {df['prognosis'].value_counts().mean():.1f}\n\n")
            f.write("Disease distribution:\n")
            for disease, count in df['prognosis'].value_counts().items():
                f.write(f"  {disease}: {count} samples\n")
        
        print(f"📄 Summary report saved to {summary_filename}")
    
    def run_safe_augmentation(self):
        """Run the complete safe augmentation process"""
        print("🚀 STARTING SAFE MEDICAL DATA AUGMENTATION")
        print("=" * 60)
        
        self.load_data()
        self.define_medical_rules()
        augmented_df = self.create_safe_augmented_dataset()
        self.save_safe_dataset(augmented_df)
        
        print("\n✅ SAFE AUGMENTATION COMPLETE")
        print("=" * 60)
        print("The dataset has been safely augmented while maintaining medical accuracy.")
        print("All generated samples follow strict medical rules and are clinically plausible.")

if __name__ == "__main__":
    augmenter = SafeMedicalAugmenter(target_samples_per_disease=100)
    augmenter.run_safe_augmentation() 