# chatbot_interface.py

print("DEBUG: chatbot_interface.py loaded from ENHANCED PROJECT COPY 2")

import pyttsx3
import re
from sklearn.tree import _tree
from config import TTS_RATE, TTS_VOLUME, MAX_SYMPTOM_SEARCH_RESULTS
import requests
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import spacy
from spacy.matcher import PhraseMatcher
import difflib
from difflib import SequenceMatcher
import traceback

# قاموس مرادفات الأعراض (عربي -> إنجليزي)
symptom_synonyms = {
    "صداع": "headache",
    "حرارة": "fever",
    "مغص": "abdominal_pain",
    "تعب": "fatigue",
    "سعال": "cough",
    "اسهال": "diarrhoea",
    "قيء": "vomiting",
    "غثيان": "nausea",
    "طفح": "skin_rash",
    "حكة": "itching",
    "دوخة": "dizziness",
    "خمول": "malaise",
    "امساك": "constipation",
    "رعشة": "shivering",
    "عطس": "continuous_sneezing",
    "الم البطن": "abdominal_pain",
    "الم المعدة": "stomach_pain",
    "ارهاق": "fatigue",
    # أضف المزيد حسب الحاجة
}

class ChatbotInterface:
    def __init__(self, data_preprocessor, model_trainer):
        self.data_preprocessor = data_preprocessor
        self.model_trainer = model_trainer
        self.user_location = None
        self.production_mode = False
        self.disease_to_specialty = self._build_disease_specialty_mapping()
        
        # Fix feature names issue by ensuring consistency
        self.feature_names = self.data_preprocessor.cols
        self.model_trainer.clf.feature_names_in_ = np.array(self.feature_names)
        
        # Medical rules for validation
        self.medical_rules = self._build_medical_rules()
        
        # Symptom clusters for better matching
        self.symptom_clusters = self._build_symptom_clusters()

        try:
            self.engine = pyttsx3.init()
            self.tts_available = True
            print("✓ Text-to-Speech initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Text-to-Speech initialization failed: {e}")
            self.tts_available = False

    def handle_single_symptom_input(self, symptom_input):
        """Robustly handle a single symptom input, with typo correction and generic term handling."""
        corrected = []
        token = symptom_input.strip().lower().replace(' ', '_')
        similar_options = [s for s in self.feature_names if token in s]
        if len(similar_options) >= 2:
            print(f"You entered '{token}'. Please specify the type(s):")
            for i, opt in enumerate(similar_options):
                print(f"  {i+1}) {opt}")
            print(f"  0) None of these / skip")
            selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,3,5): ").strip()
            indices = [int(x) for x in selected.split(',') if x.strip().isdigit()]
            for idx in indices:
                if 1 <= idx <= len(similar_options):
                    if similar_options[idx-1] not in corrected:
                        corrected.append(similar_options[idx-1])
            return corrected
        if token in self.feature_names:
            corrected.append(token)
            return corrected
        matches = difflib.get_close_matches(token, self.feature_names, n=3, cutoff=0.0)
        if matches:
            best = matches[0]
            ratio = SequenceMatcher(None, token, best).ratio()
            if ratio > 0.8:
                print(f"Interpreting '{token}' as '{best}'.")
                # After fuzzy correction, check if the corrected token is a generic term
                corrected_token = best
                similar_options = [s for s in self.feature_names if corrected_token in s]
                if len(similar_options) >= 2:
                    print(f"Please specify the type(s) of '{corrected_token}':")
                    for i, opt in enumerate(similar_options):
                        print(f"  {i+1}) {opt}")
                    print(f"  0) None of these / skip")
                    selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,3,5): ").strip()
                    indices = [int(x) for x in selected.split(',') if x.strip().isdigit()]
                    for idx in indices:
                        if 1 <= idx <= len(similar_options):
                            if similar_options[idx-1] not in corrected:
                                corrected.append(similar_options[idx-1])
                    return corrected
                elif corrected_token in self.feature_names:
                    corrected.append(corrected_token)
                    return corrected
                else:
                    # Fallback: just return the best match even if not in feature_names
                    corrected.append(best)
                    return corrected
            elif ratio > 0.6:
                print(f"Unrecognized symptom: '{token}'. Did you mean:")
                for i, match in enumerate(matches):
                    print(f"  {i+1}) {match}")
                print(f"  0) None of these / skip")
                while True:
                    choice = input(f"Select the correct symptom for '{token}' (1-{len(matches)} or 0 to skip): ").strip()
                    if choice.isdigit():
                        idx = int(choice)
                        if idx == 0:
                            print(f"Skipping '{token}'.")
                            break
                        elif 1 <= idx <= len(matches):
                            if matches[idx-1] not in corrected:
                                corrected.append(matches[idx-1])
                            break
                    print("Invalid choice. Please try again.")
                return corrected
        print(f"No good match found for '{token}'. Skipping.")
        return corrected

    def _build_medical_rules(self):
        """Build medical validation rules"""
        return {
            'red_flags': {
                'chest_pain': ['Heart attack', 'Pneumonia', 'GERD'],
                'severe_headache': ['Migraine', 'Hypertension', 'Brain hemorrhage'],
                'high_fever': ['Malaria', 'Typhoid', 'Dengue'],
                'yellowing_skin': ['Jaundice', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C'],
                'blood_in_sputum': ['Tuberculosis', 'Pneumonia'],
                'paralysis': ['Paralysis (brain hemorrhage)', 'Stroke']
            },
            'symptom_disease_mapping': {
                'itching': ['Fungal infection', 'Allergy', 'Drug Reaction', 'Psoriasis', 'Chicken pox'],
                'internal_itching': ['Jaundice', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Chronic cholestasis'],
                'headache': ['Migraine', 'Hypertension', 'Brain hemorrhage', 'Cervical spondylosis'],
                'fever': ['Malaria', 'Typhoid', 'Dengue', 'Tuberculosis', 'Pneumonia'],
                'cough': ['Bronchial Asthma', 'Tuberculosis', 'Pneumonia', 'Common Cold'],
                'abdominal_pain': ['Peptic ulcer disease', 'Gastroenteritis', 'Chronic cholestasis'],
                'skin_rash': ['Fungal infection', 'Allergy', 'Drug Reaction', 'Chicken pox'],
                'yellowing_of_eyes': ['Jaundice', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C'],
                'dark_urine': ['Jaundice', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C'],
                'fatigue': ['Malaria', 'Typhoid', 'Dengue', 'Tuberculosis', 'Diabetes'],
                'muscle_pain': ['Malaria', 'Dengue', 'Typhoid', 'Tuberculosis'],
                'vomiting': ['Gastroenteritis', 'Peptic ulcer disease', 'Malaria', 'Dengue', 'Typhoid'],
                'nausea': ['Gastroenteritis', 'Peptic ulcer disease', 'Malaria', 'Dengue', 'Typhoid'],
                'diarrhoea': ['Gastroenteritis', 'Peptic ulcer disease', 'Chronic cholestasis', 'Typhoid'],
                # Additional common symptoms
                'chest_pain': ['Heart attack', 'Pneumonia', 'GERD', 'Bronchial Asthma'],
                'breathlessness': ['Bronchial Asthma', 'Pneumonia', 'Heart attack', 'Tuberculosis'],
                'joint_pain': ['Arthritis', 'Osteoarthristis', 'Rheumatoid arthritis'],
                'stomach_pain': ['Peptic ulcer disease', 'Gastroenteritis', 'Chronic cholestasis'],
                'acidity': ['GERD', 'Peptic ulcer disease', 'Gastroenteritis'],
                'burning_micturition': ['Urinary tract infection', 'Diabetes'],
                'weight_loss': ['Diabetes', 'Tuberculosis', 'Hyperthyroidism', 'AIDS'],
                'weight_gain': ['Hypothyroidism', 'Diabetes'],
                'anxiety': ['Hypertension', 'Diabetes', 'Hyperthyroidism'],
                'mood_swings': ['Diabetes', 'Hyperthyroidism', 'Hypothyroidism'],
                'cold_hands_and_feets': ['Hypothyroidism', 'Diabetes'],
                'shivering': ['Malaria', 'Typhoid', 'Dengue', 'Pneumonia'],
                'chills': ['Malaria', 'Typhoid', 'Dengue', 'Pneumonia'],
                'continuous_sneezing': ['Allergy', 'Common Cold'],
                'ulcers_on_tongue': ['Peptic ulcer disease', 'Gastroenteritis'],
                'muscle_wasting': ['Diabetes', 'Tuberculosis', 'AIDS'],
                'spotting_ urination': ['Urinary tract infection', 'Diabetes'],
                'constipation': ['Chronic cholestasis', 'Peptic ulcer disease'],
                'family_history': ['Diabetes', 'Hypertension', 'Heart attack'],
                'mucoid_sputum': ['Tuberculosis', 'Pneumonia', 'Bronchial Asthma'],
                'rusty_sputum': ['Tuberculosis', 'Pneumonia'],
                'lack_of_concentration': ['Diabetes', 'Hypertension', 'Hyperthyroidism'],
                'visual_disturbances': ['Diabetes', 'Hypertension', 'Migraine'],
                'receiving_blood_transfusion': ['Hepatitis B', 'Hepatitis C', 'AIDS'],
                'receiving_unsterile_injections': ['Hepatitis B', 'Hepatitis C', 'AIDS'],
                'coma': ['Diabetes', 'Brain hemorrhage', 'Hypertension'],
                'stiff_neck': ['Meningitis', 'Cervical spondylosis'],
                'loss_of_balance': ['(vertigo) Paroymsal  Positional Vertigo', 'Brain hemorrhage'],
                'unsteadiness': ['(vertigo) Paroymsal  Positional Vertigo', 'Brain hemorrhage'],
                'weakness_of_one_body_side': ['Paralysis (brain hemorrhage)', 'Stroke'],
                'altered_sensorium': ['Diabetes', 'Brain hemorrhage', 'Hypertension'],
                'bladder_discomfort': ['Urinary tract infection', 'Diabetes'],
                'foul_smell_of urine': ['Urinary tract infection', 'Diabetes'],
                'continuous_feel_of_urine': ['Urinary tract infection', 'Diabetes'],
                'passage_of_gases': ['Gastroenteritis', 'Peptic ulcer disease'],
                'internal_itching': ['Jaundice', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C'],
                'toxic_look_(typhos)': ['Typhoid', 'Malaria', 'Dengue'],
                'yellowing_skin': ['Jaundice', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C'],
                'yellow_crust_ooze': ['Impetigo', 'Fungal infection'],
                'prognosis': ['All diseases']  # This will be filtered out
            },
            'body_systems': {
                'cardiovascular': ['chest_pain', 'palpitations', 'hypertension'],
                'respiratory': ['cough', 'breathlessness', 'chest_pain'],
                'gastrointestinal': ['abdominal_pain', 'nausea', 'vomiting', 'diarrhoea'],
                'neurological': ['headache', 'paralysis', 'altered_sensorium'],
                'dermatological': ['itching', 'skin_rash', 'nodal_skin_eruptions'],
                'hepatobiliary': ['internal_itching', 'yellowing_of_eyes', 'dark_urine', 'yellowing_skin']
            }
        }

    def _build_symptom_clusters(self):
        """Build symptom clusters for better matching"""
        return {
            'headache_cluster': ['headache', 'migraine', 'severe_headache', 'mild_headache'],
            'fever_cluster': ['fever', 'high_fever', 'mild_fever', 'shivering'],
            'respiratory_cluster': ['cough', 'breathlessness', 'chest_pain', 'blood_in_sputum'],
            'gastro_cluster': ['abdominal_pain', 'nausea', 'vomiting', 'diarrhoea', 'constipation'],
            'skin_cluster': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'blister'],
            'internal_cluster': ['internal_itching', 'yellowing_of_eyes', 'dark_urine', 'abdominal_pain'],
            'fatigue_cluster': ['fatigue', 'malaise', 'weakness', 'loss_of_appetite'],
            'hepatobiliary_cluster': ['internal_itching', 'yellowing_of_eyes', 'dark_urine', 'yellowing_skin']
        }

    def _build_disease_specialty_mapping(self):
        """Build a mapping from disease to medical specialty using training data"""
        mapping = {}
        if hasattr(self.data_preprocessor, 'training'):
            df = self.data_preprocessor.training
            if 'prognosis' in df.columns and 'Medical Specialties' in df.columns:
                for _, row in df.iterrows():
                    mapping[row['prognosis']] = row['Medical Specialties']
        return mapping

    def text_to_speech(self, text):
        """Function to convert text to speech"""
        if not self.tts_available:
            return
        try:
            self.engine.setProperty('rate', TTS_RATE)
            self.engine.setProperty('volume', TTS_VOLUME)
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️  Warning: Text-to-Speech error: {e}")

    def getInfo(self):
        """Get user information and location"""
        print("-----------------------------------HealthCare ChatBot-----------------------------------")
        print("\nYour Name? \t\t\t\t\t", end="->")
        while True:
            name = input("").strip()
            if name.lower() == 'undo':
                print("Undo: Please enter your name again.")
                continue
            if name:
                print(f"Hello {name}! 👋")
            else:
                print("Hello! 👋")
                name = "User"
            break
        while True:
            location = self.get_location()
            if location:
                self.user_location = location
                break
        # Return both name and location for further undo handling in the next step
        return name, self.user_location

    def get_location(self):
        while True:
            location = input("Please enter your location (city or country): ").strip()
            if location.lower() == 'undo':
                print("Let's try entering your location again.")
                continue  # Repeats the prompt
            if location:  # If not empty
                return location
            else:
                print("Location cannot be empty. Please try again.")

    def check_pattern(self, dis_list, inp):
        """Check pattern matching for symptoms with improved matching"""
        try:
            inp = inp.replace(' ', '_').lower()
            
            # First try exact match
            if inp in dis_list:
                return (1, [inp])
            
            # Then try prefix matching (symptoms that start with the input)
            prefix_matches = [item for item in dis_list if item.lower().startswith(inp)]
            
            # Also try substring matching (symptoms that contain the input)
            substring_matches = [item for item in dis_list if inp in item.lower()]
            
            # Combine and remove duplicates while preserving order
            all_matches = []
            for item in prefix_matches + substring_matches:
                if item not in all_matches:
                    all_matches.append(item)
            
            if len(all_matches) > MAX_SYMPTOM_SEARCH_RESULTS:
                all_matches = all_matches[:MAX_SYMPTOM_SEARCH_RESULTS]
            
            return (1, all_matches) if all_matches else (0, [])
        except Exception as e:
            print(f"❌ Error in pattern matching: {e}")
            return 0, []

    def get_closest_symptom(self, user_input):
        """Suggest the closest valid symptom using fuzzy matching."""
        matches = difflib.get_close_matches(user_input, self.feature_names, n=1, cutoff=0.7)
        return matches[0] if matches else None

    def get_valid_input(self, prompt, valid_options=None, input_type=str, symptom_mode=False):
        """Get valid input from user with error handling and fuzzy matching for symptoms if symptom_mode is True."""
        while True:
            user_input = input(prompt).strip().lower()
            if user_input == 'undo':
                return 'undo'
            if user_input == 'main':
                return 'main_menu'
            # Normalize yes/no/y/n
            if valid_options:
                yn_map = {'y': 'yes', 'n': 'no', 'yes': 'yes', 'no': 'no'}
                if set(valid_options) == set(['yes', 'no']) or set(valid_options) == set(['yes', 'no', 'back']):
                    if user_input in yn_map:
                        mapped = yn_map[user_input]
                        if mapped in valid_options:
                            return mapped
                    if 'back' in valid_options and user_input == 'back':
                        return 'back'
                    print("Please enter 'yes'/'y' or 'no'/'n'" + (" or 'back'" if 'back' in valid_options else "") + ".")
                    continue
                if set(valid_options) == set(['y', 'n']):
                    if user_input in yn_map:
                        mapped = yn_map[user_input]
                        return 'y' if mapped == 'yes' else 'n'
                    print("Please enter 'yes'/'y' or 'no'/'n'.")
                    continue
                if user_input in valid_options:
                    return user_input
                print(f"Please enter one of: {', '.join(valid_options)}")
                continue
            if input_type == int:
                try:
                    return int(user_input)
                except ValueError:
                    print("Please enter a valid number.")
                    continue
            elif symptom_mode:
                # For symptom mode, just return the raw input and let handle_single_symptom_input process it
                return user_input
            else:
                return user_input

    def _get_feature_importances(self):
        """Get feature importances from the underlying model, even if calibrated."""
        clf = self.model_trainer.clf
        if isinstance(clf, CalibratedClassifierCV):
            # Use the first calibrated classifier's estimator
            return clf.calibrated_classifiers_[0].estimator.feature_importances_
        return clf.feature_importances_

    def get_medical_relevant_symptoms(self, initial_symptom):
        """Get medically relevant symptoms based on medical knowledge"""
        relevant_symptoms = []
        
        # Check medical rules for symptom-disease mapping
        for symptom, diseases in self.medical_rules['symptom_disease_mapping'].items():
            if symptom == initial_symptom:  # Exact match for better precision
                # Get symptoms for these diseases
                for disease in diseases:
                    if disease in self.data_preprocessor.reduced_data.index:
                        disease_symptoms = []
                        for col in self.data_preprocessor.reduced_data.columns:
                            if col != "Medical Specialties" and self.data_preprocessor.reduced_data.loc[disease, col] == 1:
                                disease_symptoms.append(col)
                        relevant_symptoms.extend(disease_symptoms)
        
        # Remove duplicates and initial symptom
        relevant_symptoms = list(set(relevant_symptoms))
        if initial_symptom in relevant_symptoms:
            relevant_symptoms.remove(initial_symptom)
        
        # Filter symptoms based on the type of initial symptom
        if initial_symptom == 'itching':
            # For skin itching, prioritize skin-related symptoms
            skin_related = ['skin_rash', 'nodal_skin_eruptions', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
            filtered_symptoms = [s for s in relevant_symptoms if s in skin_related]
            if filtered_symptoms:
                relevant_symptoms = filtered_symptoms
        elif initial_symptom == 'internal_itching':
            # For internal itching, prioritize hepatobiliary symptoms
            hepatobiliary_related = ['yellowing_of_eyes', 'dark_urine', 'yellowing_skin', 'abdominal_pain']
            filtered_symptoms = [s for s in relevant_symptoms if s in hepatobiliary_related]
            if filtered_symptoms:
                relevant_symptoms = filtered_symptoms
        elif initial_symptom in ['chest_pain', 'breathlessness']:
            # For respiratory symptoms, prioritize respiratory-related symptoms
            respiratory_related = ['cough', 'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum', 'fever']
            filtered_symptoms = [s for s in relevant_symptoms if s in respiratory_related]
            if filtered_symptoms:
                relevant_symptoms = filtered_symptoms
        elif initial_symptom in ['vomiting', 'nausea', 'diarrhoea', 'abdominal_pain', 'stomach_pain']:
            # For gastrointestinal symptoms, prioritize GI-related symptoms
            gi_related = ['acidity', 'ulcers_on_tongue', 'passage_of_gases', 'constipation', 'fatigue', 'fever', 'chills', 'loss_of_appetite', 'weight_loss']
            filtered_symptoms = [s for s in relevant_symptoms if s in gi_related]
            if filtered_symptoms:
                relevant_symptoms = filtered_symptoms
        elif initial_symptom in ['burning_micturition', 'spotting_ urination', 'bladder_discomfort']:
            # For urinary symptoms, prioritize urinary-related symptoms
            urinary_related = ['foul_smell_of urine', 'continuous_feel_of_urine', 'fatigue', 'fever']
            filtered_symptoms = [s for s in relevant_symptoms if s in urinary_related]
            if filtered_symptoms:
                relevant_symptoms = filtered_symptoms
        elif initial_symptom in ['headache', 'migraine']:
            # For neurological symptoms, prioritize neuro-related symptoms
            neuro_related = ['visual_disturbances', 'lack_of_concentration', 'altered_sensorium', 'nausea']
            filtered_symptoms = [s for s in relevant_symptoms if s in neuro_related]
            if filtered_symptoms:
                relevant_symptoms = filtered_symptoms
        elif initial_symptom in ['joint_pain', 'muscle_pain']:
            # For musculoskeletal symptoms, prioritize MSK-related symptoms
            msk_related = ['muscle_wasting', 'fatigue', 'weakness', 'stiff_neck']
            filtered_symptoms = [s for s in relevant_symptoms if s in msk_related]
            if filtered_symptoms:
                relevant_symptoms = filtered_symptoms
        
        # Sort by feature importance
        symptom_importance = []
        feature_importances = self._get_feature_importances()
        for symptom in relevant_symptoms:
            if symptom in self.feature_names:
                idx = self.feature_names.index(symptom)
                importance = feature_importances[idx]
                symptom_importance.append((symptom, importance))
        
        # Sort by importance and return top 10 (increased from 5)
        symptom_importance.sort(key=lambda x: x[1], reverse=True)
        return [symptom for symptom, _ in symptom_importance[:10]]

    def validate_medical_prediction(self, symptoms, predicted_disease, confidence):
        """Validate prediction using medical rules"""
        # Check for red flag symptoms
        red_flags = []
        for symptom in symptoms:
            if symptom in self.medical_rules['red_flags']:
                expected_diseases = self.medical_rules['red_flags'][symptom]
                if predicted_disease not in expected_diseases:
                    red_flags.append(f"{symptom} suggests {', '.join(expected_diseases)}")
        
        # Check symptom-disease consistency
        if predicted_disease in self.data_preprocessor.reduced_data.index:
            disease_symptoms = []
            for col in self.data_preprocessor.reduced_data.columns:
                if col != "Medical Specialties" and self.data_preprocessor.reduced_data.loc[predicted_disease, col] == 1:
                    disease_symptoms.append(col)
            
            # Calculate symptom match percentage
            matched_symptoms = [s for s in symptoms if s in disease_symptoms]
            match_percentage = len(matched_symptoms) / len(symptoms) if symptoms else 0
            
            # Adjust confidence based on medical validation
            if match_percentage > 0.8:
                confidence *= 1.2  # Boost confidence for good matches
            elif match_percentage < 0.5:
                confidence *= 0.8  # Reduce confidence for poor matches
        
        return confidence, red_flags

    def predict_disease_with_medical_validation(self, confirmed_symptoms):
        """Predict disease with medical validation, severity analysis, and improved confidence"""
        # Calculate symptom severity
        severity_score, severity_level, high_severity_symptoms, present_symptoms = self.calculate_symptom_severity(confirmed_symptoms)
        
        # Create input vector using severity scores
        input_vector = [self.data_preprocessor.severityDictionary.get(feature, 0) if feature in confirmed_symptoms else 0 for feature in self.feature_names]
        
        # Get prediction probabilities
        predicted_proba = self.model_trainer.clf.predict_proba([input_vector])[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predicted_proba)[::-1][:3]
        top_3_diseases = self.data_preprocessor.le.inverse_transform(top_3_indices)
        top_3_confidences = predicted_proba[top_3_indices]
        
        # Apply medical validation to top prediction
        top_disease = top_3_diseases[0]
        top_confidence = top_3_confidences[0]
        
        # Validate and adjust confidence
        adjusted_confidence, red_flags = self.validate_medical_prediction(
            confirmed_symptoms, top_disease, top_confidence
        )
        
        # Further adjust confidence based on severity
        final_confidence = self.adjust_confidence_by_severity(adjusted_confidence, severity_score, high_severity_symptoms)
        
        # Cap confidence at 1.0
        final_confidence = min(final_confidence, 1.0)
        
        # Update confidence in the list
        top_3_confidences[0] = final_confidence
        
        # Check if adjusted confidence is sufficient
        if final_confidence < 0.4:  # Higher threshold for medical predictions
            if red_flags:
                print("🔴 Medical considerations:")
                for flag in red_flags:
                    print(f"   - {flag}")
            predicted_disease = None
            confidence = None
        else:
            predicted_disease = top_disease
            confidence = final_confidence
        
        return predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms

    def extract_symptoms_from_text(self, user_text, symptom_list):
        """
        Extract symptoms from free text using spaCy PhraseMatcher (English only).
        """
        nlp = spacy.load("en_core_web_sm")
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        # دعم كل من underscore والمسافة
        patterns = [nlp.make_doc(symptom.replace('_', ' ')) for symptom in symptom_list]
        matcher.add("SYMPTOMS", patterns)
        doc = nlp(user_text)
        found = set()
        for match_id, start, end in matcher(doc):
            # استخرج النص الأصلي من قائمة الأعراض (مع underscore)
            span_text = doc[start:end].text.lower().replace(' ', '_')
            for symptom in symptom_list:
                if span_text == symptom or span_text.replace('_', ' ') == symptom.replace('_', ' '):
                    found.add(symptom)
        return list(found)

    def start_chatbot(self):
        try:
            print("🤖 Starting Medically Enhanced Healthcare Chatbot...")
            name, location = self.getInfo()
            # After getting name and location, ask if user wants doctor or diagnosis
            while True:
                print("\nWhat would you like to do?")
                print("1) Diagnosis (disease prediction)")
                print("2) Find a doctor")
                print("3) Find a hospital")
                print("4) I am done / Exit")
                print("(Type 'undo' to go back and enter your location.)")
                main_choice = self.get_valid_input("Enter 1, 2, 3, or 4: ", valid_options=["1", "2", "3", "4", "undo"])
                if main_choice == 'undo':
                    print("Undo: Returning to location input.")
                    self.user_location = self.get_location()
                    continue
                elif main_choice == "1":
                    self.run_diagnosis_flow()
                    continue
                elif main_choice == "2":
                    while True:
                        specialty = input("Enter the medical specialty you are looking for (e.g., Cardiology, Neurology, Dermatology): ").strip()
                        if specialty.lower() == 'undo':
                            print("Undo: Returning to location input.")
                            self.user_location = self.get_location()
                            break
                        doctors = self.get_doctors_by_specialization(specialty)
                        location_doctors = [doc for doc in doctors if doc.get('location', '').strip().lower() == str(self.user_location).strip().lower()]
                        if location_doctors:
                            print(f"\nDoctors specializing in {specialty} in {self.user_location}:")
                            for doc in location_doctors:
                                print(f"- Name: {doc.get('name', 'Unknown')}")
                                print(f"  Specialty: {doc.get('specialization', 'N/A')}")
                                print(f"  Location: {doc.get('location', 'N/A')}")
                                print(f"  Phone: {doc.get('phone', 'N/A')}")
                                print(f"  Email: {doc.get('email', 'N/A')}")
                                print("---------------------------")
                        else:
                            print(f"No doctors found for specialty '{specialty}' in {self.user_location}.")
                            if doctors:
                                print(f"\nDoctors specializing in {specialty} in other locations:")
                                for doc in doctors:
                                    print(f"- Name: {doc.get('name', 'Unknown')}")
                                    print(f"  Specialty: {doc.get('specialization', 'N/A')}")
                                    print(f"  Location: {doc.get('location', 'N/A')}")
                                    print(f"  Phone: {doc.get('phone', 'N/A')}")
                                    print(f"  Email: {doc.get('email', 'N/A')}")
                                    print("---------------------------")
                            else:
                                print(f"No doctors found for specialty '{specialty}' in any location.")
                        again = self.get_valid_input("Do you want to search for another doctor? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                        if again == 'undo':
                            print("Undo: Returning to location input.")
                            self.user_location = self.get_location()
                            break
                        if again in ["yes", "y"]:
                            continue
                        else:
                            print("Returning to main menu...")
                            break
                    continue
                elif main_choice == "3":
                    while True:
                        hosp_location = input("Enter the city or country you are looking for hospitals in: ").strip()
                        if hosp_location.lower() == 'undo':
                            print("Undo: Returning to location input.")
                            self.user_location = self.get_location()
                            break
                        filtered_hospitals, all_hospitals = self.get_hospitals_by_location(hosp_location)
                        if filtered_hospitals:
                            print(f"\nHospitals in {hosp_location}:")
                            for hosp in filtered_hospitals:
                                print(f"- Name: {hosp.get('name', 'Unknown')}")
                                print(f"  Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}")
                                print(f"  Phone: {hosp.get('phone', 'N/A')}")
                                print(f"  Email: {hosp.get('email', 'N/A')}")
                                print(f"  Address: {hosp.get('address', 'N/A')}")
                                print(f"  Website: {hosp.get('website', 'N/A')}")
                                print("---------------------------")
                        else:
                            print(f"No hospitals found in {hosp_location}.")
                            if all_hospitals:
                                print(f"\nAvailable hospitals in other locations:")
                                for hosp in all_hospitals:
                                    print(f"- Name: {hosp.get('name', 'Unknown')}")
                                    print(f"  Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}")
                                    print(f"  Phone: {hosp.get('phone', 'N/A')}")
                                    print(f"  Email: {hosp.get('email', 'N/A')}")
                                    print(f"  Address: {hosp.get('address', 'N/A')}")
                                    print(f"  Website: {hosp.get('website', 'N/A')}")
                                    print("---------------------------")
                            else:
                                print(f"No hospitals found in any location.")
                        again = self.get_valid_input("Do you want to search for another hospital? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                        if again == 'undo':
                            print("Undo: Returning to location input.")
                            self.user_location = self.get_location()
                            break
                        if again in ["yes", "y"]:
                            continue
                        else:
                            print("Returning to main menu...")
                            break
                    continue
                elif main_choice == "4":
                    confirm_exit = self.get_valid_input("Are you sure you want to exit? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                    if confirm_exit == 'undo':
                        print("Undo: Returning to location input.")
                        self.user_location = self.get_location()
                        continue
                    if confirm_exit in ["yes", "y"]:
                        print("\nThank you for using the Healthcare Chatbot! Have a great day! 🙏")
                        return
                    else:
                        print("Returning to main menu...")
                        continue
            # Continue with the original chatbot flow (diagnosis)
            while True:
                print("Please choose input method:")
                print("1) Traditional (one symptom at a time)")
                print("2) Free text (write all symptoms in one sentence)")
                method = self.get_valid_input("Enter 1 or 2: ", valid_options=["1", "2"])
                if method == 'undo':
                    print("Undo: Returning to location input.")
                    self.user_location = self.get_location()
                    continue
                elif method == 'main_menu':
                    continue  # This will return to the main menu loop
                else:
                    break
            if method == "2":
                print("Please write all the symptoms you are experiencing in one sentence (e.g., I have headache and fever and muscle pain):")
                user_text = input("-> ")
                if user_text.strip().lower() == 'undo':
                    print("Undo: Returning to location input.")
                    self.user_location = self.get_location()
                    return
                # --- Tokenize and fuzzy-correct all user-entered symptoms ---
                leading_phrases = [
                    r'^i have ', r'^i am suffering from ', r'^i am having ', r'^i feel ', r'^i am ', r'^i\'m ', r'^i got ', r'^i\s+',
                    r'^my ', r'^having ', r'^suffering from ', r'^experiencing ', r'^with ', r'^and ', r'^, ', r'^\s+'
                ]
                tokens = re.split(r',| and | و | و|,|\band\b', user_text)
                cleaned_tokens = []
                for t in tokens:
                    t = t.strip().lower()
                    for phrase in leading_phrases:
                        t = re.sub(phrase, '', t)
                    t = t.strip()
                    if t:
                        t = t.replace(' ', '_')
                        cleaned_tokens.append(t)
                corrected = []
                for token in cleaned_tokens:
                    corrected.extend(self.handle_single_symptom_input(token))
                if not corrected:
                    print("No symptoms extracted. Switching to traditional mode.")
                    self.tree_to_code(self.model_trainer.clf, self.data_preprocessor.cols)
                    print("-" * 130)
                    print("Thank you for using Healthcare Chatbot! 🙏")
                    print("⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                    return
                # Ask for duration (with confirmation)
                while True:
                    num_days = self.get_valid_input('Okay. From how many days? ', input_type=int)
                    if num_days == 'main_menu':
                        return
                    print(f'You entered {num_days} days. Is this correct? (yes/y or no/n)')
                    confirm_days = input('-> ').strip().lower()
                    if confirm_days in ["yes", "y"]:
                        break
                    elif confirm_days in ["no", "n"]:
                        print('Please re-enter the number of days.')
                        continue
                    else:
                        print('Please enter yes/y or no/n.')
                        continue
                # Ask about relevant symptoms for each confirmed symptom (no duplicates)
                all_symptoms = set(corrected)
                related_answers = []
                i = 0
                while i < len(corrected):
                    symptom = corrected[i]
                    relevant = self.get_medical_relevant_symptoms(symptom)
                    if relevant:
                        print(f"Are you experiencing any of these symptoms related to {symptom}?")
                        rels = [rel for rel in relevant if rel not in all_symptoms]
                        j = 0
                        while j < len(rels):
                            rel = rels[j]
                            response = self.get_valid_input(f"   {rel}? (yes/y or no/n or back): ", valid_options=["yes", "y", "no", "n", "back"], symptom_mode=True)
                            if response == 'main_menu':
                                return
                            if response == "back":
                                if j > 0:
                                    j -= 1
                                    continue
                                else:
                                    print('Already at the first related symptom.')
                                    continue
                            elif response in ["yes", "y"]:
                                all_symptoms.add(rel)
                                related_answers.append((rel, "yes"))
                                j += 1
                            elif response in ["no", "n"]:
                                related_answers.append((rel, "no"))
                                j += 1
                            else:
                                print('Please enter yes/y, no/n, or back.')
                                continue
                    i += 1
                # Diagnose
                predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms))
                conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                _, _, _, present_symptoms = self.calculate_symptom_severity(list(all_symptoms))
                post_pred, post_conf, was_swapped = self.postprocess_prediction(predicted_disease, confidence, top_3_diseases, top_3_confidences, [s for s, _ in present_symptoms])
                if was_swapped:
                    predicted_disease = post_pred
                    confidence = post_conf
                    description = self.data_preprocessor.description_list.get(predicted_disease, "")
                    precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                    severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
                else:
                    description = self.data_preprocessor.description_list.get(predicted_disease, "")
                    precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                    severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)

                # --- NEW OUTPUT STYLE ---
                print(f"\n\U0001F4CB Current symptoms: {', '.join(list(all_symptoms))}")
                if confidence is None or confidence < 0.5:
                    print(f"\n\u26A0\uFE0F  I need at least 4 symptoms for a proper diagnosis. You have {len(list(all_symptoms))}.")
                    print("Let me ask about some common symptoms:")
                    for s in ['fever', 'fatigue', 'loss_of_appetite', 'weight_loss', 'chills']:
                        if s not in all_symptoms:
                            yn = self.get_valid_input(f"   {s.replace('_', ' ')}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                            if yn in ["yes", "y"]:
                                all_symptoms.add(s)
                    print(f"\n\U0001F4CA Current analysis: {len(list(all_symptoms))} symptoms, confidence: {conf_str}")
                    print("To improve accuracy, please tell me about any other symptoms you're experiencing:")
                    extra = input("Additional symptom 1 (or press Enter to skip): ").strip().replace(' ', '_')
                    if extra:
                        corrected_extra = self.handle_single_symptom_input(extra)
                        for ce in corrected_extra:
                            if ce not in all_symptoms:
                                all_symptoms.add(ce)
                    # Re-run prediction with updated symptoms
                    predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms))
                    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                if predicted_disease:
                    print(f"\n\U0001FA7A You may have: {predicted_disease}")
                    print(f"\U0001F52C Model confidence: {conf_str}")
                    if confidence is not None and confidence < 0.5:
                        print("\U0001F534 Low confidence - please consult a healthcare professional immediately")
                else:
                    print(f"\n\U0001F534 Unable to make a confident prediction. Please consult a healthcare professional.")
                desc_to_show = None
                if isinstance(description, list):
                    seen = set()
                    for desc in description:
                        if desc and desc not in seen:
                            desc_to_show = desc
                            break
                elif isinstance(description, str) and description.strip():
                    desc_to_show = description.strip()
                if desc_to_show:
                    print(f"\n\U0001F4DD Description: {desc_to_show}")
                else:
                    print("\n\U0001F4DD Description: No description available for this condition.")
                if precautions:
                    print("\n\U0001F4A1 Take the following precautions:")
                    for i, precaution in enumerate(precautions):
                        if precaution.strip():
                            print(f"   {i+1}) {precaution}")
                # Doctor recommendations (always show in both modes)
                self._provide_doctor_recommendations(predicted_disease)
                print("-" * 130)
                print("Thank you for using Healthcare Chatbot! \U0001F64F")
                print("\u26A0\uFE0F  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                return  # Return to main menu after diagnosis
            self.tree_to_code(self.model_trainer.clf, self.data_preprocessor.cols)
            print("-" * 130)
            print("Thank you for using Healthcare Chatbot! 🙏")
            print("⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
        except Exception as e:
            print("❌ Chatbot error:")
            traceback.print_exc()
            print("Please restart the application.")

    def tree_to_code(self, tree, feature_names, confirmed_symptoms=None):
        """
        Main chatbot interaction function - ENHANCED VERSION (English, robust undo)
        """
        try:
            if confirmed_symptoms is None:
                while True:
                    # Step 1: Symptom entry
                    confirmed_symptoms = []
                    restart_symptom_entry = False
                    while True:
                        self.text_to_speech("Enter the symptom you are experiencing")
                        print("\nEnter the symptom you are experiencing  \t\t", end="->")
                        symptom_input = self.get_valid_input("", symptom_mode=True)
                        if symptom_input == 'undo':
                            print("Undo: Please enter the symptom again.")
                            continue
                        if not symptom_input:
                            print("Please enter a symptom.")
                            continue
                        
                        # --- Use the exact same logic as free text mode ---
                        corrected = self.handle_single_symptom_input(symptom_input)
                        if len(corrected) > 0:
                            confirmed_symptoms.extend(corrected)
                            break
                        else:
                            print("Symptom not recognized. Please try again.")

                    # Step 1.5: Symptom review/edit
                    while True:
                        print(f"\nHere are the symptoms I have: {', '.join(confirmed_symptoms)}")
                        confirm = input("Would you like to add, remove, or edit any symptoms? (yes/y or no/n)\n-> ").strip().lower()
                        if confirm == "yes" or confirm == "y":
                            print("Enter the full, final list of symptoms separated by commas (or type 'undo' to go back and re-enter symptoms):")
                            final = input("-> ").strip().lower()
                            if final == 'undo':
                                print("Undo: Returning to symptom entry step.")
                                restart_symptom_entry = True
                                break
                            raw_symptoms = [s.strip() for s in final.split(',') if s.strip()]
                            validated = []
                            for token in raw_symptoms:
                                # Use the centralized symptom input handling
                                corrected_tokens = self.handle_single_symptom_input(token)
                                validated.extend(corrected_tokens)
                            if validated:
                                confirmed_symptoms = validated
                            else:
                                print("No valid symptoms entered. Please try again.")
                                continue
                        else:
                            break
                    if restart_symptom_entry:
                        continue

                    # Step 2: Days input (with confirmation)
                    while True:
                        num_days = self.get_valid_input("Okay. For how many days have you had these symptoms? : ", input_type=int)
                        print(f'You entered {num_days} days. Is this correct? (yes/y or no/n)')
                        confirm_days = input('-> ').strip().lower()
                        if confirm_days in ["yes", "y"]:
                            break
                        elif confirm_days in ["no", "n"]:
                            print('Please re-enter the number of days.')
                            continue
                        else:
                            print('Please enter yes/y or no/n.')
                            continue
                    if num_days == 'undo':
                        continue  # Go back to symptom review

                    # Step 3: Yes/No follow-up questions (with back command)
                    relevant_symptoms = self.get_medical_relevant_symptoms(confirmed_symptoms[0])
                    if relevant_symptoms:
                        self.text_to_speech("Are you experiencing any of these related symptoms?")
                        if not self.production_mode:
                            print("🔍 Are you experiencing any of these medically related symptoms?")
                        rels = [symptom for symptom in relevant_symptoms if symptom not in confirmed_symptoms]
                        j = 0
                        while j < len(rels):
                            symptom = rels[j]
                            self.text_to_speech(f"{symptom}, are you experiencing it?")
                            response = self.get_valid_input(f"   {symptom}? (yes/y or no/n or back): ", valid_options=["yes", "y", "no", "n", "back"], symptom_mode=True)
                            if response == 'back':
                                if j > 0:
                                    j -= 1
                                    continue
                                else:
                                    print('Already at the first related symptom.')
                                    continue
                            elif response in ["yes", "y"]:
                                confirmed_symptoms.append(symptom)
                                j += 1
                            elif response in ["no", "n"]:
                                j += 1
                            else:
                                print('Please enter yes/y, no/n, or back.')
                                continue

                    # Step 4: Make prediction with medical validation
                    while True:
                        predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(confirmed_symptoms)
                        conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                        _, _, _, present_symptoms = self.calculate_symptom_severity(confirmed_symptoms)
                        post_pred, post_conf, was_swapped = self.postprocess_prediction(predicted_disease, confidence, top_3_diseases, top_3_confidences, [s for s, _ in present_symptoms])
                        if was_swapped:
                            predicted_disease = post_pred
                            confidence = post_conf
                            description = self.data_preprocessor.description_list.get(predicted_disease, "")
                            precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                            severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
                        else:
                            description = self.data_preprocessor.description_list.get(predicted_disease, "")
                            precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                            severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)

                        print(f"\n\U0001F4CB Current symptoms: {', '.join(confirmed_symptoms)}")
                        if confidence is None or confidence < 0.5 or not predicted_disease or predicted_disease.lower() == 'none':
                            print(f"\n\u26A0\uFE0F  I need at least 4 symptoms for a proper diagnosis. You have {len(confirmed_symptoms)}.")
                            print("Let me ask about some common symptoms:")
                            for s in ['fever', 'fatigue', 'loss_of_appetite', 'weight_loss', 'chills']:
                                if s not in confirmed_symptoms:
                                    yn = self.get_valid_input(f"   {s.replace('_', ' ')}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                                    if yn in ["yes", "y"]:
                                        confirmed_symptoms.append(s)
                            print(f"\n\U0001F4CA Current analysis: {len(confirmed_symptoms)} symptoms, confidence: {conf_str}")
                            print("To improve accuracy, please tell me about any other symptoms you're experiencing:")
                            extra = input("Additional symptom 1 (or press Enter to skip): ").strip().replace(' ', '_')
                            if extra:
                                corrected_extra = self.handle_single_symptom_input(extra)
                                for ce in corrected_extra:
                                    if ce not in confirmed_symptoms:
                                        confirmed_symptoms.append(ce)
                            continue  # Re-run prediction with updated symptoms
                        if predicted_disease:
                            print(f"\n\U0001FA7A You may have: {predicted_disease}")
                            print(f"\U0001F52C Model confidence: {conf_str}")
                            if confidence is not None and confidence < 0.5:
                                print("\U0001F534 Low confidence - please consult a healthcare professional immediately")
                        else:
                            print(f"\n\U0001F534 Unable to make a confident prediction. Please consult a healthcare professional.")
                        desc_to_show = None
                        if isinstance(description, list):
                            seen = set()
                            for desc in description:
                                if desc and desc not in seen:
                                    desc_to_show = desc
                                    break
                        elif isinstance(description, str) and description.strip():
                            desc_to_show = description.strip()
                        if desc_to_show:
                            print(f"\n\U0001F4DD Description: {desc_to_show}")
                        else:
                            print("\n\U0001F4DD Description: No description available for this condition.")
                        if precautions:
                            print("\n\U0001F4A1 Take the following precautions:")
                            for i, precaution in enumerate(precautions):
                                if precaution.strip():
                                    print(f"   {i+1}) {precaution}")
                        self._provide_doctor_recommendations(predicted_disease)
                        print("-" * 130)
                        print("Thank you for using Healthcare Chatbot! \U0001F64F")
                        print("\u26A0\uFE0F  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                        return  # Return to main menu after diagnosis

        except Exception as e:
            print("❌ An error occurred during diagnosis:")
            traceback.print_exc()
            print("Please try again or consult a healthcare professional.")

    def _provide_doctor_recommendations(self, predicted_disease):
        """Provide doctor recommendations based on predicted disease"""
        try:
            specialization = self.disease_to_specialty.get(predicted_disease, None)
            if not specialization:
                print(f"⚠️  No medical specialty found for {predicted_disease}")
                return
                
            doctors = self.get_doctors_by_specialization(specialization)
            if not doctors:
                print(f"⚠️  No doctors found for specialization: {specialization}")
                return
                
            # Filter by location if available
            filtered_doctors = []
            if self.user_location:
                location_lower = self.user_location.lower().strip()
                for doc in doctors:
                    doc_city = str(doc.get('city', '')).lower().strip()
                    doc_country = str(doc.get('country', '')).lower().strip()
                    doc_location = f"{doc_city} {doc_country}".strip()
                    if (location_lower in doc_location or 
                        doc_city in location_lower or 
                        doc_country in location_lower):
                        filtered_doctors.append(doc)
                        
            if filtered_doctors:
                print(f"\n🩺 Recommended Doctors for {specialization} in {self.user_location}:")
                for doc in filtered_doctors[:3]:  # Show top 3
                    print(f"- Name: {doc.get('name')}")
                    print(f"  Location: {doc.get('city')}, {doc.get('country')}")
                    print(f"  Phone: {doc.get('phone', 'Not available')}")
                    print(f"  Profile Link: {doc.get('ImgUrl', 'Not available')}")
                    print("----------------------")
            elif doctors:
                print(f"\n🩺 No doctors found in {self.user_location}. Showing top doctors for {specialization}:")
                for doc in doctors[:3]:  # Show top 3
                    print(f"- Name: {doc.get('name')}")
                    print(f"  Location: {doc.get('city')}, {doc.get('country')}")
                    print(f"  Phone: {doc.get('phone', 'Not available')}")
                    print(f"  Profile Link: {doc.get('ImgUrl', 'Not available')}")
                    print("----------------------")
                        
        except Exception as e:
            print(f"⚠️  Error getting doctor recommendations: {e}")

    def get_doctors_by_specialization(self, specialization):
        """Fetch doctors from API by specialization (improved matching)"""
        url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            doctors = data.get('data', [])
            
            # Improved specialization matching
            filtered = []
            specialization_lower = specialization.lower().strip()
            
            for doc in doctors:
                doc_specialization = doc.get('specialization', '').lower().strip()
                # Check for exact match or substring match
                if (specialization_lower == doc_specialization or 
                    specialization_lower in doc_specialization or 
                    doc_specialization in specialization_lower):
                    filtered.append(doc)
            
            return filtered
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not fetch doctors from API: {e}")
            return []
        except Exception as e:
            print(f"⚠️  An error occurred while processing doctor data: {e}")
            return []

    def get_medical_specialty_for_disease(self, disease_name):
        """Get the medical specialty for a given disease from the dataset"""
        try:
            # Find the disease in the reduced_data and get its medical specialty
            if disease_name in self.data_preprocessor.reduced_data.index:
                medical_specialty = self.data_preprocessor.reduced_data.loc[disease_name, 'Medical Specialties']
                return medical_specialty
            else:
                # Fallback: try to find a similar disease
                for disease in self.data_preprocessor.reduced_data.index:
                    if disease_name.lower() in disease.lower() or disease.lower() in disease_name.lower():
                        medical_specialty = self.data_preprocessor.reduced_data.loc[disease, 'Medical Specialties']
                        return medical_specialty
                return None
        except Exception as e:
            print(f"⚠️  Error getting medical specialty: {e}")
            return None

    def calculate_symptom_severity(self, confirmed_symptoms):
        """
        Calculate overall severity score based on present symptoms
        Returns: severity_score, severity_level, high_severity_symptoms
        """
        if not hasattr(self.data_preprocessor, 'severityDictionary'):
            return 0, "Unknown", []
        
        total_severity = 0
        high_severity_symptoms = []
        present_symptoms = []
        
        for symptom in confirmed_symptoms:
            if symptom in self.data_preprocessor.severityDictionary:
                severity_value = self.data_preprocessor.severityDictionary[symptom]
                total_severity += severity_value
                present_symptoms.append((symptom, severity_value))
                
                # Identify high severity symptoms (severity >= 6)
                if severity_value >= 6:
                    high_severity_symptoms.append((symptom, severity_value))
        
        # Determine severity level
        if total_severity >= 20:
            severity_level = "Critical"
        elif total_severity >= 15:
            severity_level = "High"
        elif total_severity >= 10:
            severity_level = "Moderate"
        elif total_severity >= 5:
            severity_level = "Mild"
        else:
            severity_level = "Very Mild"
        
        return total_severity, severity_level, high_severity_symptoms, present_symptoms

    def adjust_confidence_by_severity(self, base_confidence, severity_score, high_severity_symptoms):
        """
        Adjust prediction confidence based on symptom severity
        High severity symptoms can increase confidence for certain diseases
        """
        adjusted_confidence = base_confidence
        
        # If there are high severity symptoms, adjust confidence
        if high_severity_symptoms:
            # High severity symptoms often indicate more serious conditions
            # This can increase confidence for certain disease predictions
            severity_bonus = min(len(high_severity_symptoms) * 0.05, 0.15)  # Max 15% bonus
            adjusted_confidence += severity_bonus
        
        # Cap confidence at 1.0
        return min(adjusted_confidence, 1.0)

    def get_severity_based_recommendations(self, severity_level, high_severity_symptoms, predicted_disease):
        """
        Generate severity-based medical recommendations
        """
        recommendations = []
        
        if severity_level == "Critical":
            recommendations.append("🚨 URGENT: Seek immediate medical attention!")
            recommendations.append("Call emergency services or visit the nearest hospital")
            recommendations.append("Do not delay treatment")
        elif severity_level == "High":
            recommendations.append("⚠️  HIGH PRIORITY: Consult a doctor within 24 hours")
            recommendations.append("Monitor symptoms closely")
            if high_severity_symptoms:
                recommendations.append(f"High severity symptoms detected: {', '.join([s[0] for s in high_severity_symptoms])}")
        elif severity_level == "Moderate":
            recommendations.append("📋 MODERATE: Schedule a doctor appointment soon")
            recommendations.append("Continue monitoring symptoms")
        elif severity_level == "Mild":
            recommendations.append("✅ MILD: Monitor symptoms and consult if they worsen")
        else:
            recommendations.append("✅ VERY MILD: Continue monitoring")
        
        # Add specific recommendations based on high severity symptoms
        for symptom, severity in high_severity_symptoms:
            if symptom == "chest_pain" and severity >= 7:
                recommendations.append("🫀 Chest pain requires immediate evaluation - could indicate heart issues")
            elif symptom == "coma" and severity >= 7:
                recommendations.append("😵 Coma is a medical emergency - seek immediate help")
            elif symptom == "blood_in_sputum" and severity >= 5:
                recommendations.append("🩸 Blood in sputum requires urgent medical evaluation")
            elif symptom == "paralysis" and severity >= 7:
                recommendations.append("🦽 Paralysis requires immediate neurological evaluation")
        
        return recommendations

    def display_debug_output(self, predicted_disease, confidence, severity_score, severity_level, high_severity_symptoms, present_symptoms, severity_recommendations):
        """Display detailed debug output for diagnosis"""
        print("\n===== DEBUG DIAGNOSIS OUTPUT =====")
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence:.1%}" if confidence is not None else "Confidence: Unknown")
        print(f"Severity Score: {severity_score}")
        print(f"Severity Level: {severity_level}")
        if high_severity_symptoms:
            print("High Severity Symptoms:")
            for symptom, score in high_severity_symptoms:
                print(f"  - {symptom} (severity: {score})")
        else:
            print("High Severity Symptoms: None")
        if present_symptoms:
            print("Present Symptoms (with severity):")
            for symptom, score in present_symptoms:
                print(f"  - {symptom} (severity: {score})")
        else:
            print("Present Symptoms: None")
        if severity_recommendations:
            print("\nMedical Recommendations:")
            for rec in severity_recommendations:
                print(f"  - {rec}")
        print("===================================\n")

    def postprocess_prediction(self, predicted_disease, confidence, top3, top3_confidences, present_symptoms):
        """Rule-based post-processing to avoid severe diagnoses for mild/common symptoms"""
        # Define mild/common symptoms and diseases
        mild_symptoms = set([
            'runny_nose', 'throat_irritation', 'headache', 'sneezing', 'fatigue', 'cough', 'chills', 'malaise',
            'muscle_pain', 'body_ache', 'nasal_congestion', 'sore_throat', 'watery_eyes', 'mild_fever', 'congestion'
        ])
        mild_diseases = set(['Common Cold', 'Allergy', 'Migraine'])
        severe_diseases = set(['Paralysis (brain hemorrhage)', 'Heart attack', 'AIDS', 'Tuberculosis', 'Malaria', 'Dengue', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis'])
        
        # If all present symptoms are mild/common
        if all(s in mild_symptoms for s in present_symptoms):
            # If top prediction is severe, confidence is low, and a mild disease is in top-3
            if predicted_disease in severe_diseases and confidence < 0.5:
                for d, conf in zip(top3, top3_confidences):
                    if d in mild_diseases:
                        # Suggest the mild disease instead
                        return d, conf, True
        return predicted_disease, confidence, False

    # Add a helper function for diagnosis flow
    def run_diagnosis_flow(self):
        while True:
            print("Please choose input method:")
            print("1) Traditional (one symptom at a time)")
            print("2) Free text (write all symptoms in one sentence)")
            method = self.get_valid_input("Enter 1 or 2: ", valid_options=["1", "2"])
            if method == 'undo':
                print("Undo: Returning to location input.")
                self.user_location = self.get_location()
                continue
            elif method == 'main_menu':
                return  # Return to main menu
            else:
                break
        if method == "2":
            print("Please write all the symptoms you are experiencing in one sentence (e.g., I have headache and fever and muscle pain):")
            user_text = input("-> ")
            if user_text.strip().lower() == 'undo':
                print("Undo: Returning to location input.")
                self.user_location = self.get_location()
                return
            # --- Tokenize and fuzzy-correct all user-entered symptoms ---
            leading_phrases = [
                r'^i have ', r'^i am suffering from ', r'^i am having ', r'^i feel ', r'^i am ', r'^i\'m ', r'^i got ', r'^i\s+',
                r'^my ', r'^having ', r'^suffering from ', r'^experiencing ', r'^with ', r'^and ', r'^, ', r'^\s+'
            ]
            tokens = re.split(r',| and | و | و|,|\band\b', user_text)
            cleaned_tokens = []
            for t in tokens:
                t = t.strip().lower()
                for phrase in leading_phrases:
                    t = re.sub(phrase, '', t)
                t = t.strip()
                if t:
                    t = t.replace(' ', '_')
                    cleaned_tokens.append(t)
            corrected = []
            for token in cleaned_tokens:
                corrected.extend(self.handle_single_symptom_input(token))
            if not corrected:
                print("No symptoms extracted. Switching to traditional mode.")
                self.tree_to_code(self.model_trainer.clf, self.data_preprocessor.cols)
                print("-" * 130)
                print("Thank you for using Healthcare Chatbot! 🙏")
                print("⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                return
            # Ask for duration (with confirmation)
            while True:
                num_days = self.get_valid_input('Okay. From how many days? ', input_type=int)
                if num_days == 'main_menu':
                    return
                print(f'You entered {num_days} days. Is this correct? (yes/y or no/n)')
                confirm_days = input('-> ').strip().lower()
                if confirm_days in ["yes", "y"]:
                    break
                elif confirm_days in ["no", "n"]:
                    print('Please re-enter the number of days.')
                    continue
                else:
                    print('Please enter yes/y or no/n.')
                    continue
            # Ask about relevant symptoms for each confirmed symptom (no duplicates)
            all_symptoms = set(corrected)
            related_answers = []
            i = 0
            while i < len(corrected):
                symptom = corrected[i]
                relevant = self.get_medical_relevant_symptoms(symptom)
                if relevant:
                    print(f"Are you experiencing any of these symptoms related to {symptom}?")
                    rels = [rel for rel in relevant if rel not in all_symptoms]
                    j = 0
                    while j < len(rels):
                        rel = rels[j]
                        response = self.get_valid_input(f"   {rel}? (yes/y or no/n or back): ", valid_options=["yes", "y", "no", "n", "back"], symptom_mode=True)
                        if response == 'main_menu':
                            return
                        if response == "back":
                            if j > 0:
                                j -= 1
                                continue
                            else:
                                print('Already at the first related symptom.')
                                continue
                        elif response in ["yes", "y"]:
                            all_symptoms.add(rel)
                            related_answers.append((rel, "yes"))
                            j += 1
                        elif response in ["no", "n"]:
                            related_answers.append((rel, "no"))
                            j += 1
                        else:
                            print('Please enter yes/y, no/n, or back.')
                            continue
                i += 1
            # Diagnose
            predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms))
            conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
            _, _, _, present_symptoms = self.calculate_symptom_severity(list(all_symptoms))
            post_pred, post_conf, was_swapped = self.postprocess_prediction(predicted_disease, confidence, top_3_diseases, top_3_confidences, [s for s, _ in present_symptoms])
            if was_swapped:
                predicted_disease = post_pred
                confidence = post_conf
                description = self.data_preprocessor.description_list.get(predicted_disease, "")
                precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
            else:
                description = self.data_preprocessor.description_list.get(predicted_disease, "")
                precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)

            # --- NEW OUTPUT STYLE ---
            print(f"\n\U0001F4CB Current symptoms: {', '.join(list(all_symptoms))}")
            if confidence is None or confidence < 0.5:
                print(f"\n\u26A0\uFE0F  I need at least 4 symptoms for a proper diagnosis. You have {len(list(all_symptoms))}.")
                print("Let me ask about some common symptoms:")
                for s in ['fever', 'fatigue', 'loss_of_appetite', 'weight_loss', 'chills']:
                    if s not in all_symptoms:
                        yn = self.get_valid_input(f"   {s.replace('_', ' ')}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                        if yn in ["yes", "y"]:
                            all_symptoms.add(s)
                    print(f"\n\U0001F4CA Current analysis: {len(list(all_symptoms))} symptoms, confidence: {conf_str}")
                    print("To improve accuracy, please tell me about any other symptoms you're experiencing:")
                    extra = input("Additional symptom 1 (or press Enter to skip): ").strip().replace(' ', '_')
                    if extra:
                        corrected_extra = self.handle_single_symptom_input(extra)
                        for ce in corrected_extra:
                            if ce not in all_symptoms:
                                all_symptoms.add(ce)
                    # Re-run prediction with updated symptoms
                    predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms))
                    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                if predicted_disease:
                    print(f"\n\U0001FA7A You may have: {predicted_disease}")
                    print(f"\U0001F52C Model confidence: {conf_str}")
                    if confidence is not None and confidence < 0.5:
                        print("\U0001F534 Low confidence - please consult a healthcare professional immediately")
                else:
                    print(f"\n\U0001F534 Unable to make a confident prediction. Please consult a healthcare professional.")
                desc_to_show = None
                if isinstance(description, list):
                    seen = set()
                    for desc in description:
                        if desc and desc not in seen:
                            desc_to_show = desc
                            break
                elif isinstance(description, str) and description.strip():
                    desc_to_show = description.strip()
                if desc_to_show:
                    print(f"\n\U0001F4DD Description: {desc_to_show}")
                else:
                    print("\n\U0001F4DD Description: No description available for this condition.")
                if precautions:
                    print("\n\U0001F4A1 Take the following precautions:")
                    for i, precaution in enumerate(precautions):
                        if precaution.strip():
                            print(f"   {i+1}) {precaution}")
                # Doctor recommendations (always show in both modes)
                self._provide_doctor_recommendations(predicted_disease)
                print("-" * 130)
                print("Thank you for using Healthcare Chatbot! \U0001F64F")
                print("\u26A0\uFE0F  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                return  # Return to main menu after diagnosis
            self.tree_to_code(self.model_trainer.clf, self.data_preprocessor.cols)
            print("-" * 130)

    def get_hospitals_by_location(self, location):
        """Fetch hospitals from API by location (city or country)"""
        url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            hospitals = data.get('data', [])
            location_lower = location.lower().strip()
            filtered = []
            for hosp in hospitals:
                hosp_city = str(hosp.get('city', '')).lower().strip()
                hosp_country = str(hosp.get('country', '')).lower().strip()
                hosp_location = f"{hosp_city} {hosp_country}".strip()
                if (location_lower in hosp_location or 
                    hosp_city in location_lower or 
                    hosp_country in location_lower):
                    filtered.append(hosp)
            return filtered, hospitals
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not fetch hospitals from API: {e}")
            return [], []
        except Exception as e:
            print(f"⚠️  An error occurred while processing hospital data: {e}")
            return [], []


