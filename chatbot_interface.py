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
            regexp = re.compile(f"{inp}", re.IGNORECASE)
            pred_list = [item for item in dis_list if regexp.search(item.lower())]
            if len(pred_list) > MAX_SYMPTOM_SEARCH_RESULTS:
                pred_list = pred_list[:MAX_SYMPTOM_SEARCH_RESULTS]
            return (1, pred_list) if pred_list else (0, [])
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
            if input_type == int:
                try:
                    return int(user_input)
                except ValueError:
                    print("Please enter a valid number.")
                    continue
            elif valid_options:
                if user_input in valid_options:
                    return user_input
                else:
                    print(f"Please enter one of: {', '.join(valid_options)}")
                    continue
            elif symptom_mode:
                # Generalized special handling for generic terms
                similar_options = [s for s in self.feature_names if user_input in s]
                if len(similar_options) > 3:
                    print(f"You entered '{user_input}'. Please specify the type(s):")
                    for i, opt in enumerate(similar_options):
                        print(f"  {i+1}) {opt}")
                    print(f"  0) None of these / skip")
                    selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,3,5): ").strip()
                    indices = [int(x) for x in selected.split(',') if x.strip().isdigit()]
                    for idx in indices:
                        if 1 <= idx <= len(similar_options):
                            if similar_options[idx-1] not in corrected:
                                corrected.append(similar_options[idx-1])
                    continue
                if user_input == 'fever':
                    fever_options = [s for s in self.feature_names if 'fever' in s]
                    print("You entered 'fever'. Please specify the type(s) of fever:")
                    for i, opt in enumerate(fever_options):
                        print(f"  {i+1}) {opt}")
                    print(f"  0) None of these / skip")
                    selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,2): ").strip()
                    indices = [int(x) for x in selected.split(',') if x.strip().isdigit()]
                    for idx in indices:
                        if 1 <= idx <= len(fever_options):
                            if fever_options[idx-1] not in corrected:
                                corrected.append(fever_options[idx-1])
                    continue
                # Fuzzy match for symptoms
                if user_input in self.feature_names:
                    return user_input
                closest = self.get_closest_symptom(user_input)
                if closest:
                    # Calculate similarity ratio
                    ratio = SequenceMatcher(None, user_input, closest).ratio()
                    if ratio > 0.8:
                        print(f"Interpreting '{user_input}' as '{closest}'.")
                        return closest
                    elif ratio > 0.6:
                        confirm = input(f"Did you mean '{closest}'? (y/n): ").strip().lower()
                        if confirm == 'y':
                            return closest
                        else:
                            print("Please try again.")
                            continue
                    else:
                        print("Symptom not recognized. Please try again.")
                        continue
                else:
                    print("Symptom not recognized. Please try again.")
                    continue
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
                print("3) I am done / Exit")
                main_choice = self.get_valid_input("Enter 1, 2, or 3: ", valid_options=["1", "2", "3", "undo"])
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
                        again = self.get_valid_input("Do you want to search for another doctor? (y/n): ", valid_options=["y", "n"])
                        if again == 'undo':
                            print("Undo: Returning to location input.")
                            self.user_location = self.get_location()
                            break
                        if again == "y":
                            continue
                        else:
                            print("Returning to main menu...")
                            break
                    continue
                elif main_choice == "3":
                    confirm_exit = self.get_valid_input("Are you sure you want to exit? (y/n): ", valid_options=["y", "n"])
                    if confirm_exit == 'undo':
                        print("Undo: Returning to location input.")
                        self.user_location = self.get_location()
                        continue
                    if confirm_exit == "y":
                        print("\nThank you for using the Healthcare Chatbot! Have a great day! 🙏")
                        return
                    else:
                        print("Returning to main menu...")
                        continue
            # Continue with the original chatbot flow (diagnosis)
            while True:
                print("Please choose input method:")
                print('Type "undo" to go back and change your location.')
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
                    # Generalized special handling for generic terms
                    similar_options = [s for s in self.feature_names if token in s]
                    if len(similar_options) > 3:
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
                    continue
                    if token == 'fever':
                        fever_options = [s for s in self.feature_names if 'fever' in s]
                        print("You entered 'fever'. Please specify the type(s) of fever:")
                        for i, opt in enumerate(fever_options):
                            print(f"  {i+1}) {opt}")
                        print(f"  0) None of these / skip")
                        selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,2): ").strip()
                        indices = [int(x) for x in selected.split(',') if x.strip().isdigit()]
                        for idx in indices:
                            if 1 <= idx <= len(fever_options):
                                if fever_options[idx-1] not in corrected:
                                    corrected.append(fever_options[idx-1])
                    continue
                    if token in self.feature_names:
                        if token not in corrected:
                            corrected.append(token)
                    else:
                        matches = difflib.get_close_matches(token, self.feature_names, n=3, cutoff=0.0)
                        if matches:
                            best = matches[0]
                            ratio = SequenceMatcher(None, token, best).ratio()
                            if ratio > 0.8:
                                print(f"Interpreting '{token}' as '{best}'.")
                                if best not in corrected:
                                    corrected.append(best)
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
                        else:
                            print(f"No good match found for '{token}'. Skipping.")
                extracted = corrected
                if not extracted:
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
                    print(f'You entered {num_days} days. Is this correct? (y/n)')
                    confirm_days = input('-> ').strip().lower()
                    if confirm_days == 'y':
                        break
                    elif confirm_days == 'n':
                        print('Please re-enter the number of days.')
                        continue
                    else:
                        print('Please enter y or n.')
                        continue
                # Ask about relevant symptoms for each confirmed symptom (no duplicates)
                all_symptoms = set(extracted)
                related_answers = []
                i = 0
                while i < len(extracted):
                    symptom = extracted[i]
                    relevant = self.get_medical_relevant_symptoms(symptom)
                    if relevant:
                        print(f"Are you experiencing any of these symptoms related to {symptom}?")
                        rels = [rel for rel in relevant if rel not in all_symptoms]
                        j = 0
                        while j < len(rels):
                            rel = rels[j]
                            response = self.get_valid_input(f"   {rel}? (yes/no/back): ", valid_options=["yes", "no", "back"], symptom_mode=True)
                            if response == 'main_menu':
                                return
                            if response == "back":
                                if j > 0:
                                    j -= 1
                                    continue
                                else:
                                    print('Already at the first related symptom.')
                                    continue
                            elif response == "yes":
                                all_symptoms.add(rel)
                                related_answers.append((rel, "yes"))
                                j += 1
                            elif response == "no":
                                related_answers.append((rel, "no"))
                                j += 1
                            else:
                                print('Please enter yes, no, or back.')
                                continue
                    i += 1
                # Diagnose
                predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms))
                conf_str = f"{confidence:.1%}" if confidence is not None else "Unknown"
                # Calculate severity details for display
                _, _, _, present_symptoms = self.calculate_symptom_severity(list(all_symptoms))
                # After model prediction, apply post-processing (only once, before any output)
                post_pred, post_conf, was_swapped = self.postprocess_prediction(predicted_disease, confidence, top_3_diseases, top_3_confidences, [s for s, _ in present_symptoms])
                if was_swapped:
                    print("\n[INFO] Based on your symptoms, a mild/common condition is more likely. Overriding model prediction.")
                    predicted_disease = post_pred
                    confidence = post_conf
                    # Update description and precautions for the new disease
                    description = self.data_preprocessor.description_list.get(predicted_disease, "")
                    precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                    severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
                else:
                    description = self.data_preprocessor.description_list.get(predicted_disease, "")
                    precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                    severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
                # Always display detailed debug output for clarity
                self.display_debug_output(predicted_disease, confidence, severity_score, severity_level,
                                        high_severity_symptoms, present_symptoms, severity_recommendations)
                # Show top-3 diseases if confidence is low
                if confidence is not None and confidence < 0.6 and top_3_diseases is not None:
                    print("\nOther possible conditions:")
                    for i, (disease, conf) in enumerate(zip(top_3_diseases, top_3_confidences)):
                        print(f"  {i+1}) {disease} (confidence: {conf:.1%})")
                # Show description and precautions in debug mode
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
                    print(f"\n📝 Description: {desc_to_show}")
                else:
                    print("\n📝 Description: No description available for this condition.")
                if precautions:
                    print("\n💡 Take the following precautions:")
                    for i, precaution in enumerate(precautions):
                        if precaution.strip():
                            print(f"   {i+1}) {precaution}")
                # Doctor recommendations (always show in both modes)
                self._provide_doctor_recommendations(predicted_disease)
                print("-" * 130)
                print("Thank you for using Healthcare Chatbot! 🙏")
                print("⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                return
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
                        conf, matched_symptoms = self.check_pattern(self.feature_names, symptom_input)
                        if conf == 1:
                            print("🔍 Searches related to input: ")
                            for num, symptom in enumerate(matched_symptoms):
                                print(f"{num}) {symptom}")
                            if len(matched_symptoms) > 1:
                                selection = self.get_valid_input(
                                    f"Select the one you meant (0 - {len(matched_symptoms)-1}): ", input_type=int
                                )
                                if 0 <= selection < len(matched_symptoms):
                                    confirmed_symptoms.append(matched_symptoms[selection])
                                    break
                                else:
                                    print("Invalid selection. Please try again.")
                            else:
                                confirmed_symptoms.append(matched_symptoms[0])
                                break
                        else:
                            print("❌ Enter valid symptom. Please try again.")

                    # Step 1.5: Symptom review/edit
                    while True:
                        print(f"\nHere are the symptoms I have: {', '.join(confirmed_symptoms)}")
                        confirm = input("Would you like to add, remove, or edit any symptoms? (y/n)\n-> ").strip().lower()
                        if confirm == "y":
                            print("Enter the full, final list of symptoms separated by commas (or type 'undo' to go back and re-enter symptoms):")
                            final = input("-> ").strip().lower()
                            if final == 'undo':
                                print("Undo: Returning to symptom entry step.")
                                restart_symptom_entry = True
                                break
                            raw_symptoms = [s.strip().replace(' ', '_') for s in final.split(',') if s.strip()]
                            validated = []
                            for token in raw_symptoms:
                                if token in self.feature_names:
                                    validated.append(token)
                                else:
                                    matches = difflib.get_close_matches(token, self.feature_names, n=3, cutoff=0.0)
                                    if matches:
                                        best = matches[0]
                                        ratio = SequenceMatcher(None, token, best).ratio()
                                        if ratio > 0.8:
                                            print(f"Interpreting '{token}' as '{best}'.")
                                            validated.append(best)
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
                                                        if matches[idx-1] not in validated:
                                                            validated.append(matches[idx-1])
                                                        break
                                            print("Invalid choice. Please try again.")
                                        else:
                                            print(f"No good match found for '{token}'. Skipping.")
                                    else:
                                        print(f"No close match found for '{token}'. Skipping.")
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
                        print(f'You entered {num_days} days. Is this correct? (y/n)')
                        confirm_days = input('-> ').strip().lower()
                        if confirm_days == 'y':
                            break
                        elif confirm_days == 'n':
                            print('Please re-enter the number of days.')
                            continue
                        else:
                            print('Please enter y or n.')
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
                            response = self.get_valid_input(f"   {symptom}? (yes/no/back): ", valid_options=["yes", "no", "back"], symptom_mode=True)
                            if response == 'back':
                                if j > 0:
                                    j -= 1
                                    continue
                                else:
                                    print('Already at the first related symptom.')
                                    continue
                            elif response == "yes":
                                confirmed_symptoms.append(symptom)
                                j += 1
                            elif response == "no":
                                j += 1
                            else:
                                print('Please enter yes, no, or back.')
                                continue

                    # Step 4: Make prediction with medical validation
                    predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(confirmed_symptoms)

                    # Calculate severity details for display
                    _, _, _, present_symptoms = self.calculate_symptom_severity(confirmed_symptoms)

                    # After model prediction, apply post-processing (only once, before any output)
                    post_pred, post_conf, was_swapped = self.postprocess_prediction(predicted_disease, confidence, top_3_diseases, top_3_confidences, [s for s, _ in present_symptoms])
                    if was_swapped:
                        print("\n[INFO] Based on your symptoms, a mild/common condition is more likely. Overriding model prediction.")
                        predicted_disease = post_pred
                        confidence = post_conf
                        # Update description and precautions for the new disease
                        description = self.data_preprocessor.description_list.get(predicted_disease, "")
                        precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                        severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
                    else:
                        description = self.data_preprocessor.description_list.get(predicted_disease, "")
                        precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                        severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)

                    # Always display detailed debug output for clarity
                    self.display_debug_output(predicted_disease, confidence, severity_score, severity_level,
                                            high_severity_symptoms, present_symptoms, severity_recommendations)

                    # Show top-3 diseases if confidence is low
                    if confidence is not None and confidence < 0.6 and top_3_diseases is not None:
                        print("\nOther possible conditions:")
                        for i, (disease, conf) in enumerate(zip(top_3_diseases, top_3_confidences)):
                            print(f"  {i+1}) {disease} (confidence: {conf:.1%})")

                    # Show description and precautions in debug mode
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
                        print(f"\n📝 Description: {desc_to_show}")
                    else:
                        print("\n📝 Description: No description available for this condition.")
                    if precautions:
                        print("\n💡 Take the following precautions:")
                        for i, precaution in enumerate(precautions):
                            if precaution.strip():
                                print(f"   {i+1}) {precaution}")
                    
                    # Doctor recommendations (always show in both modes)
                    self._provide_doctor_recommendations(predicted_disease)

                    break  # Exit the main loop if all steps are completed

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
            print('Type "undo" to go back and change your location or "main" to return to the main menu.')
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
                # Generalized special handling for generic terms
                similar_options = [s for s in self.feature_names if token in s]
                if len(similar_options) > 3:
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
                    continue
                if token == 'fever':
                    fever_options = [s for s in self.feature_names if 'fever' in s]
                    print("You entered 'fever'. Please specify the type(s) of fever:")
                    for i, opt in enumerate(fever_options):
                        print(f"  {i+1}) {opt}")
                    print(f"  0) None of these / skip")
                    selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,2): ").strip()
                    indices = [int(x) for x in selected.split(',') if x.strip().isdigit()]
                    for idx in indices:
                        if 1 <= idx <= len(fever_options):
                            if fever_options[idx-1] not in corrected:
                                corrected.append(fever_options[idx-1])
                    continue
                if token in self.feature_names:
                    if token not in corrected:
                        corrected.append(token)
                else:
                    matches = difflib.get_close_matches(token, self.feature_names, n=3, cutoff=0.0)
                    if matches:
                        best = matches[0]
                        ratio = SequenceMatcher(None, token, best).ratio()
                        if ratio > 0.8:
                            print(f"Interpreting '{token}' as '{best}'.")
                            if best not in corrected:
                                corrected.append(best)
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
                        else:
                            print(f"No good match found for '{token}'. Skipping.")
                    else:
                        print(f"No close match found for '{token}'. Skipping.")
            extracted = corrected
            if not extracted:
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
                print(f'You entered {num_days} days. Is this correct? (y/n)')
                confirm_days = input('-> ').strip().lower()
                if confirm_days == 'y':
                    break
                elif confirm_days == 'n':
                    print('Please re-enter the number of days.')
                    continue
                else:
                    print('Please enter y or n.')
                    continue
            # Ask about relevant symptoms for each confirmed symptom (no duplicates)
            all_symptoms = set(extracted)
            related_answers = []
            i = 0
            while i < len(extracted):
                symptom = extracted[i]
                relevant = self.get_medical_relevant_symptoms(symptom)
                if relevant:
                    print(f"Are you experiencing any of these symptoms related to {symptom}?")
                    rels = [rel for rel in relevant if rel not in all_symptoms]
                    j = 0
                    while j < len(rels):
                        rel = rels[j]
                        response = self.get_valid_input(f"   {rel}? (yes/no/back): ", valid_options=["yes", "no", "back"], symptom_mode=True)
                        if response == 'main_menu':
                            return
                        if response == "back":
                            if j > 0:
                                j -= 1
                                continue
                            else:
                                print('Already at the first related symptom.')
                                continue
                        elif response == "yes":
                            all_symptoms.add(rel)
                            related_answers.append((rel, "yes"))
                            j += 1
                        elif response == "no":
                            related_answers.append((rel, "no"))
                            j += 1
                        else:
                            print('Please enter yes, no, or back.')
                            continue
                i += 1
            # Diagnose
            predicted_disease, confidence, top_3_diseases, top_3_confidences, severity_score, severity_level, high_severity_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms))
            conf_str = f"{confidence:.1%}" if confidence is not None else "Unknown"
            # Calculate severity details for display
            _, _, _, present_symptoms = self.calculate_symptom_severity(list(all_symptoms))
            # After model prediction, apply post-processing (only once, before any output)
            post_pred, post_conf, was_swapped = self.postprocess_prediction(predicted_disease, confidence, top_3_diseases, top_3_confidences, [s for s, _ in present_symptoms])
            if was_swapped:
                print("\n[INFO] Based on your symptoms, a mild/common condition is more likely. Overriding model prediction.")
                predicted_disease = post_pred
                confidence = post_conf
                # Update description and precautions for the new disease
                description = self.data_preprocessor.description_list.get(predicted_disease, "")
                precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
            else:
                description = self.data_preprocessor.description_list.get(predicted_disease, "")
                precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
            # Always display detailed debug output for clarity
            self.display_debug_output(predicted_disease, confidence, severity_score, severity_level,
                                    high_severity_symptoms, present_symptoms, severity_recommendations)
            # Show top-3 diseases if confidence is low
            if confidence is not None and confidence < 0.6 and top_3_diseases is not None:
                print("\nOther possible conditions:")
                for i, (disease, conf) in enumerate(zip(top_3_diseases, top_3_confidences)):
                    print(f"  {i+1}) {disease} (confidence: {conf:.1%})")
            # Show description and precautions in debug mode
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
                print(f"\n📝 Description: {desc_to_show}")
            else:
                print("\n📝 Description: No description available for this condition.")
            if precautions:
                print("\n💡 Take the following precautions:")
                for i, precaution in enumerate(precautions):
                    if precaution.strip():
                        print(f"   {i+1}) {precaution}")
            # Doctor recommendations (always show in both modes)
            self._provide_doctor_recommendations(predicted_disease)
            print("-" * 130)
            print("Thank you for using Healthcare Chatbot! 🙏")
            print("⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
            return
        # Traditional mode: use tree_to_code for full interactive flow
        self.tree_to_code(self.model_trainer.clf, self.data_preprocessor.cols)
        print("-" * 130)


