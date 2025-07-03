#!/usr/bin/env python3
"""
Enhanced Healthcare Chatbot Interface
Symptom-Based Disease Prediction with AI and Machine Learning
"""

import os
import sys
import warnings
import requests
import json
import random
import string
from datetime import datetime, timedelta
import spacy
from spacy.matcher import PhraseMatcher
import time
import re
from sklearn.tree import _tree
from config import (
    MAX_SYMPTOM_SEARCH_RESULTS, 
    MAX_FOLLOW_UP_QUESTIONS,
    MAX_FOLLOW_UP_ROUNDS,
    MAX_HOSPITAL_DISPLAY,
    MAX_DOCTOR_DISPLAY,
    APPOINTMENT_DAYS_AHEAD,
    TIME_SLOTS,
    CONFIDENCE_THRESHOLDS,
    SEVERITY_LEVELS,
    BOOKING_REF_LENGTH,
    BOOKING_ARRIVAL_MINUTES
)
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import difflib
from difflib import SequenceMatcher
import traceback
import pandas as pd
from datetime import datetime


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

        # Cache for expensive operations
        self._dataset_patterns_cache = None
        self._symptom_correlations_cache = None
        
        # Warning tracking to prevent repetition
        self._shown_warnings = set()
        
        # Configurable confidence thresholds (data-driven)
        self.confidence_thresholds = CONFIDENCE_THRESHOLDS

        try:
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
        # Use api_mode to determine behavior
        if hasattr(self, 'api_mode') and self.api_mode:
            # In API mode, just return the list of options for the API handler to process
            if len(similar_options) >= 2:
                return similar_options
        else:
            # CLI/terminal mode: interactive disambiguation
            if len(similar_options) >= 2:
                print(f"You entered '{token}'. Please specify the type(s):")
                for i, opt in enumerate(similar_options):
                    print(f"  {i+1}) {opt}")
                print(f"  0) None of these / skip")
                try:
                    selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,3,5): ").strip()
                except EOFError:
                    print("\n❌ No input available. Skipping selection.")
                    return corrected
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
                    try:
                        selected = input(f"Select all that apply (comma-separated numbers, e.g. 1,3,5): ").strip()
                    except EOFError:
                        print("\n❌ No input available. Skipping selection.")
                        return corrected
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
                    try:
                        choice = input(f"Select the correct symptom for '{token}' (1-{len(matches)} or 0 to skip): ").strip()
                    except EOFError:
                        print("\n❌ No input available. Skipping selection.")
                        break
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
        """Build intelligent medical validation rules based on dataset analysis"""
        return {
            'symptom_severity': {
                'chest_pain': 7, 'high_fever': 7, 'coma': 7, 'paralysis': 7,
                'blood_in_sputum': 6, 'yellowing_skin': 6, 'acute_liver_failure': 7,
                'headache': 3, 'cough': 2, 'fatigue': 2, 'fever': 4, 'mild_fever': 3
            },
            'disease_risk_levels': {
                'high_risk': ['Heart attack', 'Paralysis (brain hemorrhage)', 'AIDS', 'Tuberculosis', 'Malaria', 'Dengue', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Brain hemorrhage'],
                'medium_risk': ['Pneumonia', 'Typhoid', 'Hepatitis A', 'GERD', 'Peptic ulcer disease'],
                'low_risk': ['Common Cold', 'Allergy', 'Migraine', 'Fungal infection', 'Acne']
            },
            'symptom_disease_patterns': {
                'headache': {
                    'common_diseases': ['Peptic ulcer disease', 'Migraine'],
                    'rare_diseases': ['Heart attack', 'Malaria', 'Typhoid'],
                    'min_symptoms_for_rare': 3
                },
                'chest_pain': {
                    'common_diseases': ['GERD', 'Peptic ulcer disease'],
                    'rare_diseases': ['Heart attack', 'Pneumonia'],
                    'min_symptoms_for_rare': 2
                }
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

    def getInfo(self):
        """Get user information and location"""
        print("-----------------------------------HealthCare ChatBot-----------------------------------")
        print("\nYour Name? \t\t\t\t\t", end="->")
        while True:
            try:
                name = input("").strip()
            except EOFError:
                print("\n❌ No input available. Using default name 'User'.")
                name = "User"
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
            try:
                location = input("Please enter your location (city or country): ").strip()
            except EOFError:
                print("\n❌ No input available. Using default location 'Unknown'.")
                return "Unknown"
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
            try:
                user_input = input(prompt).strip().lower()
            except EOFError:
                print("\n❌ No input available. Using default value.")
                if valid_options:
                    return valid_options[0]
                if input_type == int:
                    return 0
                return ""
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
        """Get medically relevant symptoms based on dataset analysis"""
        relevant_symptoms = []
        
        # Analyze dataset patterns to find diseases that have this symptom
        patterns = self.analyze_dataset_patterns()
        
        # Find diseases that have this symptom
        diseases_with_symptom = []
        for disease, pattern in patterns.items():
            if initial_symptom in pattern['symptoms']:
                diseases_with_symptom.append(disease)
        
        # Get symptoms from these diseases
        for disease in diseases_with_symptom:
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
        
        # Filter symptoms based on the type of initial symptom (keep existing logic for now)
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

    def predict_disease_with_medical_validation(self, confirmed_symptoms, denied_symptoms=None):
        """Predict disease with universal medical validation and intelligent follow-up questions"""
        if denied_symptoms is None:
            denied_symptoms = set()
        # Calculate symptom severity
        severity_score, severity_level, high_severity_symptoms, present_symptoms = self.calculate_symptom_severity(confirmed_symptoms)
        # Create input vector using severity scores
        input_vector = [self.data_preprocessor.severityDictionary.get(feature, 0) if feature in confirmed_symptoms else 0 for feature in self.feature_names]
        # Fix: Use DataFrame to avoid sklearn warning
        input_df = pd.DataFrame([input_vector], columns=self.feature_names)
        # Get prediction probabilities
        predicted_proba = self.model_trainer.clf.predict_proba(input_df)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predicted_proba)[::-1][:3]
        top_3_diseases = self.data_preprocessor.le.inverse_transform(top_3_indices)
        top_3_confidences = predicted_proba[top_3_indices]
        
        # Apply universal safety check to top prediction
        top_disease = top_3_diseases[0]
        top_confidence = top_3_confidences[0]
        
        # Universal safety validation (with respect for denied symptoms)
        adjusted_confidence, safety_warnings, risk_level = self.get_universal_safety_check(
            confirmed_symptoms, top_disease, top_confidence, denied_symptoms
        )
        
        # Further adjust confidence based on severity
        final_confidence = self.adjust_confidence_by_severity(adjusted_confidence, severity_score, high_severity_symptoms)
        
        # Cap confidence at 1.0
        final_confidence = min(final_confidence, 1.0)
        
        # Update confidence in the list
        top_3_confidences[0] = final_confidence
        
        # Check if we need to escalate questioning
        needs_escalation = self.should_escalate_questioning(confirmed_symptoms, top_disease, final_confidence)
        
        # Get intelligent follow-up questions
        follow_up_questions, denied_symptoms = self.get_intelligent_follow_up_questions(confirmed_symptoms, top_disease, final_confidence, set(), denied_symptoms)
        
        # Determine if prediction should be made or more questions asked
        # ALWAYS ask follow-up questions if available, regardless of confidence
        should_make_prediction = True
        if len(follow_up_questions) > 0:
            should_make_prediction = False  # Ask follow-up questions first
        
        # Set confidence thresholds based on risk level
        if risk_level == 'high_risk':
            min_confidence = self.confidence_thresholds['high_risk']
        elif risk_level == 'medium_risk':
            min_confidence = self.confidence_thresholds['medium_risk']
        else:
            min_confidence = self.confidence_thresholds['low_risk']
        
        # Always make a prediction, but warn about low confidence
        if final_confidence < min_confidence:
            # Use warning management to prevent repetition
            unique_warnings = self._manage_warnings(safety_warnings, "safety_check")
            if unique_warnings:
                print("🔴 Medical Safety Warnings:")
                for warning in unique_warnings:
                    print(f"   - {warning}")
                print(f"   - Confidence too low ({final_confidence:.1%}) for {risk_level} disease")
            # Don't set prediction to None - keep the prediction but warn about low confidence
            # should_make_prediction = True  # Always make prediction
        
        return top_disease, top_confidence, top_3_diseases, top_3_confidences, should_make_prediction, follow_up_questions, safety_warnings, risk_level, denied_symptoms

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
                print("4) Book hospital appointment")
                print("5) I am done / Exit")
                print("(Type 'undo' to go back and enter your location.)")
                main_choice = self.get_valid_input("Enter 1, 2, 3, 4, or 5: ", valid_options=["1", "2", "3", "4", "5", "undo"])
                if main_choice == 'undo':
                    print("Undo: Returning to location input.")
                    self.user_location = self.get_location()
                    continue
                elif main_choice == "1":
                    self.run_diagnosis_flow()
                    continue  # Return to main menu after diagnosis
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
                                print(f"  Location: {doc.get('city', 'N/A')}, {doc.get('country', 'N/A')}")
                                print(f"  Phone: {doc.get('phone', 'N/A')}")
                                rate = doc.get('rate')
                                if rate is not None:
                                    print(f"  Rating: {rate}/5")
                                else:
                                    print(f"  Rating: N/A")
                                print(f"  Gender: {doc.get('gender', 'N/A')}")
                                # Only show fields that actually exist and have values
                                if doc.get('ImgUrl'):
                                    print(f"  Image: {doc.get('ImgUrl')}")
                                if doc.get('Url'):
                                    print(f"  Profile: {doc.get('Url')}")
                                print("---------------------------")
                        else:
                            print(f"No doctors found for specialty '{specialty}' in {self.user_location}.")
                            if doctors:
                                print(f"\nDoctors specializing in {specialty} in other locations:")
                                for doc in doctors:
                                    print(f"- Name: {doc.get('name', 'Unknown')}")
                                    print(f"  Specialty: {doc.get('specialization', 'N/A')}")
                                    print(f"  Location: {doc.get('city', 'N/A')}, {doc.get('country', 'N/A')}")
                                    print(f"  Phone: {doc.get('phone', 'N/A')}")
                                    rate = doc.get('rate')
                                    if rate is not None:
                                        print(f"  Rating: {rate}/5")
                                    else:
                                        print(f"  Rating: N/A")
                                    print(f"  Gender: {doc.get('gender', 'N/A')}")
                                    # Only show fields that actually exist and have values
                                    if doc.get('ImgUrl'):
                                        print(f"  Image: {doc.get('ImgUrl')}")
                                    if doc.get('Url'):
                                        print(f"  Profile: {doc.get('Url')}")
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
                            for hosp in filtered_hospitals[:MAX_HOSPITAL_DISPLAY]:  # Show top 5 hospitals
                                print(f"- Name: {hosp.get('name', 'Unknown')}")
                                print(f"   📍 Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}")
                                print(f"   📞 Emergency Phone: {hosp.get('phone', 'N/A')}")
                                rate = hosp.get('rate')
                                if rate is not None:
                                    print(f"   ⭐ Rating: {rate}/5")
                                else:
                                    print(f"   ⭐ Rating: N/A")
                                print(f"   🏗️  Established: {hosp.get('Established', 'N/A')}")
                                # Only show URL if it exists
                                if hosp.get('Url'):
                                    print(f"   🌐 Profile: {hosp.get('Url')}")
                                print("---------------------------")
                        else:
                            print(f"No hospitals found in {hosp_location}.")
                            if all_hospitals:
                                print(f"\nAvailable hospitals in other locations:")
                                for hosp in all_hospitals:
                                    print(f"- Name: {hosp.get('name', 'Unknown')}")
                                    print(f"  Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}")
                                    print(f"  Phone: {hosp.get('phone', 'N/A')}")
                                    rate = hosp.get('rate')
                                    if rate is not None:
                                        print(f"  Rating: {rate}/5")
                                    else:
                                        print(f"  Rating: N/A")
                                    print(f"  Established: {hosp.get('Established', 'N/A')}")
                                    # Only show URL if it exists
                                    if hosp.get('Url'):
                                        print(f"   🌐 Profile: {hosp.get('Url')}")
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
                    # Show hospital booking menu
                    self.show_hospital_booking_menu(location=self.user_location)
                    continue
                elif main_choice == "5":
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
        except Exception as e:
            print("❌ An error occurred in the chatbot:", e)
            import traceback
            traceback.print_exc()
            print("Please try again or contact support if the problem persists.")

    def tree_to_code(self, tree, feature_names, confirmed_symptoms=None):
        """
        Traditional chatbot interaction function - ENHANCED VERSION with voice
        """
        try:
            if confirmed_symptoms is None:
                while True:
                    # Step 1: Symptom entry
                    confirmed_symptoms = []
                    restart_symptom_entry = False
                    while True:
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
                        print(f"\nHere are the symptoms I have: {', '.join(self._format_symptom_list(confirmed_symptoms))}")
                        confirm = input("Would you like to change this symptom? (yes/y or no/n)\n-> ").strip().lower()
                        if confirm == "yes" or confirm == "y":
                            print("Enter the new symptom (or type 'undo' to cancel):")
                            new_symptom = input("-> ").strip().lower()
                            if new_symptom == 'undo':
                                print("Edit cancelled. Keeping the original symptom.")
                                continue  # Go back to review prompt, keep original symptom
                            # Use the centralized symptom input handling
                            corrected_tokens = self.handle_single_symptom_input(new_symptom)
                            if corrected_tokens:
                                confirmed_symptoms = corrected_tokens
                            else:
                                print("Symptom not recognized. Please try again.")
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
                        print("Are you experiencing any of these related symptoms?")
                        if not self.production_mode:
                            print("🔍 Are you experiencing any of these medically related symptoms?")
                        rels = [symptom for symptom in relevant_symptoms if symptom not in confirmed_symptoms]
                        j = 0
                        while j < len(rels):
                            symptom = rels[j]
                            formatted_symptom = self._format_symptom_name(symptom)
                            response = self.get_valid_input(f"   {formatted_symptom}? (yes/y or no/n or back): ", valid_options=["yes", "y", "no", "n", "back"], symptom_mode=True)
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
                    denied_symptoms = set()  # Track denied symptoms for the entire session
                    predicted_disease, confidence, top_3_diseases, top_3_confidences, should_make_prediction, follow_up_questions, safety_warnings, risk_level, denied_symptoms = self.predict_disease_with_medical_validation(confirmed_symptoms, denied_symptoms)
                    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                    _, _, _, present_symptoms = self.calculate_symptom_severity(confirmed_symptoms)
                    
                    # Handle intelligent follow-up questions if needed
                    if follow_up_questions:
                        print(f"\n🔍 I need more information for a reliable diagnosis. Let me ask some important questions:")
                        print(f"Current symptoms: {', '.join(self._format_symptom_list(list(confirmed_symptoms)))}")
                        print(f"Potential concern: {predicted_disease} ({conf_str} confidence)")
                        
                        if safety_warnings:
                            # Use warning management to prevent repetition
                            unique_warnings = self._manage_warnings(safety_warnings, "follow_up")
                            if unique_warnings:
                                print("\n⚠️  Medical Safety Warnings:")
                                for warning in unique_warnings:
                                    print(f"   - {warning}")
                        
                        print(f"\nPlease answer these questions:")
                        follow_up_rounds = 0
                        max_follow_up_rounds = MAX_FOLLOW_UP_ROUNDS  # Limit to 2 rounds maximum
                        previously_asked_symptoms = set()  # Track asked symptoms
                        
                        while follow_up_rounds < max_follow_up_rounds and not should_make_prediction and follow_up_questions:
                            follow_up_rounds += 1
                            
                            for i, question in enumerate(follow_up_questions[:MAX_FOLLOW_UP_QUESTIONS], 1):  # Limit to 6 questions
                                if question.startswith("Critical:"):
                                    symptom = question.replace("Critical: ", "").replace(" ", "_")
                                    formatted_symptom = self._format_symptom_name(symptom)
                                    if symptom not in previously_asked_symptoms:
                                        response = self.get_valid_input(f"   {i}. Critical: {formatted_symptom}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                                        previously_asked_symptoms.add(symptom)
                                        if response in ["yes", "y"]:
                                            confirmed_symptoms.append(symptom)
                                        elif response in ["no", "n"]:
                                            denied_symptoms.add(symptom)  # Track denied symptoms
                                elif question.startswith("Important:"):
                                    symptom = question.replace("Important: ", "").replace(" ", "_")
                                    formatted_symptom = self._format_symptom_name(symptom)
                                    if symptom not in previously_asked_symptoms:
                                        response = self.get_valid_input(f"   {i}. Important: {formatted_symptom}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                                        previously_asked_symptoms.add(symptom)
                                        if response in ["yes", "y"]:
                                            confirmed_symptoms.append(symptom)
                                        elif response in ["no", "n"]:
                                            denied_symptoms.add(symptom)  # Track denied symptoms
                            
                            # Deduplicate symptoms
                            confirmed_symptoms = list(set(confirmed_symptoms))
                            
                            # Re-run prediction with updated symptoms and denied_symptoms
                            predicted_disease, confidence, top_3_diseases, top_3_confidences, should_make_prediction, follow_up_questions, safety_warnings, risk_level, denied_symptoms = self.predict_disease_with_medical_validation(confirmed_symptoms, denied_symptoms)
                            
                            # Get new follow-up questions excluding previously asked ones and filtered by denied symptoms
                            if not should_make_prediction:
                                new_follow_up_questions, denied_symptoms = self.get_intelligent_follow_up_questions(confirmed_symptoms, predicted_disease, confidence, previously_asked_symptoms, denied_symptoms)
                                # Only continue if we have new questions to ask
                                if new_follow_up_questions:
                                    follow_up_questions = new_follow_up_questions
                                else:
                                    # No more questions to ask, force prediction
                                    should_make_prediction = True
                                    break
                            
                            conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                            
                            # If still not making prediction and we've reached max rounds, force a safe prediction
                            if follow_up_rounds >= max_follow_up_rounds and not should_make_prediction:
                                print(f"\nBased on your symptoms, I'll provide the best possible diagnosis:")
                                predicted_disease, confidence, top_3_diseases, top_3_confidences = self.get_safe_prediction_after_ruling_out_severe(confirmed_symptoms, top_3_diseases, top_3_confidences)
                                conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                                should_make_prediction = True
                                break
                    
                    # Get description and precautions for the predicted disease
                    description = self.data_preprocessor.description_list.get(predicted_disease, "")
                    precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                    
                    # Display final prediction with explanation
                    self.display_prediction_with_explanation(predicted_disease, confidence, top_3_diseases, top_3_confidences, confirmed_symptoms)
                    
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
                        print(f"\n📋 Description: {desc_to_show}")
                    else:
                        print("\n📋 Description: No description available for this condition.")
                    
                    if precautions:
                        print("\n💊 Take the following precautions:")
                        for i, precaution in enumerate(precautions):
                            if precaution.strip():
                                print(f"   {i+1}) {precaution}")
                    
                    self._provide_doctor_recommendations(predicted_disease)
                    
                    print("-" * 130)
                    print("Thank you for using Healthcare Chatbot! \U0001F64F")
                    print("⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                    return  # Return to main menu after diagnosis
                    
        except Exception as e:
            print("❌ An error occurred during diagnosis:")
            import traceback
            traceback.print_exc()
            print("Please try again or consult a healthcare professional.")

    def _provide_doctor_recommendations(self, predicted_disease):
        """Provide doctor recommendations based on predicted disease - ENHANCED VERSION with Emergency Hospital Support"""
        try:
            specialization = self.disease_to_specialty.get(predicted_disease, None)
            if not specialization:
                print(f"⚠️  No medical specialty found for {predicted_disease}")
                return
            
            # 🚨 EMERGENCY CASE: Recommend hospitals instead of doctors
            if specialization == "Emergency":
                print(f"\n🚨 EMERGENCY ALERT: {predicted_disease} detected!")
                print("⚠️  This is a medical emergency requiring immediate hospital care.")
                print("🏥 Recommending nearby hospitals for emergency treatment:")
                
                # Get location-based emergency number
                emergency_number = self._get_emergency_numbers_by_location(self.user_location)
                
                # Get hospitals in user's location
                if self.user_location:
                    filtered_hospitals, all_hospitals = self.get_hospitals_by_location(self.user_location)
                    
                    if filtered_hospitals:
                        print(f"\n🏥 Emergency Hospitals in {self.user_location}:")
                        for i, hosp in enumerate(filtered_hospitals[:MAX_HOSPITAL_DISPLAY], 1):  # Show top 5 hospitals
                            print(f"{i}. {hosp.get('name', 'Unknown')}")
                            print(f"   📍 Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}")
                            print(f"   📞 Emergency Phone: {hosp.get('phone', 'N/A')}")
                            rate = hosp.get('rate')
                            if rate is not None:
                                print(f"   ⭐ Rating: {rate}/5")
                            else:
                                print(f"   ⭐ Rating: N/A")
                            print(f"   🏗️  Established: {hosp.get('Established', 'N/A')}")
                            # Only show URL if it exists
                            if hosp.get('Url'):
                                print(f"   🌐 Profile: {hosp.get('Url')}")
                            print("---------------------------")
                        
                        print(f"\n🚨 IMMEDIATE ACTION REQUIRED:")
                        print(f"📞 Call emergency services: {emergency_number}")
                        print("1. Go to the nearest hospital emergency department")
                        print("2. Do not delay seeking medical attention")
                        print("3. Bring someone with you if possible")
                        
                    else:
                        print(f"\n⚠️  No hospitals found in {self.user_location}")
                        print(f"🚨 Please call emergency services immediately: {emergency_number}")
                        print("📞 Additional emergency numbers: 911 (US) / 112 (EU) / 999 (UK)")
                        
                        # Show available hospitals in other locations
                        if all_hospitals:
                            print(f"\n🏥 Available hospitals in other locations:")
                            unique_cities = set()
                            for hosp in all_hospitals:
                                city = hosp.get('city', '')
                                if city:
                                    unique_cities.add(city)
                            for city in sorted(unique_cities)[:10]:  # Show top 10 cities
                                print(f"  - {city}")
                else:
                    emergency_number = self._get_emergency_numbers_by_location("unknown")
                    print(f"🚨 Please call emergency services immediately: {emergency_number}")
                    print("📞 Additional emergency numbers: 911 (US) / 112 (EU) / 999 (UK)")
                
                return  # Exit early for emergency cases
                
            # 🏥 NON-EMERGENCY CASE: Recommend doctors as usual
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
                for doc in filtered_doctors:
                    print(f"- Name: {doc.get('name', 'Unknown')}")
                    print(f"  Specialty: {doc.get('specialization', 'N/A')}")
                    print(f"  Location: {doc.get('city', 'N/A')}, {doc.get('country', 'N/A')}")
                    print(f"  Phone: {doc.get('phone', 'N/A')}")
                    rate = doc.get('rate')
                    if rate is not None:
                        print(f"  Rating: {rate}/5")
                    else:
                        print(f"  Rating: N/A")
                    print(f"  Gender: {doc.get('gender', 'N/A')}")
                    # Only show fields that actually exist and have values
                    if doc.get('ImgUrl'):
                        print(f"  Image: {doc.get('ImgUrl')}")
                    if doc.get('Url'):
                        print(f"  Profile: {doc.get('Url')}")
                    print("---------------------------")
            elif doctors:
                print(f"\n🩺 No doctors found in {self.user_location}. Showing top doctors for {specialization}:")
                for doc in doctors:
                    print(f"- Name: {doc.get('name', 'Unknown')}")
                    print(f"  Specialty: {doc.get('specialization', 'N/A')}")
                    print(f"  Location: {doc.get('city', 'N/A')}, {doc.get('country', 'N/A')}")
                    print(f"  Phone: {doc.get('phone', 'N/A')}")
                    rate = doc.get('rate')
                    if rate is not None:
                        print(f"  Rating: {rate}/5")
                    else:
                        print(f"  Rating: N/A")
                    print(f"  Gender: {doc.get('gender', 'N/A')}")
                    # Only show fields that actually exist and have values
                    if doc.get('ImgUrl'):
                        print(f"  Image: {doc.get('ImgUrl')}")
                    if doc.get('Url'):
                        print(f"  Profile: {doc.get('Url')}")
                    print("---------------------------")
                        
        except Exception as e:
            print(f"⚠️  Error getting recommendations: {e}")

    def _provide_emergency_hospital_recommendations(self, predicted_disease, user_location=None):
        """Provide emergency hospital recommendations for emergency conditions"""
        try:
            if not user_location:
                user_location = self.user_location
            
            # Get location-based emergency number
            emergency_number = self._get_emergency_numbers_by_location(user_location)
            
            print(f"\n🚨 EMERGENCY ALERT: {predicted_disease} detected!")
            print("⚠️  This is a medical emergency requiring immediate hospital care.")
            print("🏥 Recommending nearby hospitals for emergency treatment:")
            
            # Get hospitals in user's location
            if user_location:
                filtered_hospitals, all_hospitals = self.get_hospitals_by_location(user_location)
                
                if filtered_hospitals:
                    print(f"\n🏥 Emergency Hospitals in {user_location}:")
                    for i, hosp in enumerate(filtered_hospitals[:MAX_HOSPITAL_DISPLAY], 1):  # Show top 5 hospitals
                        print(f"{i}. {hosp.get('name', 'Unknown')}")
                        print(f"   📍 Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}")
                        print(f"   📞 Emergency Phone: {hosp.get('phone', 'N/A')}")
                        rate = hosp.get('rate')
                        if rate is not None:
                            print(f"   ⭐ Rating: {rate}/5")
                        else:
                            print(f"   ⭐ Rating: N/A")
                        print(f"   🏗️  Established: {hosp.get('Established', 'N/A')}")
                        # Only show URL if it exists
                        if hosp.get('Url'):
                            print(f"   🌐 Profile: {hosp.get('Url')}")
                        print("---------------------------")
                    
                    print(f"\n🚨 IMMEDIATE ACTION REQUIRED:")
                    print(f"📞 Call emergency services: {emergency_number}")
                    print("1. Go to the nearest hospital emergency department")
                    print("2. Do not delay seeking medical attention")
                    print("3. Bring someone with you if possible")
                    
                    # Provide emergency-specific advice based on disease
                    if predicted_disease == "Heart attack":
                        print("\n💔 HEART ATTACK SPECIFIC ADVICE:")
                        print(f"- Call emergency services immediately: {emergency_number}")
                        print("- Sit down and rest, avoid any physical exertion")
                        print("- Take aspirin if available (unless allergic)")
                        print("- Loosen tight clothing")
                        print("- Stay calm and wait for emergency responders")
                    
                    elif predicted_disease == "Drug Reaction":
                        print("\n💊 DRUG REACTION SPECIFIC ADVICE:")
                        print(f"- Call emergency services immediately: {emergency_number}")
                        print("- Stop taking the medication if possible")
                        print("- Monitor for breathing difficulties")
                        print("- If severe, use epinephrine auto-injector if available")
                        print("- Bring the medication container to the hospital")
                    
                else:
                    print(f"\n⚠️  No hospitals found in {user_location}")
                    print(f"🚨 Please call emergency services immediately: {emergency_number}")
                    print("📞 Additional emergency numbers: 911 (US) / 112 (EU) / 999 (UK)")
                    
                    # Show available hospitals in other locations
                    if all_hospitals:
                        print(f"\n🏥 Available hospitals in other locations:")
                        unique_cities = set()
                        for hosp in all_hospitals:
                            city = hosp.get('city', '')
                            if city:
                                unique_cities.add(city)
                        for city in sorted(unique_cities)[:10]:  # Show top 10 cities
                            print(f"  - {city}")
            else:
                print(f"🚨 Please call emergency services immediately: {emergency_number}")
                print("📞 Additional emergency numbers: 911 (US) / 112 (EU) / 999 (UK)")
                
        except Exception as e:
            print(f"⚠️  Error getting emergency hospital recommendations: {e}")

    def _get_emergency_numbers_by_location(self, location):
        """Get emergency numbers based on user location from JSON configuration"""
        try:
            import json
            import os
            
            # Load emergency numbers from JSON file
            json_file_path = os.path.join(os.path.dirname(__file__), 'emergency_numbers.json')
            
            if not os.path.exists(json_file_path):
                print(f"⚠️  Warning: Emergency numbers file not found at {json_file_path}")
                return "911"  # Fallback
            
            with open(json_file_path, 'r', encoding='utf-8') as file:
                emergency_data = json.load(file)
            
            # Extract emergency numbers and major cities
            emergency_numbers = {}
            major_cities = emergency_data.get('major_cities', {})
            fallback_number = emergency_data.get('fallback_number', '911')
            
            # Flatten the regional emergency numbers
            for region, countries in emergency_data.get('emergency_numbers', {}).items():
                emergency_numbers.update(countries)
            
            # Add major cities
            emergency_numbers.update(major_cities)
            
            if not location:
                return fallback_number
            
            location_lower = location.lower().strip()
            
            # Direct match
            if location_lower in emergency_numbers:
                return emergency_numbers[location_lower]
            
            # Partial match for cities within countries
            for country, number in emergency_numbers.items():
                if country in location_lower or location_lower in country:
                    return number
            
            # Fallback to default emergency number
            return fallback_number
            
        except Exception as e:
            print(f"⚠️  Error loading emergency numbers: {e}")
            return "911"  # Default fallback

    def get_doctors_by_specialization(self, specialization):
        """Fetch doctors from API by specialization (improved matching) - ALL PAGES"""
        base_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
        all_doctors = []
        
        try:
            # Fetch first page to get total pages info
            response = requests.get(base_url)
            response.raise_for_status()
            data = response.json()
            
            total_pages = data.get('totalPages', 1)
            current_page = data.get('currentPage', 1)
            
            print(f"📄 Fetching doctors from {total_pages} pages...")
            
            # Fetch all pages
            for page in range(1, total_pages + 1):
                page_url = f"{base_url}?page={page}"
                print(f"  Loading page {page}/{total_pages}...")
                
                response = requests.get(page_url)
                response.raise_for_status()
                data = response.json()
                doctors = data.get('data', [])
                all_doctors.extend(doctors)
                
                # Small delay to be respectful to the API
                import time
                time.sleep(0.1)
            
            print(f"✅ Total doctors fetched: {len(all_doctors)}")
            
            # Improved specialization matching
            filtered = []
            specialization_lower = specialization.lower().strip()
            
            for doc in all_doctors:
                doc_specialization = doc.get('specialization', '').lower().strip()
                # Check for exact match or substring match
                if (specialization_lower == doc_specialization or 
                    specialization_lower in doc_specialization or 
                    doc_specialization in specialization_lower):
                    filtered.append(doc)
            
            # Sort doctors by rating (highest to lowest)
            filtered.sort(key=lambda doc: self._extract_rating_value(doc), reverse=True)
            
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
        if total_severity >= SEVERITY_LEVELS['critical']:
            severity_level = "Critical"
        elif total_severity >= SEVERITY_LEVELS['high']:
            severity_level = "High"
        elif total_severity >= SEVERITY_LEVELS['moderate']:
            severity_level = "Moderate"
        elif total_severity >= SEVERITY_LEVELS['mild']:
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
                recommendations.append(f"High severity symptoms detected: {', '.join([self._format_symptom_name(s[0]) for s in high_severity_symptoms])}")
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
        # print("\n===== DEBUG DIAGNOSIS OUTPUT =====")
        pass

    def postprocess_prediction(self, predicted_disease, confidence, top3, top3_confidences, present_symptoms):
        """Rule-based post-processing to avoid severe diagnoses for mild/common symptoms"""
        # Define mild/common symptoms and diseases
        mild_symptoms = set([
            'runny_nose', 'throat_irritation', 'headache', 'sneezing', 'fatigue', 'cough', 'chills', 'malaise',
            'muscle_pain', 'body_ache', 'nasal_congestion', 'sore_throat', 'watery_eyes', 'mild_fever', 'congestion'
        ])
        mild_diseases = set(['Common Cold', 'Allergy', 'Migraine'])
        severe_diseases = set(['Paralysis (brain hemorrhage)', 'Heart attack', 'AIDS', 'Tuberculosis', 'Malaria', 'Dengue', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Brain hemorrhage'])
        
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
        # Reset warning tracker for new diagnosis session
        self._reset_warning_tracker()
        
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
        
        if method == "1":
            # Traditional mode - use tree_to_code method
            self.tree_to_code(self.model_trainer.clf, self.data_preprocessor.cols)
            return  # Return to main menu after diagnosis
            
        elif method == "2":
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
                return  # Return to main menu after diagnosis
            
            # Add symptom review and edit functionality for free text mode
            while True:
                print(f"\nHere are the symptoms I have: {', '.join(self._format_symptom_list(corrected))}")
                print("What would you like to do?")
                print("1) Add a symptom")
                print("2) Remove a symptom") 
                print("3) Edit all symptoms (re-enter full list)")
                print("4) Continue with current symptoms")
                choice = input("Enter 1, 2, 3, or 4: ").strip()
                
                if choice == "1":
                    # Add a symptom
                    print("Enter the symptom you want to add:")
                    new_symptom = input("-> ").strip().lower()
                    if new_symptom == 'undo':
                        print("Undo: Returning to symptom entry step.")
                        return  # Go back to main menu to restart
                    
                    # Use the same preprocessing logic as initial free text input
                    leading_phrases = [
                        r'^i have ', r'^i am suffering from ', r'^i am having ', r'^i feel ', r'^i am ', r'^i\'m ', r'^i got ', r'^i\s+',
                        r'^my ', r'^having ', r'^suffering from ', r'^experiencing ', r'^with ', r'^and ', r'^, ', r'^\s+'
                    ]
                    tokens = re.split(r',| and | و | و|,|\band\b', new_symptom)
                    cleaned_tokens = []
                    for t in tokens:
                        t = t.strip().lower()
                        for phrase in leading_phrases:
                            t = re.sub(phrase, '', t)
                        t = t.strip()
                        if t:
                            t = t.replace(' ', '_')
                            cleaned_tokens.append(t)
                    
                    for token in cleaned_tokens:
                        corrected_tokens = self.handle_single_symptom_input(token)
                        for ct in corrected_tokens:
                            if ct not in corrected:
                                corrected.append(ct)
                                print(f"Added: {ct}")
                            else:
                                print(f"Symptom '{ct}' is already in your list.")
                    continue
                    
                elif choice == "2":
                    # Remove a symptom
                    if len(corrected) == 1:
                        print("You only have one symptom. You cannot remove it.")
                        continue
                    
                    print("Which symptom would you like to remove?")
                    for i, symptom in enumerate(corrected, 1):
                        formatted_symptom = self._format_symptom_name(symptom)
                        print(f"{i}) {formatted_symptom}")
                    
                    try:
                        remove_choice = int(input("Enter the number of the symptom to remove: ").strip())
                        if 1 <= remove_choice <= len(corrected):
                            removed_symptom = corrected.pop(remove_choice - 1)
                            formatted_removed = self._format_symptom_name(removed_symptom)
                            print(f"Removed: {formatted_removed}")
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
                    continue
                    
                elif choice == "3":
                    # Edit all symptoms (re-enter full list)
                    print("Enter the full, final list of symptoms separated by commas (or type 'undo' to go back and re-enter symptoms):")
                    final = input("-> ").strip().lower()
                    if final == 'undo':
                        print("Undo: Returning to symptom entry step.")
                        return  # Go back to main menu to restart
                    
                    # Use the same preprocessing logic as initial free text input
                    leading_phrases = [
                        r'^i have ', r'^i am suffering from ', r'^i am having ', r'^i feel ', r'^i am ', r'^i\'m ', r'^i got ', r'^i\s+',
                        r'^my ', r'^having ', r'^suffering from ', r'^experiencing ', r'^with ', r'^and ', r'^, ', r'^\s+'
                    ]
                    tokens = re.split(r',| and | و | و|,|\band\b', final)
                    cleaned_tokens = []
                    for t in tokens:
                        t = t.strip().lower()
                        for phrase in leading_phrases:
                            t = re.sub(phrase, '', t)
                        t = t.strip()
                        if t:
                            t = t.replace(' ', '_')
                            cleaned_tokens.append(t)
                    
                    validated = []
                    for token in cleaned_tokens:
                        # Use the centralized symptom input handling
                        corrected_tokens = self.handle_single_symptom_input(token)
                        validated.extend(corrected_tokens)
                    if validated:
                        corrected = validated
                    else:
                        print("No valid symptoms entered. Please try again.")
                        continue
                    continue
                    
                elif choice == "4":
                    # Continue with current symptoms
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
                    continue
            
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
                    print(f"Are you experiencing any of these symptoms related to {self._format_symptom_name(symptom)}?")
                    rels = [rel for rel in relevant if rel not in all_symptoms]
                    j = 0
                    while j < len(rels):
                        rel = rels[j]
                        formatted_symptom = self._format_symptom_name(rel)
                        response = self.get_valid_input(f"   {formatted_symptom}? (yes/y or no/n or back): ", valid_options=["yes", "y", "no", "n", "back"], symptom_mode=True)
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
            predicted_disease, confidence, top_3_diseases, top_3_confidences, should_make_prediction, follow_up_questions, safety_warnings, risk_level, denied_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms), set())
            conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
            _, _, _, present_symptoms = self.calculate_symptom_severity(list(all_symptoms))
            post_pred, post_conf, was_swapped = self.postprocess_prediction(predicted_disease, confidence, top_3_diseases, top_3_confidences, [s for s, _ in present_symptoms])
            
            # Handle intelligent follow-up questions if needed
            follow_up_processed = False  # Track if follow-up questions were processed
            if follow_up_questions:
                print(f"\n🔍 I need more information for a reliable diagnosis. Let me ask some important questions:")
                print(f"Current symptoms: {', '.join(self._format_symptom_list(list(all_symptoms)))}")
                print(f"Potential concern: {predicted_disease} ({conf_str} confidence)")
                
                if safety_warnings:
                    # Use warning management to prevent repetition
                    unique_warnings = self._manage_warnings(safety_warnings, "follow_up")
                    if unique_warnings:
                        print("\n⚠️  Medical Safety Warnings:")
                        for warning in unique_warnings:
                            print(f"   - {warning}")
                
                print(f"\nPlease answer these questions:")
                follow_up_rounds = 0
                max_follow_up_rounds = MAX_FOLLOW_UP_ROUNDS  # Limit to 2 rounds maximum
                previously_asked_symptoms = set()  # Track asked symptoms
                denied_symptoms = set()  # Track denied symptoms
                confirmed_symptoms = list(all_symptoms)  # Initialize with current symptoms
                
                while follow_up_rounds < max_follow_up_rounds and not should_make_prediction and follow_up_questions:
                    follow_up_rounds += 1
                    
                    for i, question in enumerate(follow_up_questions[:MAX_FOLLOW_UP_QUESTIONS], 1):  # Limit to 6 questions
                        if question.startswith("Critical:"):
                            symptom = question.replace("Critical: ", "").replace(" ", "_")
                            formatted_symptom = self._format_symptom_name(symptom)
                            if symptom not in previously_asked_symptoms:
                                response = self.get_valid_input(f"   {i}. Critical: {formatted_symptom}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                                previously_asked_symptoms.add(symptom)
                                if response in ["yes", "y"]:
                                    confirmed_symptoms.append(symptom)
                                elif response in ["no", "n"]:
                                    denied_symptoms.add(symptom)  # Track denied symptoms
                        elif question.startswith("Important:"):
                            symptom = question.replace("Important: ", "").replace(" ", "_")
                            formatted_symptom = self._format_symptom_name(symptom)
                            if symptom not in previously_asked_symptoms:
                                response = self.get_valid_input(f"   {i}. Important: {formatted_symptom}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                                previously_asked_symptoms.add(symptom)
                                if response in ["yes", "y"]:
                                    confirmed_symptoms.append(symptom)
                                elif response in ["no", "n"]:
                                    denied_symptoms.add(symptom)  # Track denied symptoms
                    
                    # Re-run prediction with updated symptoms
                    predicted_disease, confidence, top_3_diseases, top_3_confidences, should_make_prediction, follow_up_questions, safety_warnings, risk_level, denied_symptoms = self.predict_disease_with_medical_validation(confirmed_symptoms, denied_symptoms)
                    
                    # Get new follow-up questions excluding previously asked ones and filtered by denied symptoms
                    if not should_make_prediction:
                        new_follow_up_questions, denied_symptoms = self.get_intelligent_follow_up_questions(confirmed_symptoms, predicted_disease, confidence, previously_asked_symptoms, denied_symptoms)
                        # Only continue if we have new questions to ask
                        if new_follow_up_questions:
                            follow_up_questions = new_follow_up_questions
                        else:
                            # No more questions to ask, force prediction
                            should_make_prediction = True
                            break
                    
                    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                    
                    # If still not making prediction and we've reached max rounds, force a safe prediction
                    if follow_up_rounds >= max_follow_up_rounds and not should_make_prediction:
                        print(f"\nBased on your symptoms, I'll provide the best possible diagnosis:")
                        predicted_disease, confidence, top_3_diseases, top_3_confidences = self.get_safe_prediction_after_ruling_out_severe(confirmed_symptoms, top_3_diseases, top_3_confidences)
                        conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                        should_make_prediction = True
                        break
                    
                    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                    
                    # If still not making prediction and we've reached max rounds, force a safe prediction
                    if follow_up_rounds >= max_follow_up_rounds and not should_make_prediction:
                        print(f"\nBased on your symptoms, I'll provide the best possible diagnosis:")
                        predicted_disease, confidence, top_3_diseases, top_3_confidences = self.get_safe_prediction_after_ruling_out_severe(confirmed_symptoms, top_3_diseases, top_3_confidences)
                        conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                        should_make_prediction = True
                        break
                    
                    follow_up_processed = True  # Mark that follow-up questions were processed
                    
                if was_swapped:
                    predicted_disease = post_pred
                    confidence = post_conf
                    description = self.data_preprocessor.description_list.get(predicted_disease, "")
                    precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                    severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)
                else:
                    description = self.data_preprocessor.description_list.get(predicted_disease, "")
                    precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
                    # Recalculate severity after follow-up questions
                    severity_score, severity_level, high_severity_symptoms, present_symptoms = self.calculate_symptom_severity(confirmed_symptoms)
                    severity_recommendations = self.get_severity_based_recommendations(severity_level, high_severity_symptoms, predicted_disease)

                # --- NEW OUTPUT STYLE ---
                print(f"\n\U0001F4CB Current symptoms: {', '.join(self._format_symptom_list(list(all_symptoms)))}")
                
                # Only ask for additional symptoms if we haven't already done follow-up questions
                if not follow_up_processed and (confidence is None or confidence < 0.5):
                    print(f"\n\u26A0\uFE0F  I need at least 4 symptoms for a proper diagnosis. You have {len(list(all_symptoms))}.")
                    print("Let me ask about some common symptoms:")
                    for s in ['fever', 'fatigue', 'loss_of_appetite', 'weight_loss', 'chills']:
                        if s not in all_symptoms:
                            yn = self.get_valid_input(f"   {s.replace('_', ' ')}? (yes/y or no/n): ", valid_options=["yes", "y", "no", "n"])
                            if yn in ["yes", "y"]:
                                all_symptoms.add(s)
                    print(f"\n\U0001F4CA Current analysis: {len(list(all_symptoms))} symptoms, confidence: {conf_str}")
                    print("To improve accuracy, please tell me about any other symptoms you're experiencing:")
                    print(f"🔍 DEBUG: Before additional symptoms, denied_symptoms = {denied_symptoms}")
                    extra = input("Additional symptom 1 (or press Enter to skip): ").strip().replace(' ', '_')
                    print(f"🔍 DEBUG: User entered: '{extra}'")
                    if extra:
                        corrected_extra = self.handle_single_symptom_input(extra)
                        for ce in corrected_extra:
                            if ce not in all_symptoms:
                                all_symptoms.add(ce)
                            else:
                                denied_symptoms.add(ce)
                                print(f"🔍 DEBUG: Added {ce} to denied_symptoms from additional symptoms, now = {denied_symptoms}")
                    else:
                        print(f"🔍 DEBUG: User skipped additional symptoms")
                    print(f"🔍 DEBUG: Before continue, denied_symptoms = {denied_symptoms}")
                    print(f"🔍 DEBUG: Current confirmed_symptoms = {confirmed_symptoms}")
                    print(f"🔍 DEBUG: About to continue to next iteration...")
                    # Re-run prediction with updated symptoms and denied_symptoms
                    predicted_disease, confidence, top_3_diseases, top_3_confidences, should_make_prediction, follow_up_questions, safety_warnings, risk_level, denied_symptoms = self.predict_disease_with_medical_validation(list(all_symptoms), denied_symptoms)
                    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
                    # Force prediction after additional symptoms section
                    should_make_prediction = True
                
                # Display final prediction with explanation
                self.display_prediction_with_explanation(predicted_disease, confidence, top_3_diseases, top_3_confidences, list(all_symptoms))
                
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
                    print(f"\n📋 Description: {desc_to_show}")
                else:
                    print("\n📋 Description: No description available for this condition.")
                
                if precautions:
                    print("\n💊 Take the following precautions:")
                    for i, precaution in enumerate(precautions):
                        if precaution.strip():
                            print(f"   {i+1}) {precaution}")
                
                self._provide_doctor_recommendations(predicted_disease)
                
                print("-" * 130)
                print("Thank you for using Healthcare Chatbot! \U0001F64F")
                print("⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
                return  # Return to main menu after diagnosis

    def get_hospitals_by_location(self, location):
        """Fetch hospitals from API by location (city or country) - ALL PAGES"""
        base_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
        all_hospitals = []
        
        try:
            # Fetch first page to get total pages info
            response = requests.get(base_url)
            response.raise_for_status()
            data = response.json()
            
            total_pages = data.get('totalPages', 1)
            current_page = data.get('currentPage', 1)
            
            print(f"📄 Fetching hospitals from {total_pages} pages...")
            
            # Fetch all pages
            for page in range(1, total_pages + 1):
                page_url = f"{base_url}?page={page}"
                print(f"  Loading page {page}/{total_pages}...")
                
                response = requests.get(page_url)
                response.raise_for_status()
                data = response.json()
                hospitals = data.get('data', [])
                all_hospitals.extend(hospitals)
                
                # Small delay to be respectful to the API
                import time
                time.sleep(0.1)
            
            print(f"✅ Total hospitals fetched: {len(all_hospitals)}")
            
            # Filter by location
            location_lower = location.lower().strip()
            filtered = []
            for hosp in all_hospitals:
                hosp_city = str(hosp.get('city', '')).lower().strip()
                hosp_country = str(hosp.get('country', '')).lower().strip()
                hosp_location = f"{hosp_city} {hosp_country}".strip()
                if (location_lower in hosp_location or 
                    hosp_city in location_lower or 
                    hosp_country in location_lower):
                    filtered.append(hosp)
            
            # Sort hospitals by rating (highest to lowest)
            filtered.sort(key=lambda hosp: self._extract_rating_value(hosp), reverse=True)
            
            return filtered, all_hospitals
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not fetch hospitals from API: {e}")
            return [], []
        except Exception as e:
            print(f"⚠️  An error occurred while processing hospital data: {e}")
            return [], []

    def book_hospital_appointment(self, hospital_id, patient_name, patient_phone, appointment_date, appointment_time, symptoms=None):
        """Book an appointment at a hospital"""
        print("\n🏥 HOSPITAL BOOKING SYSTEM")
        print("=" * 50)
        
        
        # Since there's no actual booking API, we'll simulate the booking process
        print(f"📋 Booking Details:")
        print(f"  Hospital ID: {hospital_id}")
        print(f"  Patient Name: {patient_name}")
        print(f"  Patient Phone: {patient_phone}")
        print(f"  Date: {appointment_date}")
        print(f"  Time: {appointment_time}")
        if symptoms:
            formatted_symptoms = self._format_symptom_list(symptoms)
            print(f"  Symptoms: {', '.join(formatted_symptoms)}")
        
        # Simulate booking confirmation
        print("\n✅ BOOKING CONFIRMED!")
        print("=" * 50)
        print("📅 Your appointment has been successfully booked.")
        print("📞 You will receive a confirmation call shortly.")
        print(f"🏥 Please arrive {BOOKING_ARRIVAL_MINUTES} minutes before your appointment time.")
        print("📋 Don't forget to bring your ID and insurance card.")
        
        # Generate booking reference
        import random
        import string
        booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=BOOKING_REF_LENGTH))
        print(f"🔢 Booking Reference: {booking_ref}")
        
        return {
            'status': 'confirmed',
            'booking_reference': booking_ref,
            'hospital_id': hospital_id,
            'patient_name': patient_name,
            'patient_phone': patient_phone,
            'appointment_date': appointment_date,
            'appointment_time': appointment_time,
            'symptoms': symptoms
        }

    def show_hospital_booking_menu(self, location=None, symptoms=None):
        """Show hospital booking menu and handle booking process"""
        print("\n🏥 HOSPITAL BOOKING MENU")
        print("=" * 50)
        
        if not location:
            location = self.get_location()
        
        # Get hospitals in the location
        filtered_hospitals, all_hospitals = self.get_hospitals_by_location(location)
        
        if not filtered_hospitals:
            print(f"❌ No hospitals found in {location}")
            print("💡 Available locations:")
            unique_cities = set()
            for hosp in all_hospitals:
                city = hosp.get('city', '')
                if city:
                    unique_cities.add(city)
            for city in sorted(unique_cities):
                print(f"  - {city}")
            return
        
        print(f"🏥 Found {len(filtered_hospitals)} hospitals in {location}:")
        print()
        
        for i, hospital in enumerate(filtered_hospitals, 1):
            name = hospital.get('name', 'Unknown')
            city = hospital.get('city', 'Unknown')
            country = hospital.get('country', 'Unknown')
            rate = hospital.get('rate')
            phone = hospital.get('phone', 'N/A')
            established = hospital.get('Established', 'N/A')
            
            print(f"{i}. {name}")
            print(f"   📍 Location: {city}, {country}")
            if rate is not None:
                print(f"   ⭐ Rating: {rate}/5")
            else:
                print(f"   ⭐ Rating: N/A")
            print(f"   📞 Phone: {phone}")
            print(f"   🏗️  Established: {established}")
            print()
        
        # Get user choice
        choice = self._get_numeric_input(
            f"Select hospital (1-{len(filtered_hospitals)}) or 'back' to return: ",
            1, len(filtered_hospitals), allow_back=True
        )
        
        if choice == 'back':
            return
        
        selected_hospital = filtered_hospitals[choice - 1]
        
        # Get booking details
        print(f"\n📋 Booking appointment at {selected_hospital.get('name', 'Unknown')}")
        print("=" * 50)
        
        patient_name = input("Enter your full name: ").strip()
        if not patient_name:
            print("❌ Name is required.")
            return
        
        patient_phone = input("Enter your phone number: ").strip()
        if not patient_phone:
            print("❌ Phone number is required.")
            return
        
        # Get appointment date
        from datetime import datetime, timedelta
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        print(f"\n📅 Available dates (starting from tomorrow):")
        for i in range(APPOINTMENT_DAYS_AHEAD):  # Show next 7 days
            date = tomorrow + timedelta(days=i)
            print(f"  {i+1}. {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
        
        date_choice = self._get_numeric_input("Select date (1-7): ", 1, APPOINTMENT_DAYS_AHEAD)
        appointment_date = (tomorrow + timedelta(days=date_choice-1)).strftime('%Y-%m-%d')
        
        # Get appointment time
        print(f"\n🕐 Available time slots:")
        
        for i, time_slot in enumerate(TIME_SLOTS, 1):
            print(f"  {i}. {time_slot}")
        
        time_choice = self._get_numeric_input(f"Select time (1-{len(TIME_SLOTS)}): ", 1, len(TIME_SLOTS))
        appointment_time = TIME_SLOTS[time_choice - 1]
        
        # Confirm booking
        print(f"\n📋 Booking Summary:")
        print(f"  Hospital: {selected_hospital.get('name', 'Unknown')}")
        print(f"  Patient: {patient_name}")
        print(f"  Phone: {patient_phone}")
        print(f"  Date: {appointment_date}")
        print(f"  Time: {appointment_time}")
        if symptoms:
            formatted_symptoms = self._format_symptom_list(symptoms)
            print(f"  Symptoms: {', '.join(formatted_symptoms)}")
        
        confirm = input("\nConfirm booking? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            # Process booking
            booking_result = self.book_hospital_appointment(
                hospital_id=selected_hospital.get('_id', 'unknown'),
                patient_name=patient_name,
                patient_phone=patient_phone,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                symptoms=symptoms
            )
            
            # Save booking to file (for demo purposes)
            self.save_booking_to_file(booking_result)
            
            print("\n🎉 Booking completed successfully!")
            print("📧 A confirmation email has been sent to your registered email.")
            print("📱 You will also receive an SMS confirmation.")
            
        else:
            print("❌ Booking cancelled.")

    def save_booking_to_file(self, booking_data):
        """Save booking data to a file for record keeping"""
        # Add timestamp to booking data
        booking_data['booking_timestamp'] = datetime.now().isoformat()
        
        # Create bookings directory if it doesn't exist
        import os
        bookings_dir = "bookings"
        if not os.path.exists(bookings_dir):
            os.makedirs(bookings_dir)
        
        # Save to file
        filename = f"bookings/booking_{booking_data['booking_reference']}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(booking_data, f, indent=2, ensure_ascii=False)
            print(f"📄 Booking details saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Could not save booking details: {e}")

    def analyze_dataset_patterns(self):
        """Analyze dataset to create universal symptom-disease patterns"""
        # Return cached result if available
        if self._dataset_patterns_cache is not None:
            return self._dataset_patterns_cache
            
        patterns = {}
        
        # Get all diseases and their symptoms from dataset
        for disease in self.data_preprocessor.reduced_data.index:
            disease_symptoms = []
            for col in self.data_preprocessor.reduced_data.columns:
                if col != "Medical Specialties" and self.data_preprocessor.reduced_data.loc[disease, col] == 1:
                    disease_symptoms.append(col)
            
            # Calculate disease characteristics
            symptom_count = len(disease_symptoms)
            avg_severity = sum(self.data_preprocessor.severityDictionary.get(s, 1) for s in disease_symptoms) / symptom_count if symptom_count > 0 else 0
            
            patterns[disease] = {
                'symptoms': disease_symptoms,
                'symptom_count': symptom_count,
                'avg_severity': avg_severity,
                'min_symptoms_required': max(2, symptom_count // 3),  # At least 1/3 of typical symptoms
                'critical_symptoms': [s for s in disease_symptoms if self.data_preprocessor.severityDictionary.get(s, 1) >= 6]
            }
        
        # Cache the result
        self._dataset_patterns_cache = patterns
        return patterns

    def get_universal_safety_check(self, symptoms, predicted_disease, confidence, denied_symptoms=None):
        """Universal safety check based on dataset patterns with respect for denied symptoms"""
        if denied_symptoms is None:
            denied_symptoms = set()
            
        patterns = self.analyze_dataset_patterns()
        
        if predicted_disease not in patterns:
            return confidence, [], 'unknown'
        
        disease_pattern = patterns[predicted_disease]
        warnings = []
        adjusted_confidence = confidence
        
        # Check if we have enough symptoms for this disease
        if len(symptoms) < disease_pattern['min_symptoms_required']:
            missing_count = disease_pattern['min_symptoms_required'] - len(symptoms)
            warnings.append(f"Need {missing_count} more symptoms for reliable {predicted_disease} diagnosis")
            # Use additive penalty instead of multiplicative
            penalty = min(self.confidence_thresholds['max_penalty'], missing_count * self.confidence_thresholds['symptom_penalty'])
            adjusted_confidence -= penalty
        
        # Check if we have critical symptoms (excluding denied ones)
        missing_critical = [s for s in disease_pattern['critical_symptoms'] if s not in symptoms and s not in denied_symptoms]
        if missing_critical:
            warnings.append(f"Missing critical symptoms: {', '.join(missing_critical)}")
            # Use additive penalty instead of multiplicative
            penalty = min(self.confidence_thresholds['max_penalty'], len(missing_critical) * self.confidence_thresholds['critical_penalty'])
            adjusted_confidence -= penalty
        
        # Check symptom severity consistency
        current_avg_severity = sum(self.data_preprocessor.severityDictionary.get(s, 1) for s in symptoms) / len(symptoms) if symptoms else 0
        if current_avg_severity < disease_pattern['avg_severity'] * 0.7:
            warnings.append(f"Symptom severity too low for {predicted_disease}")
            # Use additive penalty instead of multiplicative
            severity_penalty = min(self.confidence_thresholds['max_penalty'], 
                                 (disease_pattern['avg_severity'] * 0.7 - current_avg_severity) * self.confidence_thresholds['severity_penalty'])
            adjusted_confidence -= severity_penalty
        
        # Ensure confidence doesn't go below minimum threshold
        adjusted_confidence = max(adjusted_confidence, self.confidence_thresholds['min_confidence'])
        
        # Determine risk level
        risk_level = 'low_risk'
        if disease_pattern['avg_severity'] >= 6 or predicted_disease in ['Heart attack', 'Brain hemorrhage', 'Paralysis (brain hemorrhage)']:
            risk_level = 'high_risk'
        elif disease_pattern['avg_severity'] >= 4:
            risk_level = 'medium_risk'
        
        return adjusted_confidence, warnings, risk_level

    def get_intelligent_follow_up_questions(self, symptoms, predicted_disease, confidence, previously_asked_symptoms=None, denied_symptoms=None):
        """Generate intelligent follow-up questions based on dataset analysis with STRICT denial exclusion"""
        if previously_asked_symptoms is None:
            previously_asked_symptoms = set()
        if denied_symptoms is None:
            denied_symptoms = set()
        
        patterns = self.analyze_dataset_patterns()
        
        if predicted_disease not in patterns:
            return [], denied_symptoms  # Return empty questions and denied_symptoms
        
        disease_pattern = patterns[predicted_disease]
        questions = []
        
        # STRICT DENIAL EXCLUSION: Get all symptoms that should be excluded based on denied symptoms
        excluded_symptoms = self.get_strictly_excluded_symptoms(denied_symptoms)
        
        # Get missing critical symptoms (not previously asked AND not excluded AND not denied)
        missing_critical = [s for s in disease_pattern['critical_symptoms'] 
                          if s not in symptoms and s not in previously_asked_symptoms and s not in excluded_symptoms and s not in denied_symptoms]
        
        if missing_critical:
            questions.extend([f"Critical: {s.replace('_', ' ')}" for s in missing_critical[:3]])
        
        # Get high-importance missing symptoms (not previously asked, not in critical, AND not excluded AND not denied)
        missing_symptoms = [s for s in disease_pattern['symptoms'] 
                          if s not in symptoms and s not in previously_asked_symptoms and s not in disease_pattern['critical_symptoms'] and s not in excluded_symptoms and s not in denied_symptoms]
        
        if missing_symptoms:
            # Sort by severity and importance
            missing_with_importance = []
            for symptom in missing_symptoms:
                severity = self.data_preprocessor.severityDictionary.get(symptom, 1)
                if symptom in self.feature_names:
                    idx = self.feature_names.index(symptom)
                    importance = self._get_feature_importances()[idx]
                    missing_with_importance.append((symptom, severity * importance))
            
            missing_with_importance.sort(key=lambda x: x[1], reverse=True)
            questions.extend([f"Important: {s.replace('_', ' ')}" for s, _ in missing_with_importance[:5]])
        
        return questions[:8], denied_symptoms  # Return questions and denied_symptoms

    def get_strictly_excluded_symptoms(self, denied_symptoms):
        """Get ALL symptoms that should be excluded based on denied symptoms using strict correlation analysis"""
        if not denied_symptoms:
            return set()
        
        correlations = self.analyze_symptom_correlations()
        excluded_symptoms = set(denied_symptoms)  # Start with the denied symptoms themselves
        
        # Add all symptoms that are correlated with any denied symptom
        for denied in denied_symptoms:
            if denied in correlations:
                for symptom, correlation_score in correlations[denied].items():
                    # Use a lower threshold for strict exclusion (0.2 instead of 0.3)
                    if correlation_score > 0.2:
                        excluded_symptoms.add(symptom)
        
        # Also add symptoms from the same clusters as denied symptoms
        clusters = self._build_symptom_clusters()
        for denied in denied_symptoms:
            for cluster_name, cluster_symptoms_list in clusters.items():
                if denied in cluster_symptoms_list:
                    # Add all symptoms from the same cluster
                    excluded_symptoms.update(cluster_symptoms_list)
        
        return excluded_symptoms

    def should_escalate_questioning(self, symptoms, predicted_disease, confidence):
        """Determine if we need more intensive questioning"""
        patterns = self.analyze_dataset_patterns()
        
        if predicted_disease not in patterns:
            return False
        
        disease_pattern = patterns[predicted_disease]
        
        # Escalate if:
        # 1. High-risk disease with insufficient symptoms
        # 2. Missing critical symptoms
        # 3. Low confidence for severe disease
        
        is_high_risk = disease_pattern['avg_severity'] >= 6 or predicted_disease in ['Heart attack', 'Brain hemorrhage']
        has_insufficient_symptoms = len(symptoms) < disease_pattern['min_symptoms_required']
        missing_critical = len([s for s in disease_pattern['critical_symptoms'] if s not in symptoms]) > 0
        low_confidence = confidence < 0.6
        
        return is_high_risk and (has_insufficient_symptoms or missing_critical or low_confidence)

    def get_safe_prediction_after_ruling_out_severe(self, symptoms, top_3_diseases, top_3_confidences):
        """Get a safe prediction after ruling out severe diseases"""
        patterns = self.analyze_dataset_patterns()
        
        # Filter out high-risk diseases
        safe_diseases = []
        safe_confidences = []
        
        for disease, confidence in zip(top_3_diseases, top_3_confidences):
            if disease in patterns:
                disease_pattern = patterns[disease]
                is_high_risk = disease_pattern['avg_severity'] >= 6 or disease in ['Heart attack', 'Brain hemorrhage']
                
                if not is_high_risk:
                    safe_diseases.append(disease)
                    safe_confidences.append(confidence)
            else:
                # If disease not in patterns, consider it safe
                safe_diseases.append(disease)
                safe_confidences.append(confidence)
        
        # Return the best safe prediction
        if safe_diseases:
            return safe_diseases[0], safe_confidences[0], safe_diseases, safe_confidences
        else:
            # If no safe diseases, return the original top prediction with low confidence
            return top_3_diseases[0], top_3_confidences[0] * 0.3, top_3_diseases, top_3_confidences

    def get_context_aware_symptoms(self, current_symptoms, predicted_disease):
        """Get context-aware symptoms based on current symptoms and predicted disease"""
        # Get medical specialty for the predicted disease
        specialty = self.get_medical_specialty_for_disease(predicted_disease)
        
        # Define specialty-specific symptom priorities
        specialty_priorities = {
            'Cardiology': ['chest_pain', 'breathlessness', 'palpitations', 'fast_heart_rate', 'swollen_legs'],
            'Neurology': ['headache', 'visual_disturbances', 'altered_sensorium', 'weakness_in_limbs', 'slurred_speech'],
            'Gastroenterology': ['abdominal_pain', 'nausea', 'vomiting', 'diarrhoea', 'constipation', 'acidity'],
            'Dermatology': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'blister', 'red_sore_around_nose'],
            'Respiratory': ['cough', 'breathlessness', 'chest_pain', 'blood_in_sputum', 'mucoid_sputum'],
            'Endocrinology': ['fatigue', 'weight_gain', 'weight_loss', 'excessive_hunger', 'polyuria'],
            'Urology': ['burning_micturition', 'spotting_urination', 'bladder_discomfort', 'foul_smell_ofurine'],
            'Orthopedics': ['joint_pain', 'muscle_pain', 'back_pain', 'knee_pain', 'hip_joint_pain']
        }
        
        # Get priority symptoms for the specialty
        priority_symptoms = specialty_priorities.get(specialty, [])
        
        # Also consider symptoms from the same clusters as current symptoms
        clusters = self._build_symptom_clusters()
        cluster_symptoms = []
        
        for symptom in current_symptoms:
            for cluster_name, cluster_symptoms_list in clusters.items():
                if symptom in cluster_symptoms_list:
                    cluster_symptoms.extend(cluster_symptoms_list)
        
        # Remove duplicates and current symptoms
        cluster_symptoms = list(set(cluster_symptoms))
        cluster_symptoms = [s for s in cluster_symptoms if s not in current_symptoms]
        
        # Combine priority symptoms and cluster symptoms
        context_symptoms = priority_symptoms + cluster_symptoms
        
        # Remove duplicates while preserving order
        seen = set()
        unique_context_symptoms = []
        for symptom in context_symptoms:
            if symptom not in seen:
                seen.add(symptom)
                unique_context_symptoms.append(symptom)
        
        return unique_context_symptoms

    def _manage_warnings(self, warnings, context=""):
        """Manage warnings to prevent repetition within the same diagnosis session"""
        if not warnings:
            return []
        
        # Create unique warning identifiers
        unique_warnings = []
        for warning in warnings:
            warning_id = f"{context}:{warning}"
            if warning_id not in self._shown_warnings:
                unique_warnings.append(warning)
                self._shown_warnings.add(warning_id)
        
        return unique_warnings

    def _reset_warning_tracker(self):
        """Reset warning tracker for new diagnosis sessions"""
        self._shown_warnings.clear()

    def analyze_symptom_correlations(self):
        """Analyze symptom co-occurrence patterns in the dataset for intelligent filtering"""
        # Return cached result if available
        if self._symptom_correlations_cache is not None:
            return self._symptom_correlations_cache
            
        correlations = {}
        
        # Use existing reduced_data to analyze symptom relationships
        symptom_cols = [col for col in self.data_preprocessor.reduced_data.columns 
                       if col != "Medical Specialties"]
        
        # Calculate correlation matrix using existing data
        for i, symptom1 in enumerate(symptom_cols):
            correlations[symptom1] = {}
            for j, symptom2 in enumerate(symptom_cols):
                if i != j:
                    # Calculate co-occurrence probability
                    both_present = 0
                    total_cases = 0
                    
                    for disease in self.data_preprocessor.reduced_data.index:
                        if (self.data_preprocessor.reduced_data.loc[disease, symptom1] == 1 and 
                            self.data_preprocessor.reduced_data.loc[disease, symptom2] == 1):
                            both_present += 1
                        if (self.data_preprocessor.reduced_data.loc[disease, symptom1] == 1 or 
                            self.data_preprocessor.reduced_data.loc[disease, symptom2] == 1):
                            total_cases += 1
                    
                    # Calculate correlation score (0 to 1, where 1 means high positive correlation)
                    if total_cases > 0:
                        correlation_score = both_present / total_cases
                    else:
                        correlation_score = 0
                    
                    correlations[symptom1][symptom2] = correlation_score
        
        # Cache the result
        self._symptom_correlations_cache = correlations
        return correlations

    def filter_symptoms_by_denials(self, candidate_symptoms, denied_symptoms, correlation_threshold=0.3):
        """Filter candidate symptoms based on denied symptoms using dataset correlations"""
        if not denied_symptoms:
            return candidate_symptoms
        
        correlations = self.analyze_symptom_correlations()
        filtered_symptoms = []
        
        for candidate in candidate_symptoms:
            should_exclude = False
            
            # Check if this candidate is highly correlated with any denied symptom
            for denied in denied_symptoms:
                if (candidate in correlations and 
                    denied in correlations[candidate] and 
                    correlations[candidate][denied] > correlation_threshold):
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_symptoms.append(candidate)
        
        return filtered_symptoms

    def generate_prediction_explanation(self, confirmed_symptoms, predicted_disease, confidence, top_3_diseases, top_3_confidences):
        """
        Generate a data-driven explanation for the prediction using feature importances and confirmed symptoms
        """
        try:
            # Get feature importances
            feature_importances = self._get_feature_importances()
            
            # Calculate symptom importance scores for confirmed symptoms
            symptom_importance_scores = []
            for symptom in confirmed_symptoms:
                if symptom in self.feature_names:
                    idx = self.feature_names.index(symptom)
                    importance = feature_importances[idx]
                    symptom_importance_scores.append((symptom, importance))
            
            # Sort by importance (highest first)
            symptom_importance_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 3 most influential symptoms
            top_influential_symptoms = [s[0] for s in symptom_importance_scores[:3]]
            
            # Create explanation text
            symptoms_text = ', '.join([self._format_symptom_name(s) for s in confirmed_symptoms])
            
            if top_influential_symptoms:
                explanation = f"Based on your symptoms: {symptoms_text}. "
                explanation += f"The most influential symptoms for this prediction were: {', '.join([self._format_symptom_name(s) for s in top_influential_symptoms])}."
            else:
                explanation = f"Based on your symptoms: {symptoms_text}."
            
            return explanation
            
        except Exception as e:
            # Fallback explanation if feature importance calculation fails
            symptoms_text = ', '.join([self._format_symptom_name(s) for s in confirmed_symptoms])
            return f"Based on your symptoms: {symptoms_text}."

    def display_prediction_with_explanation(self, predicted_disease, confidence, top_3_diseases, top_3_confidences, confirmed_symptoms):
        """
        Display prediction with explanation and show top-3 when confidence is low
        """
        conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"
        
        # Generate explanation
        explanation = self.generate_prediction_explanation(confirmed_symptoms, predicted_disease, confidence, top_3_diseases, top_3_confidences)
        
        # Display main prediction
        if predicted_disease:
            print(f"\n🏥 You may have: {predicted_disease}")
            print(f"🔬 Model confidence: {conf_str}")
            print(f"💡 {explanation}")
            
            # Show top-3 alternatives if confidence is low
            if confidence is not None and confidence < 0.6:
                print(f"\n🔍 Other possibilities (lower confidence):")
                for i, (disease, conf) in enumerate(zip(top_3_diseases[1:], top_3_confidences[1:]), 2):
                    if conf > 0.1:  # Only show if confidence > 10%
                        print(f"   {i}. {disease} ({conf*100:.1f}%)")
                
                print(f"\n💡 Tip: Consider adding more symptoms for a more accurate diagnosis.")
            
            if confidence is not None and confidence < 0.5:
                print("⚠️  Low confidence - please consult a healthcare professional immediately")
        else:
            print(f"\n⚠️  Unable to make a confident prediction. Please consult a healthcare professional.")

    def _format_symptom_name(self, symptom):
        """Convert symptom name from underscore format to user-friendly format"""
        if not symptom:
            return symptom
        
        # Replace underscores with spaces and capitalize properly
        return symptom.replace('_', ' ').title()

    def _format_symptom_list(self, symptoms):
        """Format a list of symptoms to user-friendly format"""
        if not symptoms:
            return []
        return [self._format_symptom_name(symptom) for symptom in symptoms]

    def _extract_rating_value(self, item, rating_key='rate'):
        """Extract and convert rating value from an item (doctor/hospital)"""
        rating = item.get(rating_key)
        if rating is None or rating == 'N/A' or rating == '':
            return 0.0  # Put items without rating at the end
        try:
            return float(rating)
        except (ValueError, TypeError):
            return 0.0

    def _get_numeric_input(self, prompt, min_val, max_val, allow_back=False):
        """Get validated numeric input within a range"""
        while True:
            try:
                choice = input(prompt).strip()
                if allow_back and choice.lower() == 'back':
                    return 'back'
                choice_num = int(choice)
                if min_val <= choice_num <= max_val:
                    return choice_num
                else:
                    print(f"❌ Please enter a number between {min_val} and {max_val}.")
            except EOFError:
                print("\n❌ No input available. Using default value.")
                return min_val
            except ValueError:
                print("❌ Please enter a valid number.")


