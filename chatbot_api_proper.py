from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from chatbot_interface import ChatbotInterface
from data_preprocessing_fixed import DataPreprocessorFixed
# ModelTrainer is defined locally in this file
import time
import difflib
from config import MODELS_DIR, TIME_SLOTS, APPOINTMENT_DAYS_AHEAD, BOOKING_ARRIVAL_MINUTES

app = FastAPI(title="Healthcare Chatbot API")

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the SAME chatbot as CLI
data_preprocessor = DataPreprocessorFixed()
data_preprocessor.initialize_all(use_augmented=False, use_ai_augmented=False, use_safe_augmented=True)

# Load the enhanced model
import joblib
import os

model_path = os.path.join(MODELS_DIR, "enhanced_ai_augmented_model.joblib")
encoder_path = os.path.join(MODELS_DIR, "enhanced_ai_augmented_label_encoder.joblib")

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("Enhanced model files not found. Please run main.py first to train the model.")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Create model trainer
class ModelTrainer:
    def __init__(self, model, label_encoder):
        self.clf = model
        self.label_encoder = label_encoder

model_trainer = ModelTrainer(model, label_encoder)

# Session storage
sessions = {}

class StartSessionRequest(BaseModel):
    pass

class StartSessionResponse(BaseModel):
    session_id: str
    prompt: str
    options: list
    state: str

class SendMessageRequest(BaseModel):
    session_id: str
    user_input: str

class SendMessageResponse(BaseModel):
    prompt: str
    options: list
    state: str

class APIChatbotInterface(ChatbotInterface):
    """Extended ChatbotInterface that supports API mode"""
    
    def __init__(self, data_preprocessor, model_trainer):
        super().__init__(data_preprocessor, model_trainer)
        self.api_mode = True
        self.current_state = "initializing"
        self.last_prompt = ""
        self.last_options = []
        # --- New for follow-up logic ---
        self.denied_symptoms = set()
        self.previously_asked_symptoms = set()
        self.follow_up_questions = []
        self.follow_up_round = 0
        self.max_follow_up_rounds = 2
        self.diagnosis_context = {}  # Store context for diagnosis session
        
    def api_get_info(self):
        """API version of getInfo - returns structured response"""
        self.current_state = "awaiting_name"
        self.last_prompt = "Your Name? ->"
        self.last_options = []
        return {
            'prompt': self.last_prompt,
            'options': self.last_options,
            'state': self.current_state
        }
    
    def api_handle_name(self, name):
        """API version of name handling"""
        if name.strip().lower() == "undo":
            return {
                'prompt': "Undo: Please enter your name again.\nYour Name? ->",
                'options': [],
                'state': "awaiting_name"
            }
        
        name = name.strip()
        if not name:
            self.name = "User"
            prompt = "Hello! 👋\nPlease enter your location (city or country):"
        else:
            self.name = name
            prompt = f"Hello {self.name}! 👋\nPlease enter your location (city or country):"
        
        self.current_state = "awaiting_location"
        return {
            'prompt': prompt,
            'options': [],
            'state': self.current_state
        }
    
    def api_handle_location(self, location):
        """API version of location handling"""
        if location.strip().lower() == "undo":
            return {
                'prompt': "Let's try entering your location again.\nPlease enter your location (city or country):",
                'options': [],
                'state': "awaiting_location"
            }
        
        location = location.strip()
        if not location:
            return {
                'prompt': "Location cannot be empty. Please try again.\nPlease enter your location (city or country):",
                'options': [],
                'state': "awaiting_location"
            }
        
        self.user_location = location
        self.current_state = "main_menu"
        return self.api_get_main_menu()
    
    def api_get_main_menu(self):
        """API version of main menu"""
        prompt = ("What would you like to do?\n"
                  "1) Diagnosis (disease prediction)\n"
                  "2) Find a doctor\n"
                  "3) Find a hospital\n"
                  "4) Book hospital appointment\n"
                  "5) I am done / Exit\n"
                  "(Type 'undo' to go back and enter your location.)\n"
                  "Enter 1, 2, 3, 4, or 5:")
        options = ["1", "2", "3", "4", "5", "undo"]
        return {
            'prompt': prompt,
            'options': options,
            'state': "main_menu"
        }
    
    def api_handle_main_menu(self, choice):
        """API version of main menu handling"""
        choice = choice.strip().lower()
        if choice == "undo":
            self.current_state = "awaiting_location"
            return {
                'prompt': "Please enter your location (city or country):",
                'options': [],
                'state': "awaiting_location"
            }
        if choice not in ["1", "2", "3", "4", "5"]:
            return {
                'prompt': "Invalid choice. Please enter 1, 2, 3, 4, or 5.\n" + self.api_get_main_menu()['prompt'],
                'options': ["1", "2", "3", "4", "5", "undo"],
                'state': "main_menu"
            }
        if choice == "1":
            self.current_state = "diagnosis_method"
            prompt = ("Please choose input method:\n"
                      "1) Traditional (one symptom at a time)\n"
                      "2) Free text (write all symptoms in one sentence)\n"
                      "Enter 1 or 2:")
            options = ["1", "2", "undo"]
        elif choice == "2":
            self.current_state = "doctor_search"
            prompt = "Enter the medical specialty you are looking for (e.g., Cardiology, Neurology, Dermatology):"
            options = ["undo"]
        elif choice == "3":
            self.current_state = "hospital_search"
            prompt = "Enter the city or country you are looking for hospitals in:"
            options = ["undo"]
        elif choice == "4":
            # If user_location is set, skip location prompt and go directly to booking_location handler
            if getattr(self, 'user_location', None):
                self.current_state = 'booking_location'
                return self.api_handle_booking_location(self.user_location)
            else:
                self.current_state = "booking_location"
                prompt = "Enter the city or country you are looking for hospitals in:"
                options = ["undo"]
                return {
                    'prompt': prompt,
                    'options': options,
                    'state': self.current_state
                }
        elif choice == "5":
            self.current_state = "exit_confirm"
            prompt = "Are you sure you want to exit? (yes/y or no/n):"
            options = ["yes", "y", "no", "n", "undo"]
        if choice not in ["4"]:
            return {
                'prompt': prompt,
                'options': options,
                'state': self.current_state
            }
    
    def api_handle_diagnosis_method(self, method):
        """API version of diagnosis method handling"""
        method = method.strip().lower()
        if method == "undo":
            return self.api_get_main_menu()
        
        if method not in ["1", "2"]:
            return {
                'prompt': ("Invalid choice. Please enter 1 or 2.\n"
                          "Please choose input method:\n"
                          "1) Traditional (one symptom at a time)\n"
                          "2) Free text (write all symptoms in one sentence)\n"
                          "Enter 1 or 2:"),
                'options': ["1", "2", "undo"],
                'state': "diagnosis_method"
            }
        
        self.diagnosis_method = "traditional" if method == "1" else "free_text"
        self.symptoms = []
        
        print(f"DEBUG: Method chosen: {method}")
        print(f"DEBUG: Diagnosis method: {self.diagnosis_method}")
        
        if method == "1":
            self.current_state = "diagnosis_symptom"
            prompt = "Enter the symptom you are experiencing ->"
            options = ["undo"]
        else:
            self.current_state = "diagnosis_free_text"
            prompt = ("Please write all the symptoms you are experiencing in one sentence "
                      "(e.g., I have headache and fever and muscle pain):")
            options = ["undo"]
        
        print(f"DEBUG: Current state set to: {self.current_state}")
        
        return {
            'prompt': prompt,
            'options': options,
            'state': self.current_state
        }
    
    def api_handle_symptom_input(self, symptom_input):
        print("DEBUG: api_handle_symptom_input called")
        """API version of symptom input handling"""
        if symptom_input.strip().lower() == "undo":
            self.current_state = "diagnosis_method"
            return {
                'prompt': "Please choose input method:\n1) Traditional (one symptom at a time)\n2) Free text (write all symptoms in one sentence)\nEnter 1 or 2:",
                'options': ["1", "2", "undo"],
                'state': self.current_state
            }
        # Use the EXACT same logic as CLI
        corrected = self.handle_single_symptom_input(symptom_input)
        if len(corrected) == 1:
            self.symptoms = corrected
            self.current_state = "diagnosis_review"
            prompt = f"Here are the symptoms I have: {', '.join(self.symptoms)}\nWould you like to change this symptom? (yes/y or no/n)\n-> "
            options = ["yes", "y", "no", "n", "undo"]
        elif len(corrected) > 1:
            self.current_state = "symptom_clarification"
            self.symptom_clarification_options = corrected  # Store options for clarification step
            prompt = (f"You entered '{symptom_input}'. Please specify the type(s):\n" +
                      '\n'.join([f"  {i+1}) {opt}" for i, opt in enumerate(corrected)]) +
                      "\n  0) None of these / skip\nSelect all that apply (comma-separated numbers, e.g. 1,3,5): ")
            options = [str(i+1) for i in range(len(corrected))] + ["0", "undo"]
        else:
            self.current_state = "diagnosis_symptom"
            prompt = "Symptom not recognized. Please try again.\nEnter the symptom you are experiencing ->"
            options = ["undo"]
        return {
            'prompt': prompt,
            'options': options,
            'state': self.current_state,
            'symptoms': corrected
        }
    
    def api_handle_diagnosis_review(self, choice):
        """API version of diagnosis review handling"""
        choice = choice.strip().lower()
        
        if choice in ["yes", "y"]:
            self.current_state = "diagnosis_symptom_edit"
            prompt = "Enter the new symptom (or type 'undo' to cancel):"
            options = ["undo"]
        elif choice in ["no", "n"]:
            self.current_state = "diagnosis_days"
            prompt = "Okay. For how many days have you had these symptoms? : "
            options = ["undo"]
        else:
            prompt = f"Please enter yes/y or no/n.\nHere are the symptoms I have: {', '.join(self.symptoms)}\nWould you like to change this symptom? (yes/y or no/n)\n-> "
            options = ["yes", "y", "no", "n"]
        
        return {
            'prompt': prompt,
            'options': options,
            'state': self.current_state
        }
    
    def api_handle_days_input(self, days_input):
        """API version of days input handling"""
        try:
            num_days = int(days_input)
            self.days = num_days
            self.current_state = "diagnosis_related"
            
            # Get related symptoms using EXACT same logic as CLI
            relevant_symptoms = self.get_medical_relevant_symptoms(self.symptoms[0])
            if relevant_symptoms:
                rels = [symptom for symptom in relevant_symptoms if symptom not in self.symptoms]
                if rels:
                    formatted_symptom = self._format_symptom_name(rels[0])
                    prompt = f"Are you experiencing any of these related symptoms?\n{formatted_symptom}? (yes/y or no/n or back): "
                    options = ["yes", "y", "no", "n", "back"]
                else:
                    return self.api_make_diagnosis()
            else:
                return self.api_make_diagnosis()
        except ValueError:
            prompt = "Please enter a valid number for days:"
            options = []
        
        return {
            'prompt': prompt,
            'options': options,
            'state': self.current_state
        }
    
    def api_make_diagnosis(self):
        """API version of diagnosis - uses EXACT same logic as CLI, now with follow-up escalation"""
        denied_symptoms = getattr(self, 'denied_symptoms', set())
        previously_asked_symptoms = getattr(self, 'previously_asked_symptoms', set())
        follow_up_round = getattr(self, 'follow_up_round', 0)
        max_follow_up_rounds = getattr(self, 'max_follow_up_rounds', 2)
        symptoms = self.symptoms

        # Run prediction
        predicted_disease, confidence, top_3_diseases, top_3_confidences, should_make_prediction, follow_up_questions, safety_warnings, risk_level, denied_symptoms = self.predict_disease_with_medical_validation(
            symptoms, denied_symptoms
        )
        conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Unknown"

        # Store context for follow-up
        self.diagnosis_context = {
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'top_3_diseases': top_3_diseases,
            'top_3_confidences': top_3_confidences,
            'should_make_prediction': should_make_prediction,
            'follow_up_questions': follow_up_questions,
            'safety_warnings': safety_warnings,
            'risk_level': risk_level,
            'denied_symptoms': denied_symptoms,
            'previously_asked_symptoms': previously_asked_symptoms,
            'follow_up_round': follow_up_round,
            'symptoms': symptoms.copy(),
        }

        # If follow-up questions are needed and we haven't hit the max rounds, ask them
        if follow_up_questions and follow_up_round < max_follow_up_rounds:
            self.current_state = "diagnosis_follow_up"
            self.follow_up_questions = follow_up_questions
            self.denied_symptoms = denied_symptoms
            self.previously_asked_symptoms = previously_asked_symptoms
            self.follow_up_round = follow_up_round
            # Ask the first follow-up question
            question = follow_up_questions[0]
            formatted = question.replace("Critical: ", "").replace("Important: ", "")
            prompt = f"I need more information for a reliable diagnosis.\nCurrent symptoms: {', '.join(self.symptoms)}\nPotential concern: {predicted_disease} ({conf_str} confidence)\n\nPlease answer:\n{question}? (yes/y or no/n)"
            return {
                'prompt': prompt,
                'options': ["yes", "y", "no", "n", "undo"],
                'state': self.current_state,
                'question': question,
                'symptoms': self.symptoms
            }

        # --- EMERGENCY LOGIC ---
        specialization = self.disease_to_specialty.get(predicted_disease, None)
        if specialization == "Emergency":
            user_location = getattr(self, 'user_location', None)
            emergency_number = self._get_emergency_numbers_by_location(user_location)
            filtered_hospitals, all_hospitals = self.get_hospitals_by_location(user_location) if user_location else ([], [])
            description = self.data_preprocessor.description_list.get(predicted_disease, "")
            precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
            # --- STRUCTURED DIAGNOSIS OBJECT ---
            diagnosis_obj = {
                'disease': predicted_disease,
                'confidence': confidence,
                'description': description,
                'precautions': precautions,
                'emergency': True,
                'emergency_number': emergency_number,
                'hospitals': filtered_hospitals,
                'top_3': list(zip(top_3_diseases, top_3_confidences)),
                'doctors': [],  # No doctor recommendations for emergency
                'risk_level': risk_level,
                'safety_warnings': safety_warnings,
                'symptoms': symptoms,
            }
            output = []
            output.append(f"\n🚨 EMERGENCY ALERT: {predicted_disease} detected!")
            output.append(f"⚠️  This is a medical emergency requiring immediate hospital care.")
            output.append(f"\n🩺 Diagnosis: {predicted_disease}")
            output.append(f"Confidence: {conf_str}")
            if description:
                output.append(f"\n📋 Description: {description}")
            if precautions:
                output.append("\n💊 Take the following precautions:")
                for i, precaution in enumerate(precautions, 1):
                    if precaution.strip():
                        output.append(f"   {i}) {precaution}")
            output.append(f"\n🏥 Recommending nearby hospitals for emergency treatment:")
            if filtered_hospitals:
                output.append(f"\n🏥 Emergency Hospitals in {user_location}:")
                for i, hosp in enumerate(filtered_hospitals[:5], 1):
                    hosp_lines = [
                        f"{i}. {hosp.get('name', 'Unknown')}",
                        f"   📍 Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}",
                        f"   📞 Emergency Phone: {hosp.get('phone', 'N/A')}",
                        f"   ⭐ Rating: {hosp.get('rate', 'N/A')}/5",
                        f"   🏗️  Established: {hosp.get('Established', 'N/A')}",
                    ]
                    if hosp.get('Url'):
                        hosp_lines.append(f"   🌐 Profile: {hosp.get('Url')}")
                    hosp_lines.append("---------------------------")
                    output.extend(hosp_lines)
            else:
                output.append(f"\n⚠️  No hospitals found in {user_location}")
            output.append(f"\n🚨 IMMEDIATE ACTION REQUIRED:")
            output.append(f"📞 Call emergency services: {emergency_number}")
            output.append("1. Go to the nearest hospital emergency department")
            output.append("2. Do not delay seeking medical attention")
            output.append("3. Bring someone with you if possible")
            output.append("\n⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
            main_menu_prompt = ("\n\nWhat would you like to do?\n"
                "1) Diagnosis (disease prediction)\n"
                "2) Find a doctor\n"
                "3) Find a hospital\n"
                "4) Book hospital appointment\n"
                "5) I am done / Exit\n"
                "(Type 'undo' to go back and enter your location.)\n"
                "Enter 1, 2, 3, 4, or 5:")
            full_prompt = '\n'.join(output) + main_menu_prompt
            self.current_state = "main_menu"
            self.denied_symptoms = set()
            self.previously_asked_symptoms = set()
            self.follow_up_questions = []
            self.follow_up_round = 0
            self.diagnosis_context = {}
            return {
                'prompt': full_prompt,
                'options': ["1", "2", "3", "4", "5", "undo"],
                'state': self.current_state,
                'diagnosis': diagnosis_obj
            }
        # --- END EMERGENCY LOGIC ---

        # Otherwise, return the diagnosis (non-emergency)
        description = self.data_preprocessor.description_list.get(predicted_disease, "")
        precautions = self.data_preprocessor.precautionDictionary.get(predicted_disease, [])
        doctor_recommendations = self._get_doctor_recommendations_data(predicted_disease)
        # --- STRUCTURED DIAGNOSIS OBJECT ---
        diagnosis_obj = {
            'disease': predicted_disease,
            'confidence': confidence,
            'description': description,
            'precautions': precautions,
            'emergency': False,
            'emergency_number': None,
            'hospitals': [],
            'top_3': list(zip(top_3_diseases, top_3_confidences)),
            'doctors': doctor_recommendations,
            'risk_level': risk_level,
            'safety_warnings': safety_warnings,
            'symptoms': symptoms,
        }
        output = []
        output.append(f"\n🩺 Diagnosis: {predicted_disease}")
        output.append(f"Confidence: {conf_str}")
        if description:
            output.append(f"\n📋 Description: {description}")
        if precautions:
            output.append("\n💊 Take the following precautions:")
            for i, precaution in enumerate(precautions, 1):
                if precaution.strip():
                    output.append(f"   {i+1}) {precaution}")
        output.append("\n⚠️  Disclaimer: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.")
        doctor_lines = []
        if doctor_recommendations:
            doctor_lines.append("\n🩺 Recommended Doctors:")
            for idx, doc in enumerate(doctor_recommendations, 1):
                line = f"{idx}. Name: {doc['name']}\n   Specialization: {doc['specialization']}\n   Location: {doc['city']}, {doc['country']}\n   Phone: {doc['phone']}\n   Rating: {doc['rate']}"
                if doc.get('gender'):
                    line += f"\n   Gender: {doc['gender']}"
                if doc.get('img_url'):
                    line += f"\n   Image: {doc['img_url']}"
                if doc.get('profile_url'):
                    line += f"\n   Profile: {doc['profile_url']}"
                doctor_lines.append(line)
        else:
            doctor_lines.append("\n⚠️  No doctor recommendations available for this diagnosis and location.")
        main_menu_prompt = ("\n\nWhat would you like to do?\n"
            "1) Diagnosis (disease prediction)\n"
            "2) Find a doctor\n"
            "3) Find a hospital\n"
            "4) Book hospital appointment\n"
            "5) I am done / Exit\n"
            "(Type 'undo' to go back and enter your location.)\n"
            "Enter 1, 2, 3, 4, or 5:")
        full_prompt = '\n'.join(output + doctor_lines) + main_menu_prompt
        self.current_state = "main_menu"
        self.denied_symptoms = set()
        self.previously_asked_symptoms = set()
        self.follow_up_questions = []
        self.follow_up_round = 0
        self.diagnosis_context = {}
        return {
            'prompt': full_prompt,
            'options': ["1", "2", "3", "4", "5", "undo"],
            'state': self.current_state,
            'diagnosis': diagnosis_obj
        }
    
    def api_handle_symptom_edit(self, symptom_input):
        """API version of symptom editing"""
        print(f"DEBUG: api_handle_symptom_edit called with input: '{symptom_input}'")
        print(f"DEBUG: Current symptoms: {self.symptoms}")
        
        if symptom_input.strip().lower() == "undo":
            self.current_state = "diagnosis_review"
            prompt = f"Edit cancelled. Keeping the original symptom.\nHere are the symptoms I have: {', '.join(self.symptoms)}\nWould you like to change this symptom? (yes/y or no/n)\n-> "
            return {
                'prompt': prompt,
                'options': ["yes", "y", "no", "n"],
                'state': self.current_state
            }
        
        # Use the EXACT same logic as CLI
        corrected = self.handle_single_symptom_input(symptom_input)
        print(f"DEBUG: handle_single_symptom_input returned: {corrected}")
        
        if len(corrected) == 1:
            self.symptoms = corrected
            self.current_state = "diagnosis_review"
            prompt = f"Here are the symptoms I have: {', '.join(self.symptoms)}\nWould you like to change this symptom? (yes/y or no/n)\n-> "
            return {
                'prompt': prompt,
                'options': ["yes", "y", "no", "n"],
                'state': self.current_state
            }
        elif len(corrected) > 1:
            self.current_state = "symptom_clarification"
            prompt = (f"You entered '{symptom_input}'. Please specify the type(s):\n" +
                      '\n'.join([f"  {i+1}) {opt}" for i, opt in enumerate(corrected)]) +
                      "\n  0) None of these / skip\nSelect all that apply (comma-separated numbers, e.g. 1,3,5): ")
            options = [str(i+1) for i in range(len(corrected))] + ["0"]
            return {
                'prompt': prompt,
                'options': options,
                'state': self.current_state
            }
        else:
            # Keep the same state but provide clear error message
            prompt = "Symptom not recognized. Please try again.\nEnter the new symptom (or type 'undo' to cancel):"
            return {
                'prompt': prompt,
                'options': ["undo"],
                'state': "diagnosis_symptom_edit"  # Explicitly set the state
            }
    
    def api_handle_free_text_input(self, user_text):
        """API version of free text symptom input handling"""
        # Tokenize and clean input
        leading_phrases = [
            r'^i have ', r'^i am suffering from ', r'^i am having ', r'^i feel ', r'^i am ', r"^i'm ", r'^i got ', r'^i\s+',
            r'^my ', r'^having ', r'^suffering from ', r'^experiencing ', r'^with ', r'^and ', r'^, ', r'^\s+'
        ]
        import re
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
        # Now, for each token, check for ambiguity
        self.free_text_tokens = cleaned_tokens.copy()  # Store for clarification
        self.free_text_symptom_results = []  # Will store (token, options) pairs
        self.free_text_selected_symptoms = []  # Final selected symptoms
        for token in cleaned_tokens:
            options = self.handle_single_symptom_input(token)
            if len(options) > 1:
                # Ambiguous, need clarification
                self.current_state = 'free_text_symptom_clarification'
                self.free_text_clarify_token = token
                self.free_text_clarify_options = options
                prompt = (f"You entered '{token}'. Please specify the type(s):\n" +
                          '\n'.join([f"  {i+1}) {opt}" for i, opt in enumerate(options)]) +
                          "\n  0) None of these / skip\nSelect all that apply (comma-separated numbers, e.g. 1,3,5): ")
                options_list = [str(i+1) for i in range(len(options))] + ["0", "undo"]
                return {
                    'prompt': prompt,
                    'options': options_list,
                    'state': self.current_state
                }
            elif len(options) == 1:
                self.free_text_selected_symptoms.append(options[0])
            # else: skip unrecognized
        # If no ambiguities, proceed to review
        self.symptoms = self.free_text_selected_symptoms
        self.current_state = "free_text_review"
        prompt = f"\nHere are the symptoms I have: {', '.join([self._format_symptom_name(s) for s in self.symptoms])}\nWhat would you like to do?\n1) Add a symptom\n2) Remove a symptom\n3) Edit all symptoms (re-enter full list)\n4) Continue with current symptoms"
        return {
            'prompt': prompt,
            'options': ["1", "2", "3", "4", "undo"],
            'state': self.current_state
        }

    def api_handle_free_text_symptom_clarification(self, user_input):
        """Handle clarification for ambiguous symptoms in free text mode"""
        user_input = user_input.strip()
        if user_input.lower() == "undo":
            self.current_state = "free_text_review"
            prompt = f"\nHere are the symptoms I have: {', '.join([self._format_symptom_name(s) for s in self.free_text_selected_symptoms])}\nWhat would you like to do?\n1) Add a symptom\n2) Remove a symptom\n3) Edit all symptoms (re-enter full list)\n4) Continue with current symptoms"
            return {
                'prompt': prompt,
                'options': ["1", "2", "3", "4", "undo"],
                'state': self.current_state
            }
        selected = [x.strip() for x in user_input.split(',') if x.strip().isdigit()]
        indices = [int(x) for x in selected if x != "0"]
        options = getattr(self, 'free_text_clarify_options', [])
        if options and indices:
            for idx in indices:
                if 1 <= idx <= len(options):
                    self.free_text_selected_symptoms.append(options[idx-1])
        # Remove the clarified token from the list and continue with the rest
        if hasattr(self, 'free_text_tokens') and self.free_text_clarify_token in self.free_text_tokens:
            self.free_text_tokens.remove(self.free_text_clarify_token)
        # Now process the next token, if any
        while self.free_text_tokens:
            token = self.free_text_tokens.pop(0)
            opts = self.handle_single_symptom_input(token)
            if len(opts) > 1:
                # Need clarification for this token
                self.current_state = 'free_text_symptom_clarification'
                self.free_text_clarify_token = token
                self.free_text_clarify_options = opts
                prompt = (f"You entered '{token}'. Please specify the type(s):\n" +
                          '\n'.join([f"  {i+1}) {opt}" for i, opt in enumerate(opts)]) +
                          "\n  0) None of these / skip\nSelect all that apply (comma-separated numbers, e.g. 1,3,5): ")
                options_list = [str(i+1) for i in range(len(opts))] + ["0", "undo"]
                return {
                    'prompt': prompt,
                    'options': options_list,
                    'state': self.current_state
                }
            elif len(opts) == 1:
                self.free_text_selected_symptoms.append(opts[0])
            # else: skip unrecognized
        # If no more tokens, proceed to review
        self.symptoms = self.free_text_selected_symptoms
        self.current_state = "free_text_review"
        prompt = f"\nHere are the symptoms I have: {', '.join([self._format_symptom_name(s) for s in self.symptoms])}\nWhat would you like to do?\n1) Add a symptom\n2) Remove a symptom\n3) Edit all symptoms (re-enter full list)\n4) Continue with current symptoms"
        return {
            'prompt': prompt,
            'options': ["1", "2", "3", "4", "undo"],
            'state': self.current_state
        }
    
    def api_handle_free_text_review(self, choice):
        """API version of free text review menu handling"""
        if choice == "1":
            self.current_state = "free_text_add_symptom"
            return {
                'prompt': "Enter the symptom you want to add:",
                'options': ["undo"],
                'state': self.current_state
            }
        elif choice == "2":
            if len(self.symptoms) == 1:
                return {
                    'prompt': "You only have one symptom. You cannot remove it.\n" + self.api_get_free_text_review_menu()['prompt'],
                    'options': ["1", "2", "3", "4", "undo"],
                    'state': self.current_state
                }
            self.current_state = "free_text_remove_symptom"
            formatted_symptoms = [f"{i}) {self._format_symptom_name(s)}" for i, s in enumerate(self.symptoms, 1)]
            prompt = "Which symptom would you like to remove?\n" + "\n".join(formatted_symptoms)
            return {
                'prompt': prompt,
                'options': [str(i) for i in range(1, len(self.symptoms) + 1)] + ["undo"],
                'state': self.current_state
            }
        elif choice == "3":
            self.current_state = "free_text_edit_all"
            return {
                'prompt': "Enter the full, final list of symptoms separated by commas (or type 'undo' to go back):",
                'options': ["undo"],
                'state': self.current_state
            }
        elif choice == "4":
            self.current_state = "diagnosis_days"
            return {
                'prompt': "Okay. For how many days have you had these symptoms? : ",
                'options': ["undo"],
                'state': self.current_state
            }
        else:
            return {
                'prompt': "Invalid choice. Please enter 1, 2, 3, or 4.\n" + self.api_get_free_text_review_menu()['prompt'],
                'options': ["1", "2", "3", "4", "undo"],
                'state': self.current_state
            }
    
    def api_handle_free_text_add_symptom(self, new_symptom):
        """Handle adding a symptom in free text review mode, with clarification if ambiguous."""
        new_symptom = new_symptom.strip()
        if new_symptom.lower() == "undo":
            self.current_state = "free_text_review"
            return self.api_get_free_text_review_menu()
        # Tokenize and clean input
        import re
        leading_phrases = [
            r'^i have ', r'^i am suffering from ', r'^i am having ', r'^i feel ', r'^i am ', r"^i'm ", r'^i got ', r'^i\s+',
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
        # For each token, check for ambiguity
        for token in cleaned_tokens:
            options = self.handle_single_symptom_input(token)
            if len(options) > 1:
                # Ambiguous, need clarification
                self.current_state = 'free_text_symptom_clarification'
                self.free_text_clarify_token = token
                self.free_text_clarify_options = options
                prompt = (f"You entered '{token}'. Please specify the type(s):\n" +
                          '\n'.join([f"  {i+1}) {opt}" for i, opt in enumerate(options)]) +
                          "\n  0) None of these / skip\nSelect all that apply (comma-separated numbers, e.g. 1,3,5): ")
                options_list = [str(i+1) for i in range(len(options))] + ["0", "undo"]
                return {
                    'prompt': prompt,
                    'options': options_list,
                    'state': self.current_state
                }
            elif len(options) == 1:
                if options[0] not in self.symptoms:
                    self.symptoms.append(options[0])
                    added = options[0]
                else:
                    added = None
            else:
                added = None
        # If all tokens processed and no ambiguity, show added message and review
        added_names = [self._format_symptom_name(s) for s in self.symptoms]
        prompt = f"Added: {', '.join(added_names)}\n\nHere are the symptoms I have: {', '.join(added_names)}\nWhat would you like to do?\n1) Add a symptom\n2) Remove a symptom\n3) Edit all symptoms (re-enter full list)\n4) Continue with current symptoms"
        self.current_state = "free_text_review"
        return {
            'prompt': prompt,
            'options': ["1", "2", "3", "4", "undo"],
            'state': self.current_state
        }
    
    def api_handle_free_text_remove_symptom(self, choice):
        """API version of removing symptom in free text mode"""
        try:
            remove_choice = int(choice)
            if 1 <= remove_choice <= len(self.symptoms):
                removed_symptom = self.symptoms.pop(remove_choice - 1)
                formatted_removed = self._format_symptom_name(removed_symptom)
                prompt = f"Removed: {formatted_removed}\n" + self.api_get_free_text_review_menu()['prompt']
                self.current_state = "free_text_review"
                return {
                    'prompt': prompt,
                    'options': ["1", "2", "3", "4"],
                    'state': self.current_state,
                    'symptoms': self.symptoms
                }
            else:
                return {
                    'prompt': "Invalid choice. Please try again.\n" + self.api_get_free_text_review_menu()['prompt'],
                    'options': ["1", "2", "3", "4"],
                    'state': self.current_state
                }
        except ValueError:
            return {
                'prompt': "Please enter a valid number.\n" + self.api_get_free_text_review_menu()['prompt'],
                'options': ["1", "2", "3", "4"],
                'state': self.current_state
            }
    
    def api_handle_free_text_edit_all(self, final_symptoms):
        """API version of editing all symptoms in free text mode"""
        if final_symptoms.strip().lower() == "undo":
            self.current_state = "free_text_review"
            return self.api_get_free_text_review_menu()
        
        # Use the EXACT same logic as CLI
        import re
        leading_phrases = [
            r'^i have ', r'^i am suffering from ', r'^i am having ', r'^i feel ', r'^i am ', r'^i\'m ', r'^i got ', r'^i\s+',
            r'^my ', r'^having ', r'^suffering from ', r'^experiencing ', r'^with ', r'^and ', r'^, ', r'^\s+'
        ]
        tokens = re.split(r',| and | و | و|,|\band\b', final_symptoms)
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
            corrected_tokens = self.handle_single_symptom_input(token)
            validated.extend(corrected_tokens)
        
        if validated:
            self.symptoms = validated
            prompt = f"Symptoms updated.\n" + self.api_get_free_text_review_menu()['prompt']
        else:
            prompt = "No valid symptoms entered. Please try again.\n" + self.api_get_free_text_review_menu()['prompt']
        
        self.current_state = "free_text_review"
        return {
            'prompt': prompt,
            'options': ["1", "2", "3", "4", "undo"],
            'state': self.current_state,
            'symptoms': self.symptoms
        }
    
    def api_handle_related_symptoms(self, user_input):
        """API version of related symptoms handling"""
        user_input = user_input.strip().lower()
        
        # Initialize tracking if not already
        if not hasattr(self, 'related_symptom_idx'):
            self.related_symptom_idx = 0
            self.denied_symptoms = set()
            self.all_symptoms = list(self.symptoms)
            self.related_symptoms_list = []
            
            # Build the list of (main_symptom, related_symptom) pairs to ask
            for main_symptom in self.symptoms:
                related = self.get_medical_relevant_symptoms(main_symptom)
                for rel in related:
                    if rel not in self.symptoms and rel not in self.related_symptoms_list:
                        self.related_symptoms_list.append((main_symptom, rel))
        
        # If no related symptoms to ask, proceed to diagnosis
        if not self.related_symptoms_list or self.related_symptom_idx >= len(self.related_symptoms_list):
            # Clean up
            delattr(self, 'related_symptom_idx')
            delattr(self, 'related_symptoms_list')
            delattr(self, 'denied_symptoms')
            delattr(self, 'all_symptoms')
            return self.api_make_diagnosis()
        
        idx = self.related_symptom_idx
        main_symptom, rel_symptom = self.related_symptoms_list[idx]
        formatted = rel_symptom.replace('_', ' ')
        
        if user_input == 'back':
            if idx > 0:
                self.related_symptom_idx -= 1
                idx = self.related_symptom_idx
                main_symptom, rel_symptom = self.related_symptoms_list[idx]
                formatted = rel_symptom.replace('_', ' ')
                prompt = f"{formatted}? (yes/y or no/n or back)"
                return {
                    'prompt': prompt,
                    'options': ["yes", "y", "no", "n", "back"],
                    'state': self.current_state
                }
            else:
                prompt = 'Already at the first related symptom.'
                return {
                    'prompt': prompt,
                    'options': ["yes", "y", "no", "n", "back"],
                    'state': self.current_state
                }
        
        if user_input in ['yes', 'y']:
            if rel_symptom not in self.all_symptoms:
                self.all_symptoms.append(rel_symptom)
            self.related_symptom_idx += 1
        elif user_input in ['no', 'n']:
            self.denied_symptoms.add(rel_symptom)
            self.related_symptom_idx += 1
        else:
            prompt = f"Please enter yes/y, no/n, or back.\n{formatted}? (yes/y or no/n or back)"
            return {
                'prompt': prompt,
                'options': ["yes", "y", "no", "n", "back"],
                'state': self.current_state
            }
        
        # Move to next related symptom or finish
        if self.related_symptom_idx < len(self.related_symptoms_list):
            next_main, next_rel = self.related_symptoms_list[self.related_symptom_idx]
            formatted = next_rel.replace('_', ' ')
            prompt = f"{formatted}? (yes/y or no/n or back)"
            return {
                'prompt': prompt,
                'options': ["yes", "y", "no", "n", "back"],
                'state': self.current_state
            }
        else:
            # All related symptoms processed
            self.symptoms = self.all_symptoms
            # Clean up
            delattr(self, 'related_symptom_idx')
            delattr(self, 'related_symptoms_list')
            delattr(self, 'denied_symptoms')
            delattr(self, 'all_symptoms')
            return self.api_make_diagnosis()
    
    def api_handle_symptom_clarification(self, user_input):
        """API version of symptom clarification"""
        user_input = user_input.strip()
        selected = [x.strip() for x in user_input.split(',') if x.strip().isdigit()]
        indices = [int(x) for x in selected if x != "0"]
        options = getattr(self, 'symptom_clarification_options', [])
        chosen = []
        if options and indices:
            for idx in indices:
                if 1 <= idx <= len(options):
                    chosen.append(options[idx-1])
        if chosen:
            self.symptoms = chosen
            self.current_state = "diagnosis_review"
            prompt = f"Here are the symptoms I have: {', '.join(self.symptoms)}\nWould you like to change this symptom? (yes/y or no/n)\n-> "
            return {
                'prompt': prompt,
                'options': ["yes", "y", "no", "n", "undo"],
                'state': self.current_state
            }
        else:
            self.current_state = "diagnosis_symptom"
            prompt = "Enter the symptom you are experiencing ->"
            return {
                'prompt': prompt,
                'options': [],
                'state': self.current_state
            }
    
    def api_handle_doctor_search(self, user_input):
        """API version of doctor search with fuzzy specialty correction"""
        if user_input.strip().lower() == "undo":
            self.current_state = "awaiting_location"
            return {
                'prompt': "Please enter your location (city or country):",
                'options': [],
                'state': self.current_state
            }
        specialty = user_input.strip()
        location = getattr(self, 'location', 'your area')
        doctors = self.get_doctors_by_specialization(specialty)
        location_doctors = [doc for doc in doctors if doc.get('location', '').strip().lower() == str(location).strip().lower()]
        output = []
        # Fuzzy correction if no doctors found
        if not doctors:
            # Get all unique specializations from all doctors
            all_doctors = self.get_doctors_by_specialization("")
            all_specialties = set(doc.get('specialization', '').strip() for doc in all_doctors if doc.get('specialization'))
            close_matches = difflib.get_close_matches(specialty, all_specialties, n=1, cutoff=0.7)
            if close_matches:
                corrected = close_matches[0]
                doctors = self.get_doctors_by_specialization(corrected)
                location_doctors = [doc for doc in doctors if doc.get('location', '').strip().lower() == str(location).strip().lower()]
                output.append(f"Did you mean: '{corrected}'? Showing results for '{corrected}'.")
                specialty = corrected
            else:
                output.append(f"No doctors found for specialty '{specialty}' in any location.")
                output.append("\nDo you want to search for another doctor? (yes/y or no/n):")
                self.current_state = "doctor_search_again"
                return {
                    'prompt': '\n'.join(output),
                    'options': ["yes", "y", "no", "n", "undo"],
                    'state': self.current_state
                }
        if location_doctors:
            output.append(f"\nDoctors specializing in {specialty} in {location}:")
            for doc in location_doctors:
                output.append(
                    f"   - Name: {doc.get('name', 'Unknown')}\n"
                    f"     Specialty: {doc.get('specialization', 'N/A')}\n"
                    f"     Location: {doc.get('city', 'N/A')}, {doc.get('country', 'N/A')}\n"
                    f"     Phone: {doc.get('phone', 'N/A')}\n"
                    f"     Rating: {doc.get('rate', 'N/A')}/5\n"
                    f"     Gender: {doc.get('gender', 'N/A')}\n"
                    f"     Image: {doc.get('ImgUrl', 'N/A')}\n"
                    f"     Profile: {doc.get('Url', 'N/A')}\n"
                )
        else:
            output.append(f"No doctors found for specialty '{specialty}' in {location}.")
            if doctors:
                output.append(f"\nDoctors specializing in {specialty} in other locations:")
                for doc in doctors:
                    output.append(
                        f"   - Name: {doc.get('name', 'Unknown')}\n"
                        f"     Specialty: {doc.get('specialization', 'N/A')}\n"
                        f"     Location: {doc.get('city', 'N/A')}, {doc.get('country', 'N/A')}\n"
                        f"     Phone: {doc.get('phone', 'N/A')}\n"
                        f"     Rating: {doc.get('rate', 'N/A')}/5\n"
                        f"     Gender: {doc.get('gender', 'N/A')}\n"
                        f"     Image: {doc.get('ImgUrl', 'N/A')}\n"
                        f"     Profile: {doc.get('Url', 'N/A')}\n"
                    )
            else:
                output.append(f"No doctors found for specialty '{specialty}' in any location.")
        output.append("\nDo you want to search for another doctor? (yes/y or no/n):")
        self.current_state = "doctor_search_again"
        return {
            'prompt': '\n'.join(output),
            'options': ["yes", "y", "no", "n", "undo"],
            'state': self.current_state
        }
    
    def api_handle_doctor_search_again(self, user_input):
        """API version of doctor search again"""
        user_input = user_input.strip().lower()
        
        if user_input in ["yes", "y"]:
            self.current_state = "doctor_search"
            return {
                'prompt': "Enter the medical specialty you are looking for (e.g., Cardiology, Neurology, Dermatology):",
                'options': ["undo"],
                'state': self.current_state
            }
        elif user_input in ["no", "n"]:
            self.current_state = "main_menu"
            return self.api_get_main_menu()
        elif user_input == "undo":
            self.current_state = "main_menu"
            return self.api_get_main_menu()
        else:
            return {
                'prompt': "Please enter yes/y or no/n.\nDo you want to search for another doctor? (yes/y or no/n):",
                'options': ["yes", "y", "no", "n", "undo"],
                'state': self.current_state
            }
    
    def api_handle_hospital_search(self, user_input):
        """API version of hospital search"""
        if user_input.strip().lower() == "undo":
            self.current_state = "awaiting_location"
            return {
                'prompt': "Please enter your location (city or country):",
                'options': [],
                'state': self.current_state
            }
        
        # Simulate hospital lookup (same as CLI)
        hosp_location = user_input.strip()
        
        # Use the EXACT same logic as CLI
        filtered_hospitals, all_hospitals = self.get_hospitals_by_location(hosp_location)
        
        output = []
        if filtered_hospitals:
            output.append(f"\nHospitals in {hosp_location}:")
            for hosp in filtered_hospitals[:5]:  # Show top 5 hospitals
                output.append(
                    f"- Name: {hosp.get('name', 'Unknown')}\n"
                    f"   \U0001F4CD Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}\n"
                    f"   \U0001F4DE Emergency Phone: {hosp.get('phone', 'N/A')}\n"
                    f"   \u2B50 Rating: {hosp.get('rate', 'N/A')}/5\n"
                    f"   \U0001F3D7\uFE0F  Established: {hosp.get('Established', 'N/A')}\n"
                    f"   \U0001F310 Profile: {hosp.get('Url', 'N/A')}\n"
                    f"---------------------------"
                )
        else:
            output.append(f"No hospitals found in {hosp_location}.")
            if all_hospitals:
                output.append(f"\nAvailable hospitals in other locations:")
                for hosp in all_hospitals:
                    output.append(
                        f"- Name: {hosp.get('name', 'Unknown')}\n"
                        f"   \U0001F4CD Location: {hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')}\n"
                        f"   \U0001F4DE Emergency Phone: {hosp.get('phone', 'N/A')}\n"
                        f"   \u2B50 Rating: {hosp.get('rate', 'N/A')}/5\n"
                        f"   \U0001F3D7\uFE0F  Established: {hosp.get('Established', 'N/A')}\n"
                        f"   \U0001F310 Profile: {hosp.get('Url', 'N/A')}\n"
                        f"---------------------------"
                    )
            else:
                output.append(f"No hospitals found in any location.")
        
        output.append("\nDo you want to search for another hospital? (yes/y or no/n):")
        
        self.current_state = "hospital_search_again"
        return {
            'prompt': '\n'.join(output),
            'options': ["yes", "y", "no", "n", "undo"],
            'state': self.current_state
        }
    
    def api_handle_hospital_search_again(self, user_input):
        """API version of hospital search again"""
        user_input = user_input.strip().lower()
        
        if user_input in ["yes", "y"]:
            self.current_state = "hospital_search"
            return {
                'prompt': "Enter the city or country you are looking for hospitals in:",
                'options': ["undo"],
                'state': self.current_state
            }
        elif user_input in ["no", "n"]:
            self.current_state = "main_menu"
            return self.api_get_main_menu()
        elif user_input == "undo":
            self.current_state = "main_menu"
            return self.api_get_main_menu()
        else:
            return {
                'prompt': "Please enter yes/y or no/n.\nDo you want to search for another hospital? (yes/y or no/n):",
                'options': ["yes", "y", "no", "n", "undo"],
                'state': self.current_state
            }
    
    def api_handle_booking_menu(self, user_input):
        """API version of booking menu - step-by-step booking flow"""
        # This method is now only used for routing, not for the initial prompt
        if not hasattr(self, 'booking_context') or self.current_state != 'booking_menu':
            self.booking_context = {}
            self.current_state = 'booking_location'
            # Do not return a prompt here; handled in main menu
        # Route to the correct booking step
        state = self.current_state
        if state == 'booking_location':
            return self.api_handle_booking_location(user_input)
        elif state == 'booking_hospital_select':
            return self.api_handle_booking_hospital_select(user_input)
        elif state == 'booking_patient_name':
            return self.api_handle_booking_patient_name(user_input)
        elif state == 'booking_patient_phone':
            return self.api_handle_booking_patient_phone(user_input)
        elif state == 'booking_date_select':
            return self.api_handle_booking_date_select(user_input)
        elif state == 'booking_time_select':
            return self.api_handle_booking_time_select(user_input)
        elif state == 'booking_confirm':
            return self.api_handle_booking_confirm(user_input)
        elif state == 'booking_result':
            return self.api_handle_booking_result()
        else:
            # Reset if unknown state
            self.current_state = 'main_menu'
            return self.api_get_main_menu()

    def api_handle_booking_location(self, user_input):
        if not hasattr(self, 'booking_context'):
            self.booking_context = {}
        # If user_location is set and user_input matches it, use it directly
        if getattr(self, 'user_location', None) and (not user_input or user_input.strip().lower() == self.user_location.strip().lower()):
            location = self.user_location
        else:
            # Only allow undo at the very first booking location prompt (if CLI does)
            location = user_input.strip()
            if not location:
                self.current_state = 'main_menu'
                return self.api_get_main_menu()
            if location:
                self.user_location = location
        self.booking_context['location'] = location
        filtered_hospitals, all_hospitals = self.get_hospitals_by_location(location)
        self.booking_context['filtered_hospitals'] = filtered_hospitals
        self.booking_context['all_hospitals'] = all_hospitals
        if not filtered_hospitals:
            prompt = f"No hospitals found in {location}.\n"
            if all_hospitals:
                prompt += "Available hospitals in other locations:\n"
                for hosp in all_hospitals:
                    prompt += f"- {hosp.get('name', 'Unknown')} ({hosp.get('city', 'N/A')}, {hosp.get('country', 'N/A')})\n"
            else:
                prompt += "No hospitals found in any location."
            prompt += "\nEnter another city/country."
            return {
                'prompt': prompt,
                'options': [],
                'state': 'booking_location'
            }
        # Show hospitals
        prompt = f"Found {len(filtered_hospitals)} hospitals in {location}:\n"
        for i, hospital in enumerate(filtered_hospitals, 1):
            name = hospital.get('name', 'Unknown')
            city = hospital.get('city', 'Unknown')
            country = hospital.get('country', 'Unknown')
            rate = hospital.get('rate')
            phone = hospital.get('phone', 'N/A')
            established = hospital.get('Established', 'N/A')
            prompt += f"{i}. {name}\n   📍 {city}, {country}\n   ⭐ {rate if rate is not None else 'N/A'}/5\n   📞 {phone}\n   🏗️  {established}\n\n"
        prompt += f"Select hospital (1-{len(filtered_hospitals)}) or 'undo' to return:"
        self.current_state = 'booking_hospital_select'
        return {
            'prompt': prompt,
            'options': [str(i) for i in range(1, len(filtered_hospitals)+1)] + ['undo'],
            'state': 'booking_hospital_select'
        }

    def api_handle_booking_hospital_select(self, user_input):
        if user_input.strip().lower() == 'undo':
            self.current_state = 'booking_location'
            return {
                'prompt': 'Enter the city or country you are looking for hospitals in:',
                'options': ['undo'],
                'state': 'booking_location'
            }
        try:
            idx = int(user_input.strip()) - 1
            hospitals = self.booking_context['filtered_hospitals']
            if idx < 0 or idx >= len(hospitals):
                raise ValueError
        except Exception:
            return {
                'prompt': f"Invalid selection. Please enter a number between 1 and {len(self.booking_context['filtered_hospitals'])} or 'undo':",
                'options': [str(i) for i in range(1, len(self.booking_context['filtered_hospitals'])+1)] + ['undo'],
                'state': 'booking_hospital_select'
            }
        self.booking_context['selected_hospital'] = self.booking_context['filtered_hospitals'][idx]
        self.current_state = 'booking_patient_name'
        return {
            'prompt': 'Enter your full name:',
            'options': ['undo'],
            'state': 'booking_patient_name'
        }

    def api_handle_booking_patient_name(self, user_input):
        if user_input.strip().lower() == 'undo':
            self.current_state = 'booking_hospital_select'
            return self.api_handle_booking_location(self.booking_context['location'])
        name = user_input.strip()
        if not name:
            return {
                'prompt': 'Name is required. Enter your full name:',
                'options': ['undo'],
                'state': 'booking_patient_name'
            }
        self.booking_context['patient_name'] = name
        self.current_state = 'booking_patient_phone'
        return {
            'prompt': 'Enter your phone number:',
            'options': ['undo'],
            'state': 'booking_patient_phone'
        }

    def api_handle_booking_patient_phone(self, user_input):
        if user_input.strip().lower() == 'undo':
            self.current_state = 'booking_patient_name'
            return {
                'prompt': 'Enter your full name:',
                'options': ['undo'],
                'state': 'booking_patient_name'
            }
        phone = user_input.strip()
        if not phone:
            return {
                'prompt': 'Phone number is required. Enter your phone number:',
                'options': ['undo'],
                'state': 'booking_patient_phone'
            }
        self.booking_context['patient_phone'] = phone
        self.current_state = 'booking_date_select'
        # Show available dates
        from datetime import datetime, timedelta
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        dates = []
        prompt = '\n📅 Available dates (starting from tomorrow):\n'
        for i in range(APPOINTMENT_DAYS_AHEAD):
            date = tomorrow + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
            prompt += f"  {i+1}. {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})\n"
        self.booking_context['available_dates'] = dates
        prompt += f"Select date (1-{APPOINTMENT_DAYS_AHEAD}):"
        return {
            'prompt': prompt,
            'options': [str(i+1) for i in range(APPOINTMENT_DAYS_AHEAD)] + ['undo'],
            'state': 'booking_date_select'
        }

    def api_handle_booking_date_select(self, user_input):
        if user_input.strip().lower() == 'undo':
            self.current_state = 'booking_patient_phone'
            return {
                'prompt': 'Enter your phone number:',
                'options': ['undo'],
                'state': 'booking_patient_phone'
            }
        try:
            idx = int(user_input.strip()) - 1
            dates = self.booking_context['available_dates']
            if idx < 0 or idx >= len(dates):
                raise ValueError
        except Exception:
            return {
                'prompt': f"Invalid selection. Please enter a number between 1 and {len(self.booking_context['available_dates'])} or 'undo':",
                'options': [str(i+1) for i in range(len(self.booking_context['available_dates']))] + ['undo'],
                'state': 'booking_date_select'
            }
        self.booking_context['appointment_date'] = self.booking_context['available_dates'][idx]
        self.current_state = 'booking_time_select'
        # Show available time slots
        prompt = '\n🕐 Available time slots:\n'
        for i, time_slot in enumerate(TIME_SLOTS, 1):
            prompt += f"  {i}. {time_slot}\n"
        prompt += f"Select time (1-{len(TIME_SLOTS)}):"
        return {
            'prompt': prompt,
            'options': [str(i+1) for i in range(len(TIME_SLOTS))] + ['undo'],
            'state': 'booking_time_select'
        }

    def api_handle_booking_time_select(self, user_input):
        if user_input.strip().lower() == 'undo':
            self.current_state = 'booking_date_select'
            return self.api_handle_booking_date_select('')
        try:
            idx = int(user_input.strip()) - 1
            if idx < 0 or idx >= len(TIME_SLOTS):
                raise ValueError
        except Exception:
            return {
                'prompt': f"Invalid selection. Please enter a number between 1 and {len(TIME_SLOTS)} or 'undo':",
                'options': [str(i+1) for i in range(len(TIME_SLOTS))] + ['undo'],
                'state': 'booking_time_select'
            }
        self.booking_context['appointment_time'] = TIME_SLOTS[idx]
        self.current_state = 'booking_confirm'
        # Show summary
        hosp = self.booking_context['selected_hospital']
        prompt = (f"\n📋 Booking Summary:\n"
                  f"  Hospital: {hosp.get('name', 'Unknown')}\n"
                  f"  Patient: {self.booking_context['patient_name']}\n"
                  f"  Phone: {self.booking_context['patient_phone']}\n"
                  f"  Date: {self.booking_context['appointment_date']}\n"
                  f"  Time: {self.booking_context['appointment_time']}\n"
                  "\nConfirm booking? (yes/no):")
        return {
            'prompt': prompt,
            'options': ['yes', 'y', 'no', 'n', 'undo'],
            'state': 'booking_confirm'
        }

    def api_handle_booking_confirm(self, user_input):
        if user_input.strip().lower() == 'undo':
            self.current_state = 'booking_time_select'
            return self.api_handle_booking_time_select('')
        if user_input.strip().lower() in ['yes', 'y']:
            # Process booking
            hosp = self.booking_context['selected_hospital']
            booking_result = self.book_hospital_appointment(
                hospital_id=hosp.get('_id', 'unknown'),
                patient_name=self.booking_context['patient_name'],
                patient_phone=self.booking_context['patient_phone'],
                appointment_date=self.booking_context['appointment_date'],
                appointment_time=self.booking_context['appointment_time'],
                symptoms=None
            )
            self.save_booking_to_file(booking_result)
            self.booking_context['booking_result'] = booking_result
            self.current_state = 'booking_result'
            return self.api_handle_booking_result()
        else:
            self.current_state = 'main_menu'
            main_menu = ("\nWhat would you like to do?\n"
                "1) Diagnosis (disease prediction)\n"
                "2) Find a doctor\n"
                "3) Find a hospital\n"
                "4) Book hospital appointment\n"
                "5) I am done / Exit\n"
                "(Type 'undo' to go back and enter your location.)\n"
                "Enter 1, 2, 3, 4, or 5:")
            prompt = '❌ Booking cancelled. Returning to main menu.' + main_menu
            return {
                'prompt': prompt,
                'options': ["1", "2", "3", "4", "5", "undo"],
                'state': 'main_menu'
            }

    def api_handle_booking_result(self):
        booking_result = self.booking_context.get('booking_result', {})
        # Compose booking details section
        details = (
            f"\n🏥 HOSPITAL BOOKING SYSTEM\n"
            + "=" * 50 +
            f"\n📋 Booking Details:\n"
            f"  Hospital ID: {booking_result.get('hospital_id', 'unknown')}\n"
            f"  Patient Name: {booking_result.get('patient_name', '')}\n"
            f"  Patient Phone: {booking_result.get('patient_phone', '')}\n"
            f"  Date: {booking_result.get('appointment_date', '')}\n"
            f"  Time: {booking_result.get('appointment_time', '')}\n"
        )
        # Confirmation and info section
        confirm = (
            f"\n✅ BOOKING CONFIRMED!\n"
            + "=" * 50 +
            f"\n📅 Your appointment has been successfully booked.\n"
            f"📞 You will receive a confirmation call shortly.\n"
            f"🏥 Please arrive {BOOKING_ARRIVAL_MINUTES} minutes before your appointment time.\n"
            f"📋 Don't forget to bring your ID and insurance card.\n"
            f"🔢 Booking Reference: {booking_result.get('booking_reference', 'N/A')}\n"
            f"📄 Booking details saved to: bookings/booking_{booking_result.get('booking_reference', 'N/A')}.json\n"
            f"\n🎉 Booking completed successfully!\n"
            f"📧 A confirmation email has been sent to your registered email.\n"
            f"📱 You will also receive an SMS confirmation.\n"
        )
        # Main menu prompt
        main_menu = (
            "\nWhat would you like to do?\n"
            "1) Diagnosis (disease prediction)\n"
            "2) Find a doctor\n"
            "3) Find a hospital\n"
            "4) Book hospital appointment\n"
            "5) I am done / Exit\n"
            "(Type 'undo' to go back and enter your location.)\n"
            "Enter 1, 2, 3, 4, or 5:"
        )
        prompt = details + confirm + main_menu
        self.current_state = 'main_menu'
        return {
            'prompt': prompt,
            'options': ["1", "2", "3", "4", "5", "undo"],
            'state': 'main_menu'
        }

    def api_handle_follow_up_question(self, user_input):
        """Handle a follow-up question answer, update state, and continue diagnosis or ask next question"""
        user_input = user_input.strip().lower()
        # Restore context
        ctx = self.diagnosis_context
        symptoms = ctx.get('symptoms', self.symptoms)
        denied_symptoms = set(ctx.get('denied_symptoms', set()))
        previously_asked_symptoms = set(ctx.get('previously_asked_symptoms', set()))
        follow_up_questions = ctx.get('follow_up_questions', [])
        follow_up_round = ctx.get('follow_up_round', 0)
        max_follow_up_rounds = self.max_follow_up_rounds
        question = follow_up_questions[0] if follow_up_questions else None
        # Parse the symptom from the question
        if question:
            if question.startswith("Critical: "):
                symptom = question.replace("Critical: ", "").replace(" ", "_")
            elif question.startswith("Important: "):
                symptom = question.replace("Important: ", "").replace(" ", "_")
            else:
                symptom = question.replace(" ", "_")
        else:
            symptom = None
        # Update symptoms/denials
        if user_input in ["yes", "y"] and symptom:
            if symptom not in symptoms:
                symptoms.append(symptom)
            previously_asked_symptoms.add(symptom)
        elif user_input in ["no", "n"] and symptom:
            denied_symptoms.add(symptom)
            previously_asked_symptoms.add(symptom)
        elif user_input == "undo":
            # Go back to previous state (could be improved)
            self.current_state = "diagnosis_days"
            return {
                'prompt': "Okay. For how many days have you had these symptoms? : ",
                'options': ["undo"],
                'state': self.current_state
            }
        # Remove the answered question
        follow_up_questions = follow_up_questions[1:]
        # If more follow-up questions in this round, ask next
        if follow_up_questions:
            self.diagnosis_context = {
                'predicted_disease': ctx.get('predicted_disease'),
                'confidence': ctx.get('confidence'),
                'top_3_diseases': ctx.get('top_3_diseases'),
                'top_3_confidences': ctx.get('top_3_confidences'),
                'should_make_prediction': ctx.get('should_make_prediction'),
                'follow_up_questions': follow_up_questions,
                'safety_warnings': ctx.get('safety_warnings'),
                'risk_level': ctx.get('risk_level'),
                'denied_symptoms': denied_symptoms,
                'previously_asked_symptoms': previously_asked_symptoms,
                'follow_up_round': follow_up_round,
                'symptoms': symptoms,
            }
            self.current_state = "diagnosis_follow_up"
            next_question = follow_up_questions[0]
            formatted = next_question.replace("Critical: ", "").replace("Important: ", "")
            prompt = f"Please answer:\n{next_question}? (yes/y or no/n)"
            return {
                'prompt': prompt,
                'options': ["yes", "y", "no", "n", "undo"],
                'state': self.current_state,
                'question': next_question,
                'symptoms': symptoms
            }
        # Otherwise, increment round and re-run prediction
        follow_up_round += 1
        self.denied_symptoms = denied_symptoms
        self.previously_asked_symptoms = previously_asked_symptoms
        self.follow_up_round = follow_up_round
        self.symptoms = symptoms
        return self.api_make_diagnosis()

    def _get_doctor_recommendations_data(self, predicted_disease):
        """Return doctor recommendations as a list of dicts for the API response."""
        recommendations = []
        specialization = self.disease_to_specialty.get(predicted_disease, None)
        if not specialization:
            return recommendations
        doctors = self.get_doctors_by_specialization(specialization)
        if not doctors:
            return recommendations
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
        # If no doctors in location, show all
        top_doctors = filtered_doctors if filtered_doctors else doctors
        # Sort by rating
        top_doctors = sorted(top_doctors, key=lambda d: self._extract_rating_value(d), reverse=True)
        # Limit to top 5
        for doc in top_doctors[:5]:
            recommendations.append({
                'name': doc.get('name', 'Unknown'),
                'specialization': doc.get('specialization', 'N/A'),
                'city': doc.get('city', 'N/A'),
                'country': doc.get('country', 'N/A'),
                'phone': doc.get('phone', 'N/A'),
                'rate': doc.get('rate', 'N/A'),
                'gender': doc.get('gender', 'N/A'),
                'img_url': doc.get('ImgUrl'),
                'profile_url': doc.get('Url'),
            })
        return recommendations

    def api_handle_exit_confirm(self, user_input):
        user_input = user_input.strip().lower()
        if user_input == 'undo':
            self.current_state = 'awaiting_location'
            return {
                'prompt': 'Please enter your location (city or country):',
                'options': [],
                'state': self.current_state
            }
        if user_input in ['yes', 'y']:
            self.current_state = 'session_end'
            prompt = '\n👋 Thank you for using the Healthcare Chatbot! Have a great day! 🙏'
            return {
                'prompt': prompt,
                'options': [],
                'state': self.current_state
            }
        elif user_input in ['no', 'n']:
            self.current_state = 'main_menu'
            return self.api_get_main_menu()
        else:
            prompt = 'Please enter yes/y or no/n.\nAre you sure you want to exit? (yes/y or no/n):'
            return {
                'prompt': prompt,
                'options': ['yes', 'y', 'no', 'n', 'undo'],
                'state': self.current_state
            }

    def api_get_free_text_review_menu(self):
        """Return the current free text review menu prompt and options."""
        prompt = f"\nHere are the symptoms I have: {', '.join([self._format_symptom_name(s) for s in self.symptoms])}\nWhat would you like to do?\n1) Add a symptom\n2) Remove a symptom\n3) Edit all symptoms (re-enter full list)\n4) Continue with current symptoms"
        return {
            'prompt': prompt,
            'options': ["1", "2", "3", "4", "undo"],
            'state': 'free_text_review'
        }

@app.post("/start_session", response_model=StartSessionResponse)
async def start_session():
    """Start a new chatbot session"""
    session_id = str(uuid.uuid4())
    session = APIChatbotInterface(data_preprocessor, model_trainer)
    sessions[session_id] = session
    
    response = session.api_get_info()
    
    return StartSessionResponse(
        session_id=session_id,
        prompt=response['prompt'],
        options=response['options'],
        state=response['state']
    )

@app.post("/send_message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """Send a message to the chatbot and get response"""
    chatbot = sessions.get(request.session_id)
    if not chatbot:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if chatbot.current_state == "diagnosis_ready":
        response = chatbot.api_make_diagnosis()
    elif chatbot.current_state == "awaiting_name":
        response = chatbot.api_handle_name(request.user_input)
    elif chatbot.current_state == "awaiting_location":
        response = chatbot.api_handle_location(request.user_input)
    elif chatbot.current_state == "main_menu":
        response = chatbot.api_handle_main_menu(request.user_input)
    elif chatbot.current_state == "diagnosis_method":
        response = chatbot.api_handle_diagnosis_method(request.user_input)
    elif chatbot.current_state == "diagnosis_symptom":
        response = chatbot.api_handle_symptom_input(request.user_input)
    elif chatbot.current_state == "diagnosis_review":
        response = chatbot.api_handle_diagnosis_review(request.user_input)
    elif chatbot.current_state == "diagnosis_days":
        response = chatbot.api_handle_days_input(request.user_input)
    elif chatbot.current_state == "diagnosis_follow_up":
        response = chatbot.api_handle_follow_up_question(request.user_input)
    elif chatbot.current_state == "symptom_clarification":
        response = chatbot.api_handle_symptom_clarification(request.user_input)
    elif chatbot.current_state == "free_text_review":
        response = chatbot.api_handle_free_text_review(request.user_input)
    elif chatbot.current_state == "free_text_add_symptom":
        response = chatbot.api_handle_free_text_add_symptom(request.user_input)
    elif chatbot.current_state == "free_text_remove_symptom":
        response = chatbot.api_handle_free_text_remove_symptom(request.user_input)
    elif chatbot.current_state == "free_text_edit_all":
        response = chatbot.api_handle_free_text_edit_all(request.user_input)
    elif chatbot.current_state == "related_symptoms":
        response = chatbot.api_handle_related_symptoms(request.user_input)
    elif chatbot.current_state == "doctor_search":
        response = chatbot.api_handle_doctor_search(request.user_input)
    elif chatbot.current_state == "doctor_search_again":
        response = chatbot.api_handle_doctor_search_again(request.user_input)
    elif chatbot.current_state == "hospital_search":
        response = chatbot.api_handle_hospital_search(request.user_input)
    elif chatbot.current_state == "hospital_search_again":
        response = chatbot.api_handle_hospital_search_again(request.user_input)
    elif chatbot.current_state == "booking_menu":
        response = chatbot.api_handle_booking_menu(request.user_input)
    elif chatbot.current_state == "exit_confirm":
        response = chatbot.api_handle_exit_confirm(request.user_input)
    elif chatbot.current_state == "diagnosis_related":
        response = chatbot.api_handle_related_symptoms(request.user_input)
    elif chatbot.current_state == "diagnosis_free_text":
        response = chatbot.api_handle_free_text_input(request.user_input)
    elif chatbot.current_state == "diagnosis_symptom_edit":
        response = chatbot.api_handle_symptom_edit(request.user_input)
    elif chatbot.current_state == "booking_location":
        response = chatbot.api_handle_booking_location(request.user_input)
    elif chatbot.current_state == "booking_hospital_select":
        response = chatbot.api_handle_booking_hospital_select(request.user_input)
    elif chatbot.current_state == "booking_patient_name":
        response = chatbot.api_handle_booking_patient_name(request.user_input)
    elif chatbot.current_state == "booking_patient_phone":
        response = chatbot.api_handle_booking_patient_phone(request.user_input)
    elif chatbot.current_state == "booking_date_select":
        response = chatbot.api_handle_booking_date_select(request.user_input)
    elif chatbot.current_state == "booking_time_select":
        response = chatbot.api_handle_booking_time_select(request.user_input)
    elif chatbot.current_state == "booking_confirm":
        response = chatbot.api_handle_booking_confirm(request.user_input)
    elif chatbot.current_state == "booking_result":
        response = chatbot.api_handle_booking_result()
    elif chatbot.current_state == "free_text_symptom_clarification":
        response = chatbot.api_handle_free_text_symptom_clarification(request.user_input)
    else:
        response = {"prompt": "Unknown state.", "options": [], "state": chatbot.current_state}

    return response

@app.get("/symptoms")
async def get_symptoms():
    """Get the complete list of symptoms the model was trained on"""
    return {"symptoms": data_preprocessor.cols}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 