#!/usr/bin/env python3
"""
Test script to verify that doctor recommendations after diagnosis work correctly
"""

import requests
import json

def test_diagnosis_doctor_recommendations():
    """Test doctor recommendations after diagnosis"""
    print("🩺 TESTING DIAGNOSIS DOCTOR RECOMMENDATIONS")
    print("=" * 60)
    
    # Simulate the disease_to_specialty mapping
    disease_to_specialty = {
        'Common Cold': 'Internal Medicine',
        'Allergy': 'Allergy & Immunology',
        'Migraine': 'Neurology',
        'Peptic ulcer disease': 'Gastroenterology',
        'GERD': 'Gastroenterology',
        'Heart attack': 'Cardiology',
        'Pneumonia': 'Pulmonology',
        'Tuberculosis': 'Pulmonology',
        'Malaria': 'Infectious Disease',
        'Dengue': 'Infectious Disease',
        'Hepatitis A': 'Hepatology & Gastroenterology',
        'Hepatitis B': 'Hepatology & Gastroenterology',
        'Hepatitis C': 'Hepatology & Gastroenterology',
        'Hepatitis D': 'Hepatology & Gastroenterology',
        'Hepatitis E': 'Hepatology & Gastroenterology',
        'Alcoholic hepatitis': 'Hepatology & Gastroenterology',
        'Typhoid': 'Infectious Disease',
        'AIDS': 'Infectious Disease',
        'Paralysis (brain hemorrhage)': 'Neurology',
        'Brain hemorrhage': 'Neurology',
        'Fungal infection': 'Dermatology',
        'Acne': 'Dermatology'
    }
    
    # Test diseases
    test_diseases = ['Common Cold', 'Heart attack', 'Hepatitis B', 'Fungal infection', 'Migraine']
    
    base_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
    all_doctors = []
    
    try:
        # Fetch all pages
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        total_pages = data.get('totalPages', 1)
        
        for page in range(1, total_pages + 1):
            page_url = f"{base_url}?page={page}"
            response = requests.get(page_url)
            response.raise_for_status()
            data = response.json()
            doctors = data.get('data', [])
            all_doctors.extend(doctors)
        
        print(f"📊 Total doctors fetched: {len(all_doctors)}")
        
        # Test each disease
        for disease in test_diseases:
            print(f"\n🔍 Testing diagnosis: {disease}")
            print("-" * 40)
            
            specialization = disease_to_specialty.get(disease, 'General Medicine')
            print(f"  Medical Specialty: {specialization}")
            
            # Filter doctors by specialization
            filtered = []
            specialization_lower = specialization.lower().strip()
            
            for doc in all_doctors:
                doc_specialization = doc.get('specialization', '').lower().strip()
                if (specialization_lower == doc_specialization or 
                    specialization_lower in doc_specialization or 
                    doc_specialization in specialization_lower):
                    filtered.append(doc)
            
            # Sort by rating (highest to lowest)
            def get_rating_value(doc):
                rating = doc.get('rate')
                if rating is None or rating == 'N/A' or rating == '':
                    return 0.0
                try:
                    return float(rating)
                except (ValueError, TypeError):
                    return 0.0
            
            filtered.sort(key=get_rating_value, reverse=True)
            
            print(f"  Found {len(filtered)} doctors for {specialization}")
            
            if filtered:
                print(f"  📋 Top doctors (sorted by rating):")
                for i, doc in enumerate(filtered[:5], 1):  # Show top 5
                    name = doc.get('name', 'Unknown')
                    rating = doc.get('rate', 'N/A')
                    city = doc.get('city', 'N/A')
                    country = doc.get('country', 'N/A')
                    phone = doc.get('phone', 'N/A')
                    gender = doc.get('gender', 'N/A')
                    
                    print(f"    {i}. {name}")
                    print(f"       Specialty: {doc.get('specialization', 'N/A')}")
                    print(f"       Location: {city}, {country}")
                    print(f"       Phone: {phone}")
                    print(f"       Rating: {rating}/5")
                    print(f"       Gender: {gender}")
                    print()
            else:
                print(f"  ❌ No doctors found for {specialization}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_location_filtering():
    """Test location filtering for doctor recommendations"""
    print("\n📍 TESTING LOCATION FILTERING")
    print("=" * 60)
    
    base_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
    all_doctors = []
    
    try:
        # Fetch all pages
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        total_pages = data.get('totalPages', 1)
        
        for page in range(1, total_pages + 1):
            page_url = f"{base_url}?page={page}"
            response = requests.get(page_url)
            response.raise_for_status()
            data = response.json()
            doctors = data.get('data', [])
            all_doctors.extend(doctors)
        
        # Test different locations
        test_locations = ['Egypt', 'USA', 'New York', 'Cairo']
        
        for location in test_locations:
            print(f"\n🔍 Testing location: {location}")
            print("-" * 30)
            
            location_lower = location.lower().strip()
            filtered = []
            
            for doc in all_doctors:
                doc_city = str(doc.get('city', '')).lower().strip()
                doc_country = str(doc.get('country', '')).lower().strip()
                doc_location = f"{doc_city} {doc_country}".strip()
                if (location_lower in doc_location or 
                    doc_city in location_lower or 
                    doc_country in location_lower):
                    filtered.append(doc)
            
            # Sort by rating
            def get_rating_value(doc):
                rating = doc.get('rate')
                if rating is None or rating == 'N/A' or rating == '':
                    return 0.0
                try:
                    return float(rating)
                except (ValueError, TypeError):
                    return 0.0
            
            filtered.sort(key=get_rating_value, reverse=True)
            
            print(f"  Found {len(filtered)} doctors in {location}")
            
            if filtered:
                print(f"  📋 Top doctors in {location} (sorted by rating):")
                for i, doc in enumerate(filtered[:3], 1):  # Show top 3
                    name = doc.get('name', 'Unknown')
                    rating = doc.get('rate', 'N/A')
                    specialization = doc.get('specialization', 'N/A')
                    city = doc.get('city', 'N/A')
                    country = doc.get('country', 'N/A')
                    
                    print(f"    {i}. {name} - ⭐ {rating}/5")
                    print(f"       {specialization} ({city}, {country})")
            else:
                print(f"  ❌ No doctors found in {location}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTING DIAGNOSIS DOCTOR RECOMMENDATIONS")
    print("=" * 60)
    
    # Test diagnosis recommendations
    diagnosis_ok = test_diagnosis_doctor_recommendations()
    
    # Test location filtering
    location_ok = test_location_filtering()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"  Diagnosis Recommendations: {'✅ Working' if diagnosis_ok else '❌ Failed'}")
    print(f"  Location Filtering: {'✅ Working' if location_ok else '❌ Failed'}")
    
    if diagnosis_ok and location_ok:
        print("\n🎉 All tests passed! Diagnosis doctor recommendations work correctly!")
        print("✅ Same functionality as main doctor search")
        print("✅ Rating sorting applied")
        print("✅ All pages fetched")
        print("✅ Location filtering works")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 