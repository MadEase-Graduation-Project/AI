#!/usr/bin/env python3
"""
Check real data from doctors and hospitals APIs
"""

import requests
import json

def show_doctors_data():
    """Show all doctors data from API"""
    print("👨‍⚕️ DOCTORS DATA FROM API")
    print("=" * 60)
    
    url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            doctors = data.get('data', [])
            
            print(f"Total doctors found: {len(doctors)}")
            print()
            
            for i, doctor in enumerate(doctors, 1):
                print(f"Doctor {i}:")
                print(f"  Name: {doctor.get('name', 'N/A')}")
                print(f"  Specialization: {doctor.get('specialization', 'N/A')}")
                print(f"  Rating: {doctor.get('rating', 'N/A')}")
                print(f"  Experience: {doctor.get('experience', 'N/A')}")
                print(f"  Location: {doctor.get('location', 'N/A')}")
                print(f"  Phone: {doctor.get('phone', 'N/A')}")
                print(f"  Email: {doctor.get('email', 'N/A')}")
                print(f"  ID: {doctor.get('_id', 'N/A')}")
                print()
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def show_hospitals_data():
    """Show all hospitals data from API"""
    print("🏥 HOSPITALS DATA FROM API")
    print("=" * 60)
    
    url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hospitals = data.get('data', [])
            
            print(f"Total hospitals found: {len(hospitals)}")
            print()
            
            for i, hospital in enumerate(hospitals, 1):
                print(f"Hospital {i}:")
                print(f"  Name: {hospital.get('name', 'N/A')}")
                print(f"  City: {hospital.get('city', 'N/A')}")
                print(f"  Country: {hospital.get('country', 'N/A')}")
                print(f"  Rating: {hospital.get('rate', 'N/A')}")
                print(f"  Phone: {hospital.get('phone', 'N/A')}")
                print(f"  Established: {hospital.get('Established', 'N/A')}")
                print(f"  Address: {hospital.get('address', 'N/A')}")
                print(f"  ID: {hospital.get('_id', 'N/A')}")
                print()
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_data_quality():
    """Analyze the quality of data from both APIs"""
    print("📊 DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Check doctors data quality
    print("👨‍⚕️ DOCTORS DATA QUALITY:")
    doctors_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
    
    try:
        response = requests.get(doctors_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            doctors = data.get('data', [])
            
            # Count missing fields
            missing_name = sum(1 for d in doctors if not d.get('name'))
            missing_spec = sum(1 for d in doctors if not d.get('specialization'))
            missing_rating = sum(1 for d in doctors if not d.get('rating'))
            missing_location = sum(1 for d in doctors if not d.get('location'))
            
            print(f"  Total doctors: {len(doctors)}")
            print(f"  Missing names: {missing_name}")
            print(f"  Missing specializations: {missing_spec}")
            print(f"  Missing ratings: {missing_rating}")
            print(f"  Missing locations: {missing_location}")
            
            # Show specializations
            specs = set(d.get('specialization', '') for d in doctors if d.get('specialization'))
            print(f"  Available specializations: {', '.join(specs)}")
            
    except Exception as e:
        print(f"  ❌ Error analyzing doctors data: {e}")
    
    print()
    
    # Check hospitals data quality
    print("🏥 HOSPITALS DATA QUALITY:")
    hospitals_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
    
    try:
        response = requests.get(hospitals_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hospitals = data.get('data', [])
            
            # Count missing fields
            missing_name = sum(1 for h in hospitals if not h.get('name'))
            missing_city = sum(1 for h in hospitals if not h.get('city'))
            missing_country = sum(1 for h in hospitals if not h.get('country'))
            missing_rating = sum(1 for h in hospitals if not h.get('rate'))
            
            print(f"  Total hospitals: {len(hospitals)}")
            print(f"  Missing names: {missing_name}")
            print(f"  Missing cities: {missing_city}")
            print(f"  Missing countries: {missing_country}")
            print(f"  Missing ratings: {missing_rating}")
            
            # Show cities
            cities = set(h.get('city', '') for h in hospitals if h.get('city'))
            print(f"  Available cities: {', '.join(cities)}")
            
    except Exception as e:
        print(f"  ❌ Error analyzing hospitals data: {e}")

if __name__ == "__main__":
    print("🔍 CHECKING REAL API DATA")
    print("=" * 60)
    
    # Show detailed data
    show_doctors_data()
    show_hospitals_data()
    
    # Analyze data quality
    analyze_data_quality()
    
    print("\n" + "=" * 60)
    print("✅ API DATA CHECK COMPLETE") 