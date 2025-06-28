#!/usr/bin/env python3
"""
Test script to check if the doctors and hospitals APIs are working correctly
"""

import requests
import json

def test_doctors_api():
    """Test the doctors API endpoint"""
    print("🔍 Testing Doctors API...")
    url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            doctors = data.get('data', [])
            print(f"✅ Doctors API is working! Found {len(doctors)} doctors")
            
            if doctors:
                print("\n📋 Sample doctor data:")
                sample_doctor = doctors[0]
                print(f"  Name: {sample_doctor.get('name', 'N/A')}")
                print(f"  Specialization: {sample_doctor.get('specialization', 'N/A')}")
                print(f"  Rating: {sample_doctor.get('rating', 'N/A')}")
                print(f"  Experience: {sample_doctor.get('experience', 'N/A')}")
                print(f"  Location: {sample_doctor.get('location', 'N/A')}")
            
            return True
        else:
            print(f"❌ Doctors API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Doctors API: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_hospitals_api():
    """Test the hospitals API endpoint"""
    print("\n🔍 Testing Hospitals API...")
    url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            hospitals = data.get('data', [])
            print(f"✅ Hospitals API is working! Found {len(hospitals)} hospitals")
            
            if hospitals:
                print("\n📋 Sample hospital data:")
                sample_hospital = hospitals[0]
                print(f"  Name: {sample_hospital.get('name', 'N/A')}")
                print(f"  City: {sample_hospital.get('city', 'N/A')}")
                print(f"  Country: {sample_hospital.get('country', 'N/A')}")
                print(f"  Rating: {sample_hospital.get('rate', 'N/A')}")
                print(f"  Phone: {sample_hospital.get('phone', 'N/A')}")
                print(f"  Established: {sample_hospital.get('Established', 'N/A')}")
            
            return True
        else:
            print(f"❌ Hospitals API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Hospitals API: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_specialization_filtering():
    """Test specialization filtering functionality"""
    print("\n🔍 Testing Specialization Filtering...")
    url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            doctors = data.get('data', [])
            
            # Test filtering by specialization
            test_specializations = ['Cardiology', 'Neurology', 'Dermatology']
            
            for spec in test_specializations:
                filtered = []
                for doc in doctors:
                    doc_spec = doc.get('specialization', '').lower().strip()
                    if spec.lower() in doc_spec or doc_spec in spec.lower():
                        filtered.append(doc)
                
                print(f"  {spec}: {len(filtered)} doctors found")
            
            return True
        else:
            print(f"❌ Cannot test filtering - API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing filtering: {e}")
        return False

def test_location_filtering():
    """Test location filtering functionality"""
    print("\n🔍 Testing Location Filtering...")
    url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hospitals = data.get('data', [])
            
            # Test filtering by location
            test_locations = ['Cairo', 'Alexandria', 'Giza']
            
            for loc in test_locations:
                filtered = []
                for hosp in hospitals:
                    hosp_city = str(hosp.get('city', '')).lower().strip()
                    hosp_country = str(hosp.get('country', '')).lower().strip()
                    if loc.lower() in hosp_city or loc.lower() in hosp_country:
                        filtered.append(hosp)
                
                print(f"  {loc}: {len(filtered)} hospitals found")
            
            return True
        else:
            print(f"❌ Cannot test filtering - API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing filtering: {e}")
        return False

if __name__ == "__main__":
    print("🏥 API Testing Script")
    print("=" * 50)
    
    # Test both APIs
    doctors_ok = test_doctors_api()
    hospitals_ok = test_hospitals_api()
    
    # Test filtering functionality
    if doctors_ok:
        test_specialization_filtering()
    
    if hospitals_ok:
        test_location_filtering()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"  Doctors API: {'✅ Working' if doctors_ok else '❌ Failed'}")
    print(f"  Hospitals API: {'✅ Working' if hospitals_ok else '❌ Failed'}")
    
    if doctors_ok and hospitals_ok:
        print("\n🎉 All APIs are working correctly!")
    else:
        print("\n⚠️  Some APIs have issues. Check the error messages above.") 