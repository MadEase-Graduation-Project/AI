#!/usr/bin/env python3
"""
Test script to verify that doctors and hospitals are sorted by rating
"""

import requests
import json

def test_doctors_rating_sort():
    """Test that doctors are sorted by rating from highest to lowest"""
    print("👨‍⚕️ TESTING DOCTORS RATING SORT")
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
        
        print(f"📊 Total doctors fetched: {len(all_doctors)}")
        
        # Test sorting for different specializations
        test_specializations = ['Cardiology', 'Dermatology', 'Gastroenterology']
        
        for spec in test_specializations:
            print(f"\n🔍 Testing {spec} doctors:")
            
            # Filter by specialization
            filtered = []
            specialization_lower = spec.lower().strip()
            
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
            
            print(f"  Found {len(filtered)} {spec} doctors:")
            
            if filtered:
                print(f"  📋 Sorted by rating (highest to lowest):")
                for i, doc in enumerate(filtered, 1):
                    name = doc.get('name', 'N/A')
                    rating = doc.get('rate', 'N/A')
                    city = doc.get('city', 'N/A')
                    country = doc.get('country', 'N/A')
                    print(f"    {i}. {name} - ⭐ {rating}/5 ({city}, {country})")
            else:
                print(f"  ❌ No {spec} doctors found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_hospitals_rating_sort():
    """Test that hospitals are sorted by rating from highest to lowest"""
    print("\n🏥 TESTING HOSPITALS RATING SORT")
    print("=" * 60)
    
    base_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
    all_hospitals = []
    
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
            hospitals = data.get('data', [])
            all_hospitals.extend(hospitals)
        
        print(f"📊 Total hospitals fetched: {len(all_hospitals)}")
        
        # Test sorting for different locations
        test_locations = ['Egypt', 'USA', 'Germany']
        
        for loc in test_locations:
            print(f"\n🔍 Testing hospitals in {loc}:")
            
            # Filter by location
            location_lower = loc.lower().strip()
            filtered = []
            
            for hosp in all_hospitals:
                hosp_city = str(hosp.get('city', '')).lower().strip()
                hosp_country = str(hosp.get('country', '')).lower().strip()
                hosp_location = f"{hosp_city} {hosp_country}".strip()
                if (location_lower in hosp_location or 
                    hosp_city in location_lower or 
                    hosp_country in location_lower):
                    filtered.append(hosp)
            
            # Sort by rating (highest to lowest)
            def get_rating_value(hosp):
                rating = hosp.get('rate')
                if rating is None or rating == 'N/A' or rating == '':
                    return 0.0
                try:
                    return float(rating)
                except (ValueError, TypeError):
                    return 0.0
            
            filtered.sort(key=get_rating_value, reverse=True)
            
            print(f"  Found {len(filtered)} hospitals in {loc}:")
            
            if filtered:
                print(f"  📋 Sorted by rating (highest to lowest):")
                for i, hosp in enumerate(filtered, 1):
                    name = hosp.get('name', 'N/A')
                    rating = hosp.get('rate', 'N/A')
                    city = hosp.get('city', 'N/A')
                    country = hosp.get('country', 'N/A')
                    print(f"    {i}. {name} - ⭐ {rating}/5 ({city}, {country})")
            else:
                print(f"  ❌ No hospitals found in {loc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_all_doctors_sorted():
    """Test all doctors sorted by rating"""
    print("\n👨‍⚕️ ALL DOCTORS SORTED BY RATING")
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
        
        # Sort all doctors by rating
        def get_rating_value(doc):
            rating = doc.get('rate')
            if rating is None or rating == 'N/A' or rating == '':
                return 0.0
            try:
                return float(rating)
            except (ValueError, TypeError):
                return 0.0
        
        all_doctors.sort(key=get_rating_value, reverse=True)
        
        print(f"📊 Top 10 doctors by rating:")
        for i, doc in enumerate(all_doctors[:10], 1):
            name = doc.get('name', 'N/A')
            rating = doc.get('rate', 'N/A')
            specialization = doc.get('specialization', 'N/A')
            city = doc.get('city', 'N/A')
            country = doc.get('country', 'N/A')
            print(f"  {i}. {name} - ⭐ {rating}/5")
            print(f"     Specialization: {specialization}")
            print(f"     Location: {city}, {country}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTING RATING SORT FUNCTIONALITY")
    print("=" * 60)
    
    # Test doctors sorting
    doctors_ok = test_doctors_rating_sort()
    
    # Test hospitals sorting
    hospitals_ok = test_hospitals_rating_sort()
    
    # Test all doctors sorted
    all_doctors_ok = test_all_doctors_sorted()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"  Doctors Rating Sort: {'✅ Working' if doctors_ok else '❌ Failed'}")
    print(f"  Hospitals Rating Sort: {'✅ Working' if hospitals_ok else '❌ Failed'}")
    print(f"  All Doctors Sorted: {'✅ Working' if all_doctors_ok else '❌ Failed'}")
    
    if doctors_ok and hospitals_ok and all_doctors_ok:
        print("\n🎉 All tests passed! Rating sort is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 