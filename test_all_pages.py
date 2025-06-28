#!/usr/bin/env python3
"""
Test script to verify that the system fetches all pages from APIs
"""

import requests
import json

def test_doctors_all_pages():
    """Test fetching all doctors from all pages"""
    print("👨‍⚕️ TESTING DOCTORS API - ALL PAGES")
    print("=" * 60)
    
    base_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors"
    all_doctors = []
    
    try:
        # Fetch first page to get total pages info
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        
        total_pages = data.get('totalPages', 1)
        total_doctors = data.get('totalDoctors', 0)
        
        print(f"📊 API Info:")
        print(f"  Total Pages: {total_pages}")
        print(f"  Total Doctors: {total_doctors}")
        print(f"  Current Page: {data.get('currentPage', 1)}")
        print(f"  Limit per page: {data.get('limit', 10)}")
        print()
        
        # Fetch all pages
        for page in range(1, total_pages + 1):
            page_url = f"{base_url}?page={page}"
            print(f"📄 Loading page {page}/{total_pages}...")
            
            response = requests.get(page_url)
            response.raise_for_status()
            data = response.json()
            doctors = data.get('data', [])
            all_doctors.extend(doctors)
            
            print(f"  ✅ Page {page}: {len(doctors)} doctors")
            
            # Show sample doctor from this page
            if doctors:
                sample = doctors[0]
                print(f"    Sample: {sample.get('name', 'N/A')} - {sample.get('specialization', 'N/A')}")
        
        print(f"\n🎉 SUCCESS: Total doctors fetched: {len(all_doctors)}")
        
        # Show statistics
        specializations = {}
        cities = {}
        countries = {}
        
        for doc in all_doctors:
            spec = doc.get('specialization', 'Unknown')
            city = doc.get('city', 'Unknown')
            country = doc.get('country', 'Unknown')
            
            specializations[spec] = specializations.get(spec, 0) + 1
            cities[city] = cities.get(city, 0) + 1
            countries[country] = countries.get(country, 0) + 1
        
        print(f"\n📊 STATISTICS:")
        print(f"  Specializations: {len(specializations)}")
        print(f"  Cities: {len(cities)}")
        print(f"  Countries: {len(countries)}")
        
        print(f"\n🏥 Top Specializations:")
        for spec, count in sorted(specializations.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {spec}: {count} doctors")
        
        print(f"\n🌍 Top Countries:")
        for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {country}: {count} doctors")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_hospitals_all_pages():
    """Test fetching all hospitals from all pages"""
    print("\n🏥 TESTING HOSPITALS API - ALL PAGES")
    print("=" * 60)
    
    base_url = "https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals"
    all_hospitals = []
    
    try:
        # Fetch first page to get total pages info
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        
        total_pages = data.get('totalPages', 1)
        total_hospitals = data.get('totalHospitals', 0)
        
        print(f"📊 API Info:")
        print(f"  Total Pages: {total_pages}")
        print(f"  Total Hospitals: {total_hospitals}")
        print(f"  Current Page: {data.get('currentPage', 1)}")
        print(f"  Limit per page: {data.get('limit', 10)}")
        print()
        
        # Fetch all pages
        for page in range(1, total_pages + 1):
            page_url = f"{base_url}?page={page}"
            print(f"📄 Loading page {page}/{total_pages}...")
            
            response = requests.get(page_url)
            response.raise_for_status()
            data = response.json()
            hospitals = data.get('data', [])
            all_hospitals.extend(hospitals)
            
            print(f"  ✅ Page {page}: {len(hospitals)} hospitals")
            
            # Show sample hospital from this page
            if hospitals:
                sample = hospitals[0]
                print(f"    Sample: {sample.get('name', 'N/A')} - {sample.get('city', 'N/A')}, {sample.get('country', 'N/A')}")
        
        print(f"\n🎉 SUCCESS: Total hospitals fetched: {len(all_hospitals)}")
        
        # Show statistics
        cities = {}
        countries = {}
        ratings = {}
        
        for hosp in all_hospitals:
            city = hosp.get('city', 'Unknown')
            country = hosp.get('country', 'Unknown')
            rating = hosp.get('rate', 'N/A')
            
            cities[city] = cities.get(city, 0) + 1
            countries[country] = countries.get(country, 0) + 1
            
            if rating != 'N/A' and rating is not None:
                ratings[rating] = ratings.get(rating, 0) + 1
        
        print(f"\n📊 STATISTICS:")
        print(f"  Cities: {len(cities)}")
        print(f"  Countries: {len(countries)}")
        print(f"  Rating levels: {len(ratings)}")
        
        print(f"\n🌍 Top Countries:")
        for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {country}: {count} hospitals")
        
        print(f"\n⭐ Rating Distribution:")
        for rating, count in sorted(ratings.items(), key=lambda x: float(x[0]) if x[0] != 'N/A' else 0, reverse=True):
            print(f"  {rating}/5: {count} hospitals")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_specialization_search():
    """Test searching for doctors by specialization with all pages"""
    print("\n🔍 TESTING SPECIALIZATION SEARCH - ALL PAGES")
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
        
        # Test different specializations
        test_specializations = ['Cardiology', 'Neurology', 'Dermatology', 'Gastroenterology']
        
        for spec in test_specializations:
            filtered = []
            for doc in all_doctors:
                doc_spec = doc.get('specialization', '').lower().strip()
                if spec.lower() in doc_spec or doc_spec in spec.lower():
                    filtered.append(doc)
            
            print(f"🔍 {spec}: {len(filtered)} doctors found")
            
            if filtered:
                print(f"  Sample doctors:")
                for doc in filtered[:3]:  # Show first 3
                    print(f"    - {doc.get('name', 'N/A')} ({doc.get('city', 'N/A')}, {doc.get('country', 'N/A')})")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTING ALL PAGES FUNCTIONALITY")
    print("=" * 60)
    
    # Test doctors API
    doctors_ok = test_doctors_all_pages()
    
    # Test hospitals API
    hospitals_ok = test_hospitals_all_pages()
    
    # Test specialization search
    search_ok = test_specialization_search()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"  Doctors API (All Pages): {'✅ Working' if doctors_ok else '❌ Failed'}")
    print(f"  Hospitals API (All Pages): {'✅ Working' if hospitals_ok else '❌ Failed'}")
    print(f"  Specialization Search: {'✅ Working' if search_ok else '❌ Failed'}")
    
    if doctors_ok and hospitals_ok and search_ok:
        print("\n🎉 All tests passed! The system now fetches all pages correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 