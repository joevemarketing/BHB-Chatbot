#!/usr/bin/env python3
"""Debug store location detection step by step."""

import requests
import json

def debug_store_detection():
    """Debug the store location detection process."""
    
    query = "Where is the BHB store near me?"
    print(f"Testing query: {query}")
    print("=" * 60)
    
    # Test the store location function directly
    print("\n1. Testing store_keywords detection:")
    store_keywords = [
        "location", "lokasi", "alamat", "branch", "cawangan",
        "near me", "near", "berhampiran", "dekat",
        "kedai", "shop", "store", "outlet",
    ]
    
    query_lower = query.lower()
    has_store_signal = any(k in query_lower for k in store_keywords)
    print(f"Query: '{query_lower}'")
    print(f"Store keywords found: {[k for k in store_keywords if k in query_lower]}")
    print(f"has_store_signal: {has_store_signal}")
    
    # Test store location search
    print("\n2. Testing store location search:")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/chat",
            json={
                "session_id": "debug_store_test",
                "message": query,
                "domain": "smart_support"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response received")
            print(f"Reply preview: {result.get('reply', 'No reply')[:150]}...")
            
            # Check if store locations are in the response
            if 'items' in result:
                items = result['items']
                print(f"Items count: {len(items)}")
                for i, item in enumerate(items[:3]):
                    print(f"  {i+1}. {item.get('name', 'Unknown')}")
            else:
                print("No items in response")
                
            # Let's also check if there's any store context being passed
            print(f"Full response keys: {list(result.keys())}")
            
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    debug_store_detection()