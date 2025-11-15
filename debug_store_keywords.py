#!/usr/bin/env python3
"""Debug why some store queries aren't being caught."""

import requests
import json

def debug_store_detection():
    """Debug store location detection for specific queries."""
    
    test_queries = [
        "Where is the BHB store near me?",
        "BHB store location near me",
        "BHB store address"
    ]
    
    store_keywords = [
        "location", "lokasi", "alamat", "branch", "cawangan",
        "near me", "near", "berhampiran", "dekat",
        "kedai", "shop", "store", "outlet",
    ]
    
    print("Testing Store Keyword Detection")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_lower = query.lower()
        
        # Check which keywords are found
        found_keywords = [k for k in store_keywords if k in query_lower]
        has_store_signal = bool(found_keywords)
        
        print(f"Lowercase: '{query_lower}'")
        print(f"Found keywords: {found_keywords}")
        print(f"has_store_signal: {has_store_signal}")
        
        # Test the actual response
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/chat",
                json={
                    "session_id": f"debug_test",
                    "message": query,
                    "domain": "smart_support"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                reply_start = result.get('reply', 'No reply')[:100]
                print(f"Response: {reply_start}...")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_store_detection()