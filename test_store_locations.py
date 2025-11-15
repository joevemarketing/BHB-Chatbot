#!/usr/bin/env python3
"""Test current store location functionality."""

import requests
import json

def test_store_queries():
    """Test various store location queries."""
    
    test_queries = [
        "Where is the BHB store near me?",
        "BHB store location near me",
        "BHB branch near me",
        "Where can I find BHB store in Penang?",
        "BHB store address",
        "BHB outlet near me",
        "BHB kedai near me",
        "BHB lokasi kedai",
        "BHB alamat cawangan"
    ]
    
    print("Testing Store Location Queries")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/chat",
                json={
                    "session_id": f"store_test_{i}",
                    "message": query,
                    "domain": "smart_support"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result.get('reply', 'No reply')
                items = result.get('items', [])
                
                print(f"Reply: {reply[:200]}...")
                print(f"Items count: {len(items)}")
                
                if items:
                    for j, item in enumerate(items[:2]):
                        name = item.get('name', 'Unknown')
                        print(f"  {j+1}. {name}")
                else:
                    print("  No store items returned")
                    
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_store_queries()