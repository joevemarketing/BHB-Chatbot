#!/usr/bin/env python3
"""Simple test to verify washer recommendations are working."""

import requests
import json

def test_simple():
    """Simple test with direct output."""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/chat",
            json={
                "session_id": "simple_test",
                "message": "recommend washers",
                "domain": "smart_support"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: API responded")
            print(f"Reply: {result.get('reply', 'No reply')}")
            
            if 'items' in result and result['items']:
                print(f"\nFound {len(result['items'])} products:")
                for i, item in enumerate(result['items'][:3]):
                    name = item.get('name', 'Unknown')
                    category = item.get('category', 'Unknown')
                    print(f"  {i+1}. {name} (Category: {category})")
            else:
                print("No products found in response")
        else:
            print(f"ERROR: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_simple()