#!/usr/bin/env python3
"""Debug product pricing to see what's in the data."""

import requests
import json

def debug_pricing():
    """Debug product pricing to see what's in the data."""
    
    test_message = "Recommend a 55\" 4K TV under $600"
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/chat",
            json={
                "session_id": "price_debug",
                "message": test_message,
                "domain": "smart_support"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Testing: {test_message}")
            print()
            
            if 'items' in result and result['items']:
                print(f"Found {len(result['items'])} products:")
                
                for i, item in enumerate(result['items'][:3]):
                    print(f"\nProduct {i+1}:")
                    print(f"  Name: {item.get('name', 'Unknown')}")
                    print(f"  Brand: {item.get('brand', 'Unknown')}")
                    print(f"  Price: {repr(item.get('price'))}")  # Use repr to see if it's None
                    print(f"  Currency: {repr(item.get('currency'))}")
                    print(f"  Category: {item.get('category', 'Unknown')}")
                    
                    # Check all keys to see what's available
                    print(f"  All keys: {list(item.keys())}")
                    
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_pricing()