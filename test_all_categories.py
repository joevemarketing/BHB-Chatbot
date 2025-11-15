#!/usr/bin/env python3
"""Test current product loading status."""

import requests
import json

def test_product_loading():
    """Test what products are currently loaded."""
    
    # Test different categories
    test_cases = [
        "recommend washers",
        "recommend fans", 
        "recommend water heaters",
        "recommend rice cookers",
        "recommend air fryers"
    ]
    
    for message in test_cases:
        print(f"\n--- Testing: {message} ---")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/chat",
                json={
                    "session_id": f"test_{message.replace(' ', '_')}",
                    "message": message,
                    "domain": "smart_support"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result.get('reply', '')
                
                # Check if response mentions the expected category
                category_word = message.split()[-1].lower()
                if category_word in reply.lower():
                    print(f"✓ Response mentions {category_word}")
                else:
                    print(f"? Response may not mention {category_word}")
                    
                if 'items' in result and result['items']:
                    print(f"✓ Found {len(result['items'])} products")
                    # Show first 2 products
                    for i, item in enumerate(result['items'][:2]):
                        name = item.get('name', 'Unknown')
                        category = item.get('category', 'Unknown')
                        print(f"  {i+1}. {name} (Cat: {category})")
                else:
                    print("✗ No products found")
            else:
                print(f"✗ Error: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_product_loading()