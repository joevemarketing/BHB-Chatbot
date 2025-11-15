#!/usr/bin/env python3
"""Test script to verify product recommendations are working correctly."""

import json
import requests
import time

def test_washer_recommendations():
    """Test that washer questions return washer products."""
    
    # Test data
    test_cases = [
        {
            "message": "What's the difference between front-load and top-load washers?",
            "expected_category": "Washer"
        },
        {
            "message": "recommend washers",
            "expected_category": "Washer"
        },
        {
            "message": "I need a washing machine",
            "expected_category": "Washer"
        }
    ]
    
    base_url = "http://127.0.0.1:8000"
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test_case['message']} ---")
        
        try:
            # Test the chat API
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "session_id": f"test_session_{i}",
                    "message": test_case["message"],
                    "domain": "smart_support"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ API Response received")
                print(f"Reply: {result.get('reply', 'No reply')[:200]}...")
                
                # Check if response mentions washers
                reply_lower = result.get('reply', '').lower()
                if any(word in reply_lower for word in ['washer', 'washing machine', 'laundry']):
                    print("✓ Response mentions washer-related terms")
                else:
                    print("✗ Response doesn't mention washer-related terms")
                    
                # Look for product recommendations in the response
                if 'items' in result and result['items']:
                    print(f"✓ Found {len(result['items'])} product recommendations")
                    for item in result['items'][:3]:  # Show first 3 items
                        name = item.get('name', 'Unknown')
                        category = item.get('category', 'Unknown')
                        print(f"  - {name} (Category: {category})")
                else:
                    print("✗ No product recommendations found")
                    
            else:
                print(f"✗ API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("✗ Request timed out after 30 seconds")
        except Exception as e:
            print(f"✗ Error: {e}")
            
        time.sleep(1)  # Small delay between tests

if __name__ == "__main__":
    print("Testing Product Recommendations for Washers")
    print("=" * 50)
    test_washer_recommendations()