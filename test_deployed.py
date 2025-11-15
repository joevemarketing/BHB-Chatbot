#!/usr/bin/env python3
"""Test the deployed version to verify washer recommendations are working."""

import requests
import json

def test_deployed_version():
    """Test the deployed Render version."""
    
    # The Render deployment URL (you may need to update this)
    base_url = "https://retail-chatbot.onrender.com"  # Default Render URL pattern
    
    print("Testing Deployed Version")
    print("=" * 40)
    
    test_cases = [
        "What's the difference between front-load and top-load washers?",
        "recommend washers",
        "I need a washing machine under RM2000"
    ]
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {message} ---")
        
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "session_id": f"deploy_test_{i}",
                    "message": message,
                    "domain": "smart_support"
                },
                timeout=45  # Render can be slower on free tier
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✓ API Response received")
                
                # Check if response mentions washers
                reply = result.get('reply', '').lower()
                if any(word in reply for word in ['washer', 'washing machine', 'laundry']):
                    print("✓ Response mentions washer-related terms")
                else:
                    print("? Response may not mention washers")
                
                # Check for products
                if 'items' in result and result['items']:
                    print(f"✓ Found {len(result['items'])} product recommendations")
                    for j, item in enumerate(result['items'][:3]):
                        name = item.get('name', 'Unknown')
                        category = item.get('category', 'Unknown')
                        print(f"  {j+1}. {name} (Category: {category})")
                else:
                    print("? No products found in response")
                    
            else:
                print(f"✗ API Error: {response.status_code}")
                if response.status_code == 503:
                    print("  Service may be starting up, try again in 30 seconds")
                elif response.status_code == 502:
                    print("  Bad gateway - deployment may still be in progress")
                    
        except requests.exceptions.Timeout:
            print("✗ Request timed out (45s) - Render may be starting up")
        except requests.exceptions.ConnectionError:
            print("✗ Connection failed - service may not be ready yet")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print()  # Empty line for readability

if __name__ == "__main__":
    test_deployed_version()