#!/usr/bin/env python3
"""Test the improved product formatting for TV recommendations."""

import requests
import json

def test_tv_recommendations():
    """Test TV recommendations with improved formatting."""
    
    print("Testing Improved TV Recommendation Formatting")
    print("=" * 50)
    
    test_message = "Recommend a 55\" 4K TV under $600"
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/chat",
            json={
                "session_id": "tv_format_test",
                "message": test_message,
                "domain": "smart_support"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Message: {test_message}")
            print(f"Reply: {result.get('reply', 'No reply')}")
            print()
            
            if 'items' in result and result['items']:
                print(f"Found {len(result['items'])} products:")
                print()
                
                for i, item in enumerate(result['items'][:3]):
                    print(f"{i+1}. {item.get('name', 'Unknown')}")
                    print(f"   Brand: {item.get('brand', 'Unknown')}")
                    print(f"   Price: {item.get('currency', 'RM')}{item.get('price', 'N/A')}")
                    print(f"   Category: {item.get('category', 'Unknown')}")
                    
                    features = item.get('features', [])
                    if features:
                        print(f"   Features: {', '.join(features)}")
                    else:
                        print("   Features: None listed")
                    
                    # Check if formatting worked (no braces or technical jargon)
                    raw_name = str(item.get('name', ''))
                    if '{' in raw_name or '}' in raw_name:
                        print("   ⚠️  Warning: Raw data still contains braces!")
                    
                    print()
            else:
                print("No products found in response")
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_tv_recommendations()