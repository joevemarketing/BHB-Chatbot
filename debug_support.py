#!/usr/bin/env python3
"""Debug the customer support function to see what's being returned."""

import requests
import json

def debug_customer_support():
    """Test just the customer support function."""
    
    test_message = "Recommend a 55\" 4K TV under $600"
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/chat",
            json={
                "session_id": "debug_test",
                "message": test_message,
                "domain": "customer_support"  # Use customer_support to isolate the function
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Message: {test_message}")
            print(f"Domain: customer_support")
            print(f"Reply: {repr(result.get('reply', 'No reply'))}")
            print()
            
            # Check if the reply contains raw product data
            reply = result.get('reply', '')
            if '{' in reply and '}' in reply:
                print("⚠️  Reply contains braces - likely raw product data!")
                print("First 200 chars of reply:")
                print(reply[:200])
            else:
                print("✓ Reply looks clean")
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_customer_support()