#!/usr/bin/env python3
"""Test the server response with a TV recommendation query."""

import requests
import json

# Test the server with a TV recommendation query
query = "Recommend a 55\" 4K TV under $600"

print(f"Testing query: {query}")
print("=" * 50)

try:
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={"session_id": "test-session-123", "message": query},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Server Response:")
        print(json.dumps(result, indent=2))
        
        # Check if there are product recommendations
        if "products" in result:
            print(f"\nFound {len(result['products'])} products:")
            for i, product in enumerate(result["products"]):
                print(f"  {i+1}. {product.get('name', 'Unknown')}")
                print(f"     Price: {product.get('currency', 'RM')}{product.get('price', 'None')}")
                print(f"     Brand: {product.get('brand', 'Unknown')}")
                print(f"     Features: {product.get('features', [])}")
                print()
        else:
            print("\nNo products found in response")
            
    else:
        print(f"Server returned status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Error connecting to server: {e}")
    print("Make sure the server is running on localhost:8000")