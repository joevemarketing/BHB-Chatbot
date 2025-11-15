#!/usr/bin/env python3
"""Test just the KB search to see what it returns."""

import sys
sys.path.append('.')

# Import the search_kb function directly
from server import search_kb, top_products_for_message

def test_searches():
    """Test both KB search and product search separately."""
    
    message = "Recommend a 55\" 4K TV under $600"
    
    print("Testing KB search:")
    kb_results = search_kb(message, limit=1)
    if kb_results:
        print(f"KB found {len(kb_results)} results:")
        for i, result in enumerate(kb_results):
            print(f"  {i+1}. Title: {result.get('title', 'No title')}")
            print(f"     Snippet: {result.get('snippet', 'No snippet')[:100]}...")
    else:
        print("  No KB results found")
    
    print("\nTesting product search:")
    product_results = top_products_for_message(message, limit=2)
    if product_results:
        print(f"Products found {len(product_results)} results:")
        for i, result in enumerate(product_results):
            print(f"  {i+1}. Name: {result.get('name', 'No name')}")
            print(f"     Category: {result.get('category', 'No category')}")
            print(f"     Features: {result.get('features', [])[:2]}")  # First 2 features
    else:
        print("  No products found")

if __name__ == "__main__":
    test_searches()