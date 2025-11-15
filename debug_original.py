#!/usr/bin/env python3
"""Debug the original product data to see what's available."""

import sys
sys.path.append('.')

from server import top_products_for_message

def debug_original_products():
    """Debug the original product data to see what's available."""
    
    message = "Recommend a 55\" 4K TV under $600"
    
    print("Testing top_products_for_message:")
    products = top_products_for_message(message, limit=2)
    
    print(f"Found {len(products)} products:")
    
    for i, product in enumerate(products):
        print(f"\nProduct {i+1} (original data):")
        print(f"  Name: {repr(product.get('name'))}")
        print(f"  Brand: {repr(product.get('brand'))}")
        print(f"  Price: {repr(product.get('price'))}")
        print(f"  Currency: {repr(product.get('currency'))}")
        print(f"  Category: {repr(product.get('category'))}")
        print(f"  Permalink: {repr(product.get('permalink'))}")
        print(f"  All keys: {list(product.keys())}")
        
        # Check if it's a BHB product
        if product.get('_source') == 'bhb.com.my':
            print("  ✓ This is a BHB product")
        else:
            print("  ? Source unknown")

if __name__ == "__main__":
    debug_original_products()