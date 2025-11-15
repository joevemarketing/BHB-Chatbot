#!/usr/bin/env python3
"""Test the format function with actual TV product data."""

import sys
import os
sys.path.append('.')

# Import the function from server
from server import format_product_for_display
import json
from pathlib import Path

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

DATA_DIR = Path("data")

# Test with TV products from main products.json
print("=== Testing format_product_for_display with TV products ===")
products = load_json(DATA_DIR / "products.json")
if products:
    # Find TV products
    tv_products = [p for p in products if p and (("tv" in (p.get("category") or "").lower()) or ("tv" in (p.get("name") or "").lower()) or ("television" in (p.get("name") or "").lower()))]
    
    if tv_products:
        print(f"Found {len(tv_products)} TV products")
        sample = tv_products[0]
        print(f"Original TV product: {json.dumps(sample, indent=2)}")
        
        # Format the product
        formatted = format_product_for_display(sample)
        print(f"\nFormatted TV product: {json.dumps(formatted, indent=2)}")
        print(f"Price display: {formatted.get('currency')}{formatted.get('price')}")
    else:
        print("No TV products found in main products.json")
        
        # Show first few products to see what we have
        print(f"\nFirst 3 products in file:")
        for i, p in enumerate(products[:3]):
            print(f"  {i+1}. {p.get('name')} - {p.get('category')} - {p.get('currency')}{p.get('price')}")