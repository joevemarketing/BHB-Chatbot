#!/usr/bin/env python3
"""Test the format_product_for_display function with real product data."""

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

# Test with BHB real products
print("=== Testing format_product_for_display with BHB real products ===")
real_products = load_json(DATA_DIR / "bhb_products_real.json")
if real_products and len(real_products) > 0:
    # Test a TV product
    tv_product = None
    for p in real_products:
        if p.get("category") == "tv":
            tv_product = p
            break
    
    if tv_product:
        print(f"Original TV product keys: {list(tv_product.keys())}")
        print(f"Original price_rm: {tv_product.get('price_rm')}")
        
        # Format the product
        formatted = format_product_for_display({
            "name": tv_product.get("brand") + " " + tv_product.get("model_name"),
            "brand": tv_product.get("brand"),
            "price_rm": tv_product.get("price_rm"),
            "currency": "RM",
            "link": tv_product.get("bhb_product_url"),
            "features": tv_product.get("features"),
            "category": tv_product.get("category"),
        })
        
        print(f"Formatted product: {json.dumps(formatted, indent=2)}")
        print(f"Formatted price: {formatted.get('price')}")
        print(f"Formatted currency: {formatted.get('currency')}")
    else:
        print("No TV product found in real products")

# Test with regular products.json format
print("\n=== Testing format_product_for_display with regular products.json ===")
products = load_json(DATA_DIR / "products.json")
if products and len(products) > 0:
    sample = products[0]
    print(f"Original product keys: {list(sample.keys())}")
    print(f"Original price: {sample.get('price')}")
    
    formatted = format_product_for_display(sample)
    print(f"Formatted product: {json.dumps(formatted, indent=2)}")
    print(f"Formatted price: {formatted.get('price')}")
    print(f"Formatted currency: {formatted.get('currency')}")