#!/usr/bin/env python3
"""Check what categories are available in BHB real products."""

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

# Check BHB real products categories
print("=== BHB Real Products Categories ===")
real_products = load_json(DATA_DIR / "bhb_products_real.json")
if real_products:
    categories = {}
    for p in real_products:
        cat = p.get("category", "unknown")
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print("Categories found:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} products")
    
    # Find a TV product (if exists)
    tv_products = [p for p in real_products if "tv" in p.get("category", "").lower()]
    if tv_products:
        print(f"\nFound {len(tv_products)} TV products")
        sample = tv_products[0]
        print(f"Sample TV product: {sample.get('brand')} {sample.get('model_name')}")
        print(f"Price: RM{sample.get('price_rm')}")
    else:
        print("\nNo TV products found in bhb_products_real.json")
        
        # Show sample of any product
        if real_products:
            sample = real_products[0]
            print(f"\nSample product: {sample.get('brand')} {sample.get('model_name')}")
            print(f"Category: {sample.get('category')}")
            print(f"Price: RM{sample.get('price_rm')}")