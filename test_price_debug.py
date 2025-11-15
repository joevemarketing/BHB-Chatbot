#!/usr/bin/env python3
"""Debug script to check price data structure in products."""

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

# Load different product files to check price structure
print("=== Checking bhb_products_real.json ===")
real_products = load_json(DATA_DIR / "bhb_products_real.json")
if real_products and len(real_products) > 0:
    sample = real_products[0]
    print(f"Sample product keys: {list(sample.keys())}")
    print(f"Price field: {sample.get('price_rm', 'NOT FOUND')}")
    print(f"Full sample: {json.dumps(sample, indent=2)}")

print("\n=== Checking products.json ===")
products = load_json(DATA_DIR / "products.json")
if products and len(products) > 0:
    sample = products[0]
    print(f"Sample product keys: {list(sample.keys())}")
    print(f"Price field: {sample.get('price', 'NOT FOUND')}")
    print(f"Currency field: {sample.get('currency', 'NOT FOUND')}")

print("\n=== Checking bhb_demo_products.json ===")
demo_products = load_json(DATA_DIR / "bhb_demo_products.json")
if demo_products and len(demo_products) > 0:
    sample = demo_products[0]
    print(f"Sample product keys: {list(sample.keys())}")
    print(f"Price field: {sample.get('price', 'NOT FOUND')}")
    print(f"Currency field: {sample.get('currency', 'NOT FOUND')}")