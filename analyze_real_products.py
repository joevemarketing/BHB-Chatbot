#!/usr/bin/env python3
import json

# Load the real products file
with open('data/bhb_products_real.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total products: {len(data)}")

# Count categories
categories = {}
for product in data:
    cat = product.get('category', 'unknown')
    categories[cat] = categories.get(cat, 0) + 1

print('\nCategories:')
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")

# Show some examples
print('\nSample products:')
for cat in ['fan', 'water_heater', 'air_conditioner'][:3]:
    products = [p for p in data if p.get('category') == cat]
    if products:
        print(f"\n{cat} examples:")
        for p in products[:2]:
            print(f"  - {p.get('brand', 'Unknown')} {p.get('model_name', 'Unknown')} (RM{p.get('price_rm', 'N/A')})")