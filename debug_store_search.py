#!/usr/bin/env python3
"""Debug the store search functionality."""

import json
from pathlib import Path

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def find_stores_by_query(q: str) -> list:
    """Simple text search over label/address/city/state for store locations."""
    q_lower = (q or "").lower()
    tokens = q_lower.split()
    results = []
    
    DATA_DIR = Path("data")
    store_locations = load_json(DATA_DIR / "bhb_store_locations.json") or []
    
    for store in store_locations:
        haystack = " ".join([
            store.get("label", ""),
            store.get("address", ""),
            store.get("city", ""),
            store.get("state", "")
        ]).lower()
        
        # Debug: show what we're searching in
        if any(tok in haystack for tok in tokens):
            print(f"MATCH FOUND for '{q}' in store: {store.get('label')}")
            print(f"  Haystack: '{haystack[:100]}...'")
            print(f"  Tokens: {tokens}")
            results.append(store)
        else:
            print(f"NO MATCH for '{q}' in store: {store.get('label')}")
            print(f"  Haystack: '{haystack[:100]}...'")
            print(f"  Tokens: {tokens}")
    
    return results

def test_store_search():
    """Test the store search with various queries."""
    
    test_queries = [
        "Where is the BHB store near me?",
        "BHB store location near me",
        "BHB store address"
    ]
    
    print("Testing Store Search Functionality")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n--- Testing: '{query}' ---")
        stores = find_stores_by_query(query)
        print(f"Found {len(stores)} stores")
        
        if stores:
            for store in stores[:2]:
                print(f"  - {store.get('label')} ({store.get('city')}, {store.get('state')})")
        print()

if __name__ == "__main__":
    test_store_search()