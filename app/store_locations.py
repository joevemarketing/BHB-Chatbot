import json
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel


DATA_PATH = os.path.join("data", "bhb_store_locations.json")


class StoreLocation(BaseModel):
    id: str
    brand_shop: Optional[str] = None
    label: str
    address: str
    city: str
    state: str
    tel: str
    hours: str


@lru_cache(maxsize=1)
def load_store_locations() -> List[StoreLocation]:
    """Load and cache store locations from JSON.
    Returns an empty list if file missing or invalid.
    """
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [StoreLocation(**item) for item in data]
    except Exception as e:
        print(f"[store_locations] Failed to load store locations: {e}")
        return []


def find_stores_by_query(q: str) -> List[StoreLocation]:
    """Simple text search over label/address/city/state.
    Splits query into tokens and returns stores where any token appears.
    """
    q_lower = (q or "").lower()
    tokens = q_lower.split()
    stores = load_store_locations()
    results: List[StoreLocation] = []
    for s in stores:
        haystack = " ".join([s.label, s.address, s.city, s.state]).lower()
        if any(tok in haystack for tok in tokens):
            results.append(s)
    return results