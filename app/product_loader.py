import json
import os
from functools import lru_cache
from typing import List

from .models import Product


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEMO_PATH = os.path.join(BASE_DIR, "data", "bhb_demo_products.json")
REAL_PATH = os.path.join(BASE_DIR, "data", "bhb_products_real.json")


@lru_cache(maxsize=1)
def _load_products() -> List[Product]:
    items: List[dict] = []
    # Load demo products (always present in repo)
    if os.path.exists(DEMO_PATH):
        with open(DEMO_PATH, "r", encoding="utf-8") as f:
            items.extend(json.load(f))
    # Optionally load real products and append
    if os.path.exists(REAL_PATH):
        try:
            with open(REAL_PATH, "r", encoding="utf-8") as f:
                real_items = json.load(f)
                if isinstance(real_items, list):
                    items.extend(real_items)
        except Exception:
            # If malformed, ignore to avoid breaking demo
            pass
    return [Product(**item) for item in items]


def get_all_products() -> List[Product]:
    return _load_products()