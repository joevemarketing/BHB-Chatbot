import json
import os
from functools import lru_cache
from typing import List

from .models import FAQItem


FAQ_PATH = os.path.join("data", "bhb_customer_service_faq.json")


@lru_cache(maxsize=1)
def load_faq() -> List[FAQItem]:
    """Load and cache customer service FAQ items from JSON.
    Returns an empty list if file missing or invalid.
    """
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [FAQItem(**item) for item in data]
    except Exception as e:
        print(f"[service_faq] Failed to load FAQ: {e}")
        return []


def find_relevant_faq(user_text: str) -> List[FAQItem]:
    """Simple keyword matcher: return FAQ items whose keywords appear in user_text."""
    t = (user_text or "").lower()
    items = load_faq()
    hits: List[FAQItem] = []
    for item in items:
        for kw in item.keywords:
            if kw.lower() in t:
                hits.append(item)
                break
    return hits