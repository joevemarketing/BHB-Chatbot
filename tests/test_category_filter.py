import json
from typing import List

from fastapi.testclient import TestClient

from app.main import app
from app.models import Product, ChatResponse


client = TestClient(app)


def _categories(items: List[dict]) -> List[str]:
    return [p.get("category") for p in items]


def test_chat_hard_category_filter_washer():
    # Simulate a user asking for a washer
    payload = {
        "messages": [
            {"role": "user", "content": "Looking for a washer around RM 1500"}
        ]
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "suggested_products" in data
    products = data["suggested_products"]
    constraints = data.get("constraints", {})
    # If category is set, ensure all suggested products match
    if constraints.get("category"):
        assert all(p.get("category") == constraints["category"] for p in products)