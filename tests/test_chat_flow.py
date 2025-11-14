from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_category_persists_across_turns():
    # Multi-turn conversation: first sets category, second only adjusts budget
    payload = {
        "messages": [
            {"role": "user", "content": "I need a washer for my family of 4."},
            {"role": "user", "content": "My budget is around RM1500."},
        ]
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    constraints = data.get("constraints") or {}
    suggested = data.get("suggested_products") or []

    # Category should stick as 'washer'
    assert constraints.get("category") == "washer"
    # All suggestions must be washers
    assert all(p.get("category") == "washer" for p in suggested)