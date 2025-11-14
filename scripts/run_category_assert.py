from fastapi.testclient import TestClient

from app.main import app


def run():
    client = TestClient(app)
    payload = {
        "messages": [
            {"role": "user", "content": "Looking for a washer around RM 1500"}
        ]
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    products = data.get("suggested_products", [])
    constraints = data.get("constraints", {})
    category = constraints.get("category")
    if category:
        assert all(p.get("category") == category for p in products), (
            f"Cross-category leak detected: { [p.get('category') for p in products] } vs {category}"
        )
    print("Category assertion passed. Suggested products:")
    for p in products:
        print(f"- {p.get('brand')} {p.get('model_name')} ({p.get('category')}) RM {p.get('price_rm')}")


if __name__ == "__main__":
    run()