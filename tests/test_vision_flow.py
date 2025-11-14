from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_vision_chat_category_lock_washer():
    # Mock image payload with a simple PNG header
    files = {
        "image": ("test.png", b"\x89PNG\r\n\x1a\n", "image/png"),
    }
    # Provide a hint in the message so constraints capture 'washer' when vision category is absent
    data = {"message": "Photo of my washer, budget around RM1500"}

    resp = client.post("/api/vision-chat", files=files, data=data)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    constraints = body.get("constraints", {})
    suggested = body.get("suggested_products", [])

    # If category was recognized from message or vision, it must stick and gate recommendations
    if constraints.get("category"):
        assert constraints["category"] == "washer"
        assert all(p.get("category") == "washer" for p in suggested)