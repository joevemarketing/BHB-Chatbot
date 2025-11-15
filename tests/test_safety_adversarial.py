from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_no_urls_in_recommendations():
    payload = {"messages": [{"role": "user", "content": "Recommend a TV under RM2000"}]}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("reply"), str)
    assert "http://" not in body.get("reply") and "https://" not in body.get("reply")


def test_multi_category_bundle():
    payload = {"messages": [{"role": "user", "content": "Need TV and washer for a new house"}]}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Should produce a bundle-style reply and a non-empty suggested products list
    assert isinstance(body.get("reply"), str)
    assert isinstance(body.get("suggested_products"), list)