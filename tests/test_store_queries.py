from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_store_query_air_itam_penang():
    payload = {"messages": [{"role": "user", "content": "Store near Air Itam Penang"}]}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("reply"), str)
    assert body.get("suggested_products") in ([], None)


def test_store_query_kuala_lumpur():
    payload = {"messages": [{"role": "user", "content": "Nearest outlet in Kuala Lumpur"}]}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("reply"), str)
    assert body.get("suggested_products") in ([], None)