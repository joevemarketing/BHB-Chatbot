from fastapi.testclient import TestClient

from server import app


client = TestClient(app)


def test_import_products_requires_auth():
    resp = client.post("/api/admin/import-products", json={"categories": ["tv"]})
    assert resp.status_code == 403


def test_import_kb_requires_auth():
    resp = client.post("/api/admin/import-kb-from-bhb", json={"urls": ["https://www.bhb.com.my/"]})
    assert resp.status_code == 403