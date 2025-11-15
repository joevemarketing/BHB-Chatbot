from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_service_golden_answer_contains_faq():
    payload = {
        "messages": [
            {"role": "user", "content": "What is your return policy?"}
        ]
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body.get("reply"), str)
    assert "check bhb.com.my" in (body.get("reply") or "")
    assert body.get("suggested_products") in ([], None)


def test_general_enquiry_has_no_products():
    payload = {
        "messages": [
            {"role": "user", "content": "I need some advice for my new place"}
        ]
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    constraints = body.get("constraints") or {}
    assert constraints.get("category") in (None, "")


def test_feedback_endpoint_accepts_submission():
    from server import app as legacy_app
    c = TestClient(legacy_app)
    payload = {
        "session_id": "test-session",
        "message": "Washer sizing answer was incorrect",
        "reply": "...",
        "correct": False,
        "notes": "Suggested 6kg for family of 5",
        "category": "washer"
    }
    resp = c.post("/api/feedback", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json().get("ok") is True


def test_admin_retrain_requires_auth():
    from server import app as legacy_app
    c = TestClient(legacy_app)
    resp = c.post("/api/admin/retrain-kb")
    assert resp.status_code == 403


def test_locked_endpoints_booking_crm_payments_absent():
    for path in [
        "/api/booking",
        "/api/crm-sync",
        "/api/payments/checkout",
    ]:
        r = client.post(path)
        assert r.status_code in (404, 405)
