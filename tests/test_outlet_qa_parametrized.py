import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


@pytest.mark.parametrize(
    "message,expected_category",
    [
        ("Looking for a 55\" TV under RM2500", "tv"),
        ("Need an aircond for 20sqm room under RM1500", "aircond"),
        ("Vacuum under RM800", "vacuum"),
        ("Washer for family of 4 under RM1500", "washer"),
        ("Fridge under RM2000", "fridge"),
        ("Ceiling fan under RM400", "fan"),
        ("Water heater under RM450", "water_heater"),
        ("Rice cooker under RM300", "rice_cooker"),
        ("Air fryer under RM500", "air_fryer"),
    ],
)
def test_category_specific_recommendations(message, expected_category):
    payload = {"messages": [{"role": "user", "content": message}]}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    constraints = body.get("constraints") or {}
    suggested = body.get("suggested_products") or []
    # Category detection present and gating recommendations
    assert constraints.get("category") == expected_category
    assert all(p.get("category") == expected_category for p in suggested)


@pytest.mark.parametrize(
    "message",
    [
        "Saya nak TV bawah RM2000",
        "Cari aircond untuk bilik kecil",
        "Perlu vacuum yang senyap",
    ],
)
def test_language_mixed_inputs(message):
    payload = {"messages": [{"role": "user", "content": message}]}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Should produce a reply and suggested products list (when catalog available)
    assert isinstance(body.get("reply"), str)


def test_budget_parsing_and_filtering():
    payload = {"messages": [{"role": "user", "content": "Looking for a TV under RM2500"}]}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    constraints = body.get("constraints") or {}
    suggested = body.get("suggested_products") or []
    hi = constraints.get("budget_max_rm")
    if hi is not None:
        for p in suggested:
            price = p.get("price_rm")
            if price is not None:
                assert price <= hi * 1.1