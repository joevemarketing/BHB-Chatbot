from server import handle_smart_support


def test_daikin_aircon_acknowledgement():
    reply, items = handle_smart_support("Do u have Daikin aircon?")
    assert isinstance(reply, str)
    # Expect Daikin acknowledgement and sizing prompt
    assert "Daikin air conditioners" in reply
    assert "room size" in reply


def test_aircon_excludes_purifier_results():
    reply, items = handle_smart_support("Looking for air conditioner")
    # No item name should include purifier when intent is air conditioner
    for it in items:
        assert "purifier" not in (it.get("name") or "").lower()


def test_policy_notes_appended_when_mentioned():
    msg = "Do u have Daikin aircon? Return Policy and Extended Warranty info"
    reply, items = handle_smart_support(msg)
    assert "Return Policy:" in reply
    assert "Extended Warranty:" in reply