"""
tests/unit/deterministic/test_route.py
--------------------------------------
Unit tests for deterministic route extraction and clinical priority resolution.
"""

from app.prescription.deterministic.route import extract_route


def test_specialized_routes():
    r, _, _, _ = extract_route("spray Fluticasone 50 mcg nasal spray into both nostrils")
    assert r == "nasal"

    r, _, _, _ = extract_route("instill Moxifloxacin 0.5% eye drops into left eye every 4 hours")
    assert r == "ophthalmic"

    r, _, _, _ = extract_route("instill 2 drops into the affected ear twice daily")
    assert r == "otic"

    r, _, _, _ = extract_route("Inhale Budecort 200 mcg Rotacap twice daily")
    assert r == "inhalation"

    r, _, _, _ = extract_route("Apply Clotrimazole 1% cream topically twice daily")
    assert r == "topical"

    r, _, _, _ = extract_route("Take one tablet orally after food")
    assert r == "oral"

