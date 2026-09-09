"""
tests/unit/deterministic/test_frequency.py
------------------------------------------
Unit tests for deterministic frequency extractor and speaking habit schedule normalization.
"""

from app.prescription.deterministic.frequency import extract_frequency


def test_numeric_doctor_schedules():
    f, span, conf = extract_frequency("Take tablet 650 mg twice daily(1-0-1) for 5 days")
    assert "1-0-1" in f

    f, span, conf = extract_frequency("Take tablet once daily(0-0-1) before breakfast")
    assert "0-0-1" in f or "bedtime" in f.lower()

    f, span, conf = extract_frequency("Take tablet 101 after food")
    assert "1-0-1" in f

    f, span, conf = extract_frequency("Take capsule 111 after lunch")
    assert "1-1-1" in f


def test_natural_language_frequencies():
    f, span, conf = extract_frequency("Take tablet every 6 hours for 3 days")
    assert f == "every 6 hours"

    f, span, conf = extract_frequency("Administer tablet every morning before food")
    assert f == "every morning"

    f, span, conf = extract_frequency("Take Stugeron Forte twice daily Morning Night for 2 weeks")
    assert "twice daily" in f.lower() or "1-0-1" in f


def test_coreference_broadcast():
    full_text = "Take Drug A. Take Drug B. Both medicines should be taken twice daily for 5 days."
    f, span, conf = extract_frequency("Take Drug A", full_prescription_text=full_text)
    assert "twice" in f.lower() or "1-0-1" in f
