"""
tests/unit/deterministic/test_duration.py
-----------------------------------------
Unit tests for deterministic duration extraction.
"""

from app.prescription.deterministic.duration import extract_duration


def test_standard_durations():
    d, span, conf = extract_duration("Take for 5 days after food")
    assert d == "5 days"

    d, span, conf = extract_duration("for 2 weeks continuously")
    assert d == "2 weeks"

    d, span, conf = extract_duration("till 7 days only")
    assert d == "7 days"


def test_single_day_and_day_one():
    d, span, conf = extract_duration("Take single dose at bedtime on day one")
    assert d == "day one"


def test_month_and_grammar():
    d, span, conf = extract_duration("re-test blood count in 2 months")
    assert d == "2 months"

    d, span, conf = extract_duration("for 5 day increase dose")
    assert d == "5 days"
