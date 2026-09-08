"""
tests/unit/deterministic/test_instruction.py
--------------------------------------------
Unit tests for deterministic primary and additional instruction extraction.
"""

from app.prescription.deterministic.instruction import extract_instructions


def test_primary_meal_and_device_instructions():
    p, a, conf = extract_instructions("Take tablet once daily before breakfast for 10 days")
    assert p == "before breakfast"

    p, a, conf = extract_instructions("Take Metformin strictly with meals to avoid stomach upset")
    assert "strictly with meals" in p

    p, a, conf = extract_instructions("Inhale Rotacap twice daily. Rinse mouth after using the inhaler.")
    assert "rinse mouth" in p.lower()


def test_additional_clinical_instructions():
    p, a, conf = extract_instructions(
        "Take Amoxicillin 500 mg for 7 days. Do not stop the antibiotic course early even if fever subsides."
    )
    assert a is not None
    assert "antibiotic course" in a.lower()

    p, a, conf = extract_instructions(
        "Take Paracetamol 650 mg. Discontinue once the fever resolves. Seek reassessment if adverse effects develop."
    )
    assert a is not None
    assert "fever resolves" in a.lower()
    assert "adverse effects" in a.lower()


def test_diet_and_lifestyle_instructions():
    full_text = "Take medicines. Include dark green leafy vegetables in diet, avoid tea near meal times, and go for morning walks daily."
    p, a, conf = extract_instructions(
        "Take Ferrous ascorbate",
        full_prescription_text=full_text,
    )
    assert a is not None
    assert "vegetables" in a.lower() or "tea" in a.lower() or "walk" in a.lower()
