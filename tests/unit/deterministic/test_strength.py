"""
tests/unit/deterministic/test_strength.py
-----------------------------------------
Unit tests for deterministic strength extraction, clinical dosage rules 1 & 2, and titration shielding.
"""

import pytest
from app.prescription.deterministic.strength import extract_strength_and_formulation
from app.drugs.repository import DrugRepository


@pytest.fixture(scope="module")
def drug_repo():
    return DrugRepository()


def test_dual_dose_splitting_and_titration_shielding(drug_repo):
    clause = "Take one tablet of Paracetamol 650 mg 20 mg twice daily(1-0-1) for 5 days, increase the dose by 100 mg after 7 days."
    drug_name, order_strength, span, conf = extract_strength_and_formulation(
        clause_text=clause,
        base_drug_name="Paracetamol",
        drug_repo=drug_repo,
    )
    # Formulation dose (650 mg) binds to drug name
    assert "Paracetamol 650 mg" in drug_name
    # Order strength is secondary formulation dose (20 mg), NOT titration (100 mg)
    assert order_strength == "20 mg"
    assert conf >= 0.85


def test_catalog_single_dose_binding(drug_repo):
    # Aten 50 exists in formulary DB -> 50 binds to drug name and order strength is preserved
    clause = "Administer ATEN tablet 50mg by mouth every morning before food"
    drug_name, order_strength, span, conf = extract_strength_and_formulation(
        clause_text=clause,
        base_drug_name="ATEN",
        drug_repo=drug_repo,
    )
    assert "ATEN 50mg" in drug_name or "ATEN 50 mg" in drug_name
    assert order_strength == "50mg"


def test_custom_single_dose_as_order_strength(drug_repo):
    # Vitamin C 500 mg: 500 is in DB, so binds to formulation name and order strength is preserved
    clause_vit = "Take Vitamin C 500 mg alongside it"
    drug_name, order_strength, span, conf = extract_strength_and_formulation(
        clause_text=clause_vit,
        base_drug_name="Vitamin C",
        drug_repo=drug_repo,
    )
    assert "Vitamin C 500 mg" in drug_name
    assert order_strength == "500 mg"

    # Custom strength not in DB (e.g. 875 mg): treated as order strength
    clause_custom = "Take Amoxicillin 875 mg once daily"
    drug_name2, order_strength2, span2, conf2 = extract_strength_and_formulation(
        clause_text=clause_custom,
        base_drug_name="Amoxicillin",
        drug_repo=drug_repo,
    )
    assert order_strength2 == "875 mg"
