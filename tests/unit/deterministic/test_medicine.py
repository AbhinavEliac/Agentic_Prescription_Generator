"""
tests/unit/deterministic/test_medicine.py
-----------------------------------------
Unit tests for deterministic drug name extraction and noise filtering.
"""

import pytest
from app.prescription.deterministic.medicine import (
    extract_medicine_candidate,
    is_valid_medication_name,
    clean_candidate_name,
)
from app.drugs.repository import DrugRepository


@pytest.fixture(scope="module")
def drug_repo():
    return DrugRepository()


def test_valid_medication_filter():
    assert is_valid_medication_name("Paracetamol") is True
    assert is_valid_medication_name("Aten 50") is True
    assert is_valid_medication_name("Amoxicillin") is True
    assert is_valid_medication_name("Dolo 650") is True


def test_invalid_medication_filter():
    assert is_valid_medication_name("doctor") is False
    assert is_valid_medication_name("patient") is False
    assert is_valid_medication_name("endoscopy report") is False
    assert is_valid_medication_name("blood pressure") is False
    assert is_valid_medication_name("fomentation") is False
    assert is_valid_medication_name("ice packs") is False
    assert is_valid_medication_name("hot water") is False
    assert is_valid_medication_name("") is False
    assert is_valid_medication_name("NONE") is False


def test_clean_candidate_name():
    assert clean_candidate_name("Take one tablet of Paracetamol") == "Paracetamol"
    assert clean_candidate_name("Administer one capsule of Amoxicillin") == "Amoxicillin"
    assert clean_candidate_name("apply a thin layer of Soframycin") == "Soframycin"


def test_extract_medicine_with_seed(drug_repo):
    med, span, conf = extract_medicine_candidate(
        "Take one tablet of Paracetamol 650 mg",
        seed_drug_name="Paracetamol",
        drug_repo=drug_repo,
    )
    assert med == "Paracetamol"
    assert conf >= 0.90


def test_extract_medicine_from_clause(drug_repo):
    med, span, conf = extract_medicine_candidate(
        "Take Pantop 40 mg once daily before breakfast",
        drug_repo=drug_repo,
    )
    assert "Pantop" in med or "PANTOP" in med.upper()
    assert conf >= 0.85
