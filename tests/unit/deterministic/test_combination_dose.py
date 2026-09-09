"""
tests/unit/deterministic/test_combination_dose.py
------------------------------------------------
Unit tests for composite dosage resolution, combination medicines,
and top-3 Did-You-Mean candidate ranking.
"""

import pytest
from app.prescription.pipeline import PrescriptionPipeline, PipelineMode
from app.drugs.repository import DrugRepository


@pytest.fixture(scope="module")
def pipeline():
    return PrescriptionPipeline()


@pytest.fixture(scope="module")
def drug_repo():
    return DrugRepository()


def test_augmentin_combination_dose_resolution(pipeline):
    """
    Doctor dictates: 'Augmentin 500 mg + 125 mg twice daily for 5 days'
    1. Composite formulation 625 mg exists in Drug_database (AUGMENTIN TAB 625MG).
    2. '500' is NOT automatically assigned as dose while dropping 125.
    3. Did-You-Mean candidates contain AUGMENTIN TAB 625MG.
    4. Schedule 'Twice a day (1-0-1)', route 'oral', and duration '5 days' are properly bound.
    """
    raw_rx = "Augmentin 500 mg + 125 mg twice daily for 5 days."
    res = pipeline.extract(raw_rx, mode=PipelineMode.FAST)

    assert len(res.items) == 1, f"Expected 1 item, got {len(res.items)}"
    item = res.items[0]

    # Dose should not be prematurely set to standalone 500
    assert item.dose != "500", "500 mg should not be set as standalone dose dropping 125 mg!"

    # Did-you-mean recommendations must suggest composite 625 formulation
    dym_names = [opt.get("drug_name", "") for opt in item.did_you_mean_options]
    if item.did_you_mean:
        dym_names.append(item.did_you_mean)

    assert any("625" in name for name in dym_names), (
        f"Expected composite formulation containing 625 in Did-You-Mean, got: {dym_names}"
    )

    # Route and timing should be resolved
    assert item.route.lower() == "oral"
    assert "twice daily" in item.frequency.lower() or "1-0-1" in item.frequency
    assert "5" in item.duration and "day" in item.duration.lower()


def test_top_3_did_you_mean_ranking_grocin(drug_repo):
    """
    Phonetic variance 'grocin' should return up to 3 similarly spelt candidates from formulary.
    """
    cands = drug_repo.find_top_did_you_mean("grocin", limit=3)
    assert len(cands) >= 1
    assert len(cands) <= 3
    assert any("crocin" in c["drug_name"].lower() or "crocin" in c["base_name"].lower() for c in cands)
    # Each candidate must contain dose, route, and available_drugs metadata
    first = cands[0]
    assert "drug_name" in first
    assert "base_name" in first
    assert "confidence" in first
    assert "available_routes" in first


def test_top_3_did_you_mean_ranking_parasita_mall(drug_repo):
    """
    Multi-token phonetic variance 'parasita mall' should ground or recommend Paracetamol formulations.
    """
    cands = drug_repo.find_top_did_you_mean("parasita mall", limit=3)
    if not cands:
        # Also test single word or normalized
        cands = drug_repo.find_top_did_you_mean("parasitamall", limit=3)

    assert len(cands) >= 1
    assert any("paracetamol" in c["drug_name"].lower() or "paracetamol" in c["base_name"].lower() for c in cands)
