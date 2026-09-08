"""
tests/unit/test_reconciler.py
-----------------------------
Unit tests for deterministic reconciliation between deterministic and LLM models.
"""

from app.prescription.schema import (
    DeterministicExtractionCandidate,
    LLMExtractionCandidate,
    ExtractionStatus,
)
from app.prescription.reconciler import ClinicalReconciler


def test_reconciler_agreement():
    reconciler = ClinicalReconciler()
    det = [
        DeterministicExtractionCandidate(
            medicine_name="Paracetamol 650 mg",
            strength="500 mg",
            frequency="twice daily",
            duration="5 days",
            route="oral",
        )
    ]
    llm = [
        LLMExtractionCandidate(
            medicine_name="Paracetamol 650 mg",
            strength="500 mg",
            frequency="twice daily",
            duration="5 days",
            route="oral",
        )
    ]
    reconciled = reconciler.reconcile(det, llm, source_text="Take Paracetamol 650 mg 500 mg twice daily for 5 days")
    assert len(reconciled) == 1
    assert reconciled[0].status == ExtractionStatus.NORMALIZED
    assert reconciled[0].confidence >= 0.95
    assert len(reconciled[0].conflicts) == 0


def test_reconciler_conflict_resolution():
    reconciler = ClinicalReconciler()
    det = [
        DeterministicExtractionCandidate(
            medicine_name="Amoxicillin",
            strength="500 mg",
            frequency="TID",
            duration="7 days",
            route="oral",
        )
    ]
    # LLM hallucinates/differs on strength (250 mg)
    llm = [
        LLMExtractionCandidate(
            medicine_name="Amoxicillin",
            strength="250 mg",
            frequency="TID",
            duration="7 days",
            route="oral",
        )
    ]
    reconciled = reconciler.reconcile(det, llm, source_text="Take Amoxicillin 500 mg TID for 7 days")
    assert len(reconciled) == 1
    # Must NOT arbitrarily choose LLM!
    assert reconciled[0].status == ExtractionStatus.CONFLICT
    assert len(reconciled[0].conflicts) > 0
    assert "strength" in reconciled[0].conflicts[0]
    # Retains deterministic value as safe anchor
    assert reconciled[0].strength == "500 mg"
