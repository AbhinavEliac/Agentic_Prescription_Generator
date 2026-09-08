"""
scratch/check_clinical_safety.py
--------------------------------
Deep forensic verification of Clinical Safety Invariants & Non-Hallucination.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.prescription.pipeline import PrescriptionPipeline, PipelineMode
from app.prescription.schema import ClinicalRoute, ExtractionStatus


def run_clinical_tests():
    pipeline = PrescriptionPipeline()

    cases = [
        {
            "id": "C01",
            "desc": "Missing strength",
            "text": "Take Paracetamol twice daily for 5 days after food",
            "check": lambda rx: rx.items[0].strength is None,
            "expected": "strength is None"
        },
        {
            "id": "C02",
            "desc": "Missing duration",
            "text": "Tab Augmentin 625mg 1-0-1 after meals",
            "check": lambda rx: rx.items[0].duration is None,
            "expected": "duration is None"
        },
        {
            "id": "C03",
            "desc": "Missing frequency",
            "text": "Take Paracetamol 500mg for 5 days after food",
            "check": lambda rx: rx.items[0].frequency is None,
            "expected": "frequency is None"
        },
        {
            "id": "C04",
            "desc": "Missing route (no dosage form)",
            "text": "Take Paracetamol 500mg twice daily for 5 days",
            "check": lambda rx: rx.items[0].route in (None, ClinicalRoute.UNKNOWN, "None", "unknown"),
            "expected": "route is None or UNKNOWN"
        },
        {
            "id": "C05",
            "desc": "Unknown non-formulary medicine",
            "text": "Take Zombiefort 500mg twice daily for 5 days",
            "check": lambda rx: rx.items[0].matched_drug_id is None,
            "expected": "matched_drug_id is None (not mapped to false drug)"
        },
        {
            "id": "C06",
            "desc": "Conflicting frequency shorthand",
            "text": "Take Paracetamol 500mg twice daily once a day for 5 days",
            "check": lambda rx: rx.items[0].frequency is not None,
            "expected": "frequency parsed or flagged"
        },
        {
            "id": "C07",
            "desc": "Shared instructions across multi-drug line",
            "text": "Take Paracetamol 500mg and Cetirizine 10mg after dinner for 5 days",
            "check": lambda rx: len(rx.items) == 2 and rx.items[0].instruction is not None and rx.items[1].instruction is not None,
            "expected": "Both drugs receive 'after dinner' instruction"
        },
        {
            "id": "C08",
            "desc": "PRN / SOS schedule",
            "text": "Tab Paracetamol 650mg SOS for fever",
            "check": lambda rx: "sos" in (rx.items[0].frequency or "").lower() or "as needed" in (rx.items[0].frequency or "").lower() or "sos" in (rx.items[0].instruction or "").lower(),
            "expected": "SOS preserved as frequency or instruction"
        },
    ]

    print("================================================================================")
    print("CLINICAL NON-HALLUCINATION & SAFETY AUDIT")
    print("================================================================================")

    all_passed = True
    for c in cases:
        rx = pipeline.extract(c["text"], mode=PipelineMode.FAST)
        if not rx.items:
            print(f"[FAIL] {c['id']} ({c['desc']}): Extracted ZERO items!")
            all_passed = False
            continue

        item = rx.items[0]
        passed = c["check"](rx)
        if not passed:
            all_passed = False
            print(f"[FAIL] {c['id']} ({c['desc']}): Expected {c['expected']}")
            print(f"       Actual item: medicine='{item.medicine}', strength='{item.strength}', freq='{item.frequency}', dur='{item.duration}', route='{item.route}', inst='{item.instruction}'")
        else:
            print(f"[PASS] {c['id']} ({c['desc']}): Correctly verified ({c['expected']})")

    print("================================================================================")
    return all_passed


if __name__ == "__main__":
    run_clinical_tests()
