"""
app/prescription/deterministic
------------------------------
Modular deterministic extraction engine for the Canonical Prescription Pipeline.
"""

from app.prescription.deterministic.medicine import extract_medicine_candidate, is_valid_medication_name
from app.prescription.deterministic.strength import extract_strength_and_formulation
from app.prescription.deterministic.frequency import extract_frequency
from app.prescription.deterministic.duration import extract_duration
from app.prescription.deterministic.route import extract_route
from app.prescription.deterministic.instruction import extract_instructions
from app.prescription.deterministic.engine import DeterministicEngine

__all__ = [
    "extract_medicine_candidate",
    "is_valid_medication_name",
    "extract_strength_and_formulation",
    "extract_frequency",
    "extract_duration",
    "extract_route",
    "extract_instructions",
    "DeterministicEngine",
]
