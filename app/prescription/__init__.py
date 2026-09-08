"""
app.prescription package
"""
from app.prescription.schema import (
    ExtractionStatus,
    ClinicalRoute,
    ExtractionStage,
    ValidationSeverity,
    TextSpan,
    FieldEvidence,
    PrescriptionItem,
    CanonicalPrescription,
    PipelineMetadata,
    DeterministicExtractionCandidate,
    LLMExtractionCandidate,
    ReconciledField,
    ReconciliationItem,
    ValidationFinding,
    ValidationResult,
)

__all__ = [
    "ExtractionStatus",
    "ClinicalRoute",
    "ExtractionStage",
    "ValidationSeverity",
    "TextSpan",
    "FieldEvidence",
    "PrescriptionItem",
    "CanonicalPrescription",
    "PipelineMetadata",
    "DeterministicExtractionCandidate",
    "LLMExtractionCandidate",
    "ReconciledField",
    "ReconciliationItem",
    "ValidationFinding",
    "ValidationResult",
    "PipelineMode",
    "PrescriptionPipeline",
    "normalize_prescription_text",
    "segment_prescription_clauses",
    "DeterministicEngine",
    "AgenticAdapter",
    "ClinicalReconciler",
    "ClinicalValidator",
    "CanonicalFormatter",
]

from app.prescription.pipeline import PipelineMode, PrescriptionPipeline
from app.prescription.normalizer import normalize_prescription_text
from app.prescription.segmenter import segment_prescription_clauses
from app.prescription.deterministic.engine import DeterministicEngine
from app.prescription.agentic.adapter import AgenticAdapter
from app.prescription.reconciler import ClinicalReconciler
from app.prescription.validator import ClinicalValidator
from app.prescription.formatter import CanonicalFormatter
