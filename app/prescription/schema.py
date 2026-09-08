"""
app/prescription/schema.py
---------------------------
Canonical Data Contracts & Domain Models for the Agentic Prescription Extractor.

Principles:
1. Single source of truth for all prescription domain models.
2. Missing information MUST remain null (None) or 'UNKNOWN' - never fabricated.
3. Every clinically relevant extracted field is traceable to source text via FieldEvidence.
4. Strongly typed intermediate models for deterministic extraction, LLM extraction,
   reconciliation, and validation stages.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import datetime
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# 1. STATUS & ENUM DEFINITIONS
# ============================================================================

class ExtractionStatus(str, Enum):
    """Lifecycle and audit status for extracted items and pipelines."""
    UNKNOWN = "UNKNOWN"
    EXTRACTED = "EXTRACTED"
    NORMALIZED = "NORMALIZED"
    VALIDATED = "VALIDATED"
    CONFLICT = "CONFLICT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    ERROR = "ERROR"


class ClinicalRoute(str, Enum):
    """Anatomical route of administration."""
    ORAL = "oral"
    INHALATION = "inhalation"
    TOPICAL = "topical"
    NASAL = "nasal"
    OPHTHALMIC = "ophthalmic"
    OTIC = "otic"
    RECTAL = "rectal"
    INTRAVENOUS = "intravenous"
    INTRAMUSCULAR = "intramuscular"
    SUBLINGUAL = "sublingual"
    TRANSDERMAL = "transdermal"
    UNKNOWN = "unknown"


class ExtractionStage(str, Enum):
    """Extraction stage identifier for audit trails."""
    NORMALIZATION = "normalization"
    SEGMENTATION = "segmentation"
    DETERMINISTIC = "deterministic"
    LLM_AGENT = "llm_agent"
    RECONCILIATION = "reconciliation"
    VALIDATION = "validation"
    FORMATTING = "formatting"


class ValidationSeverity(str, Enum):
    """Severity classification for validation findings."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ============================================================================
# 2. EVIDENCE & TRACEABILITY MODELS
# ============================================================================

class TextSpan(BaseModel):
    """Character offset span in the source text."""
    start: Optional[int] = Field(default=None, ge=0, description="0-indexed start character offset")
    end: Optional[int] = Field(default=None, ge=0, description="0-indexed end character offset")
    verbatim: str = Field(default="", description="Exact verbatim text snippet from the raw source")
    text: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_verbatim_and_text(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "text" in data and not data.get("verbatim"):
                data["verbatim"] = data["text"]
            elif "verbatim" in data and not data.get("text"):
                data["text"] = data["verbatim"]
        return data

    @model_validator(mode="after")
    def validate_span_offsets(self) -> TextSpan:
        if not self.verbatim and self.text:
            self.verbatim = self.text
        if not self.text and self.verbatim:
            self.text = self.verbatim
        if self.start is not None and self.end is not None:
            if self.end < self.start:
                raise ValueError(f"Span end offset ({self.end}) cannot precede start offset ({self.start})")
        return self


class FieldEvidence(BaseModel):
    """Traceable provenance link binding an extracted field to its source text."""
    field_name: str = Field(..., description="Name of the extracted clinical field")
    source_text: str = Field(default="", description="Raw text segment or verbatim phrase providing evidence")
    value: Optional[str] = None
    span: Optional[TextSpan] = Field(default=None, description="Precise character offsets in the raw input")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score for this extraction")
    source_model: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_source_text(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "value" in data and not data.get("source_text"):
                data["source_text"] = str(data["value"])
            elif "source_text" in data and not data.get("value"):
                data["value"] = str(data["source_text"])
        return data

    @classmethod
    def from_text(cls, field_name: str, source_text: str, confidence: Optional[float] = None) -> FieldEvidence:
        return cls(
            field_name=field_name,
            source_text=source_text,
            value=source_text,
            confidence=confidence,
        )


# ============================================================================
# 3. CANONICAL PRESCRIPTION ITEM & PRESCRIPTION
# ============================================================================

class PrescriptionItem(BaseModel):
    """
    Canonical strongly-typed representation of a single prescribed medication line item.
    
    Invariants:
    - Missing fields MUST be None (null) or UNKNOWN.
    - No fabricated or synthetic placeholders (e.g. 'NONE' strings are prohibited).
    """
    item_id: int = Field(default=1, ge=1, description="1-indexed sequence identifier for this prescription item")
    
    # Core Clinical Fields (Nullable - strictly no fabricated values)
    medicine: str = Field(default="", min_length=1, description="Canonical or identified medication/active substance name")
    medicine_name: Optional[str] = None
    raw_medicine_name: Optional[str] = Field(default=None, description="Verbatim medicine token spoken/written in raw input")
    matched_drug_id: Optional[str] = Field(default=None, description="Formulary drug ID from DrugRepository if matched")
    
    strength: Optional[str] = Field(default=None, description="Formulation strength (e.g., '650 mg', '50 mcg'). Null if not stated")
    dose: Optional[str] = Field(default=None, description="Numeric dose value or count (e.g., '1', '650'). Null if not stated")
    dose_unit: Optional[str] = Field(default=None, description="Clinical unit of measurement (e.g., 'mg', 'ml', 'tablet'). Null if not stated")
    
    frequency: Optional[str] = Field(default=None, description="Administration schedule (e.g., 'twice daily (1-0-1)'). Null if not stated")
    duration: Optional[str] = Field(default=None, description="Duration span (e.g., '5 days', '2 weeks'). Null if not stated")
    route: Any = Field(default=ClinicalRoute.UNKNOWN, description="Anatomical route of delivery")
    
    instruction: Optional[str] = Field(default=None, description="Primary administration instructions (meals, devices). Null if not stated")
    additional_instruction: Optional[str] = Field(default=None, description="Secondary clinical instructions, titrations, contingency advice. Null if not stated")
    
    # Audit, Quality & Provenance
    status: ExtractionStatus = Field(default=ExtractionStatus.EXTRACTED, description="Current lifecycle status of this item")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall item confidence score")
    evidence: Dict[str, FieldEvidence] = Field(default_factory=dict, description="Field-level provenance mapping")
    
    # Formulary relational links
    available_routes: List[str] = Field(default_factory=list, description="Permissible anatomical routes mapped in formulary")
    validation_warnings: List[str] = Field(default_factory=list, description="Non-blocking clinical warnings or ambiguities")
    review_reasons: List[str] = Field(default_factory=list, description="Reasons flagging this item for review")

    @model_validator(mode="before")
    @classmethod
    def sync_medicine_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "medicine_name" in data and not data.get("medicine"):
                data["medicine"] = data["medicine_name"]
            elif "medicine" in data and not data.get("medicine_name"):
                data["medicine_name"] = data["medicine"]
        return data

    @model_validator(mode="after")
    def sync_medicine_after(self) -> PrescriptionItem:
        if not self.medicine and self.medicine_name:
            self.medicine = self.medicine_name
        if not self.medicine_name and self.medicine:
            self.medicine_name = self.medicine
        return self

    @field_validator("medicine")
    @classmethod
    def validate_medicine_not_placeholder(cls, v: str) -> str:
        clean = v.strip()
        if not clean or clean.upper() in {"NONE", "NULL", "UNKNOWN", "N/A"}:
            raise ValueError(f"Medicine name cannot be a placeholder: '{v}'")
        return clean

    @field_validator("strength", "frequency", "duration", "instruction", "additional_instruction", "dose", "dose_unit")
    @classmethod
    def clean_empty_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean or clean.upper() in {"NONE", "NULL", "UNKNOWN", "N/A"}:
            return None
        return clean

    def attach_evidence(self, field_name: str, source_text: str, confidence: Optional[float] = None) -> None:
        """Attaches field-level source evidence."""
        self.evidence[field_name] = FieldEvidence(
            field_name=field_name,
            source_text=source_text,
            value=source_text,
            confidence=confidence,
        )


class PipelineMetadata(BaseModel):
    """Metadata detailing the execution context of the extraction."""
    pipeline_version: str = Field(default="3.0.0", description="Core pipeline engine version")
    pipeline_mode: str = Field(default="standard", description="Execution mode: fast | standard")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Total execution time in milliseconds")
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict, description="Latency breakdown per stage")
    model_name: Optional[str] = Field(default=None, description="LLM/STT model name if utilized")
    device_used: str = Field(default="cpu", description="Hardware device (cpu | cuda)")
    fallback_invoked: bool = Field(default=False, description="True if LLM or primary engine fell back")


class CanonicalPrescription(BaseModel):
    """
    The unified canonical prescription object. Single source of truth for the entire system.
    """
    prescription_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique UUID")
    session_id: Optional[int] = Field(default=None, description="Associated database session/process ID")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    
    raw_input_text: str = Field(default="", description="Verbatim raw input transcript or dictation")
    raw_text: Optional[str] = None
    punctuated_text: Optional[str] = Field(default=None, description="Punctuation-normalized transcript")
    normalized_text: Optional[str] = None
    
    items: List[PrescriptionItem] = Field(default_factory=list, description="Extracted medication items")
    
    status: ExtractionStatus = Field(default=ExtractionStatus.EXTRACTED, description="Overall prescription lifecycle status")
    overall_status: Optional[ExtractionStatus] = None
    validation_warnings: List[str] = Field(default_factory=list, description="Prescription-level warnings")
    
    metadata: PipelineMetadata = Field(default_factory=PipelineMetadata, description="Execution and telemetry metadata")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total execution time in milliseconds")

    @model_validator(mode="before")
    @classmethod
    def sync_canonical_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "raw_text" in data and not data.get("raw_input_text"):
                data["raw_input_text"] = data["raw_text"]
            elif "raw_input_text" in data and not data.get("raw_text"):
                data["raw_text"] = data["raw_input_text"]
            if "normalized_text" in data and not data.get("punctuated_text"):
                data["punctuated_text"] = data["normalized_text"]
            elif "punctuated_text" in data and not data.get("normalized_text"):
                data["normalized_text"] = data["punctuated_text"]
            if "overall_status" in data and not data.get("status"):
                data["status"] = data["overall_status"]
            elif "status" in data and not data.get("overall_status"):
                data["overall_status"] = data["status"]
        return data

    @model_validator(mode="after")
    def sync_canonical_after(self) -> CanonicalPrescription:
        if not self.raw_input_text and self.raw_text:
            self.raw_input_text = self.raw_text
        if not self.raw_text and self.raw_input_text:
            self.raw_text = self.raw_input_text
        if not self.punctuated_text and self.normalized_text:
            self.punctuated_text = self.normalized_text
        if not self.normalized_text and self.punctuated_text:
            self.normalized_text = self.punctuated_text
        if not self.overall_status and self.status:
            self.overall_status = self.status
        if not self.status and self.overall_status:
            self.status = self.overall_status
        return self

    @property
    def total_medicines(self) -> int:
        return len(self.items)

    @property
    def is_valid(self) -> bool:
        return self.status in {ExtractionStatus.VALIDATED, ExtractionStatus.NORMALIZED, ExtractionStatus.EXTRACTED}


# ============================================================================
# 4. TYPED INTERMEDIATE PIPELINE MODELS
# ============================================================================
# 4. TYPED INTERMEDIATE PIPELINE MODELS
# ============================================================================

class DeterministicExtractionCandidate(BaseModel):
    """Intermediate model produced by the deterministic regex & formulary matcher."""
    raw_clause: str = ""
    medicine_name: Optional[str] = None
    medicine_candidates: List[str] = Field(default_factory=list)
    matched_drug_id: Optional[str] = None
    formulation_strength: Optional[str] = None
    order_strength: Optional[str] = None
    strength: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Any = ClinicalRoute.UNKNOWN
    instruction: Optional[str] = None
    additional_instruction: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    matched_by_soundex: bool = False
    evidence_tokens: Dict[str, str] = Field(default_factory=dict)
    evidence_spans: Dict[str, TextSpan] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    @property
    def effective_medicine_name(self) -> str:
        return self.medicine_name or (self.medicine_candidates[0] if self.medicine_candidates else "")


class LLMExtractionCandidate(BaseModel):
    """Intermediate model produced by the LangGraph multi-agent LLM extractor."""
    medicine_id: int = 1
    medicine_name: Optional[str] = None
    drug_name: str = ""
    strength: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instruction: Optional[str] = None
    additional_instruction: Optional[str] = None
    raw_agent_responses: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    evidence_spans: Dict[str, TextSpan] = Field(default_factory=dict)
    reasoning: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.medicine_name or self.drug_name


class ReconciledField(BaseModel):
    """Reconciled field tracking consensus, source, and conflict."""
    field_name: str
    selected_value: Optional[str] = None
    source_stage: ExtractionStage
    deterministic_value: Optional[str] = None
    llm_value: Optional[str] = None
    has_conflict: bool = False
    evidence: Optional[FieldEvidence] = None


class ReconciliationItem(BaseModel):
    """Intermediate item resulting from reconciling deterministic and LLM outputs."""
    item_id: int = 1
    medicine: str = ""
    medicine_name: Optional[str] = None
    strength: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Any = "oral"
    instruction: Optional[str] = None
    additional_instruction: Optional[str] = None
    status: ExtractionStatus = ExtractionStatus.NORMALIZED
    confidence: float = 0.90
    conflicts: List[str] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)
    fields: Dict[str, ReconciledField] = Field(default_factory=dict)
    overall_status: ExtractionStatus = ExtractionStatus.NORMALIZED
    has_conflict: bool = False
    conflict_details: List[str] = Field(default_factory=list)
    deterministic_candidate: Optional[DeterministicExtractionCandidate] = None
    llm_candidate: Optional[LLMExtractionCandidate] = None

    @property
    def effective_name(self) -> str:
        return self.medicine_name or self.medicine


class ValidationFinding(BaseModel):
    """Individual clinical QA or anti-hallucination validation finding."""
    item_id: Optional[int] = None
    field_name: Optional[str] = None
    severity: ValidationSeverity
    rule_id: str
    message: str
    rejected_value: Optional[Any] = None


class ValidationResult(BaseModel):
    """Intermediate model emitted by the validation boundary."""
    status: ExtractionStatus
    is_valid: bool
    findings: List[ValidationFinding] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    sanitized_items: List[PrescriptionItem] = Field(default_factory=list)
    requires_agent_feedback: bool = False
    targeted_feedback: Dict[str, str] = Field(default_factory=dict)
