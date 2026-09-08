"""
app/api/schemas.py
-------------------
Canonical API Contracts for the Agentic Prescription Generator Gateway.

Contains:
- Request schemas (Extraction, Transcription, Drug Search, Validation)
- Response schemas (Extraction, Transcription, Drug Catalog, Did-You-Mean)
- Error schemas (Standardized error envelopes, validation error details)
- Validation schemas (Input validation, QA audit response)
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import datetime
from pydantic import BaseModel, Field, field_validator
from app.prescription.schema import (
    CanonicalPrescription,
    PrescriptionItem,
    ExtractionStatus,
    ValidationFinding,
    ValidationSeverity,
)


# ============================================================================
# 1. ENUMS & CONFIGURATIONS
# ============================================================================

class ExtractionMode(str, Enum):
    """Execution strategy for prescription extraction."""
    AUTO = "auto"                    # Deterministic first; falls back to LLM if ambiguous
    DETERMINISTIC_ONLY = "deterministic_only"  # Sub-15ms fast path without LLM
    LLM_AGENT = "llm_agent"          # Full multi-agent LangGraph pipeline


class STTModelKey(str, Enum):
    """Supported Speech-to-Text model identifiers."""
    WHISPER_AYUSH = "whisper_ayush"
    CANARY_1B = "canary_1b"
    PARAKEET_TDT = "parakeet_tdt"
    MOONSHINE_BASE = "moonshine_base"
    MOONSHINE_TINY = "moonshine_tiny"
    WHISPER_LARGE_TURBO = "whisper_large_turbo"
    WHISPER_BASE = "whisper_base"
    WHISPER_TINY = "whisper_tiny"


# ============================================================================
# 2. REQUEST SCHEMAS
# ============================================================================

class PrescriptionExtractionRequest(BaseModel):
    """Request payload for extracting structured clinical data from text."""
    text: str = Field(..., min_length=1, description="Prescription text or doctor voice transcription")
    session_id: Optional[Union[int, str]] = Field(default=None, description="Optional SQLite session/process ID")
    process_id: Optional[Union[int, str]] = Field(default=None, description="Backwards-compatible process ID")
    patient_id: Optional[str] = Field(default=None, description="Clinical patient identifier")
    visit_id: Optional[str] = Field(default=None, description="Consultation/encounter ID")
    mode: ExtractionMode = Field(default=ExtractionMode.AUTO, description="Extraction mode (auto, deterministic_only, llm_agent)")
    fast_mode: Optional[bool] = Field(default=None, description="Backwards-compatible fast mode flag")
    device: str = Field(default="cpu", description="Compute device preference (cpu | cuda)")
    llm_model: Optional[str] = Field(default=None, description="Explicit LLM model name if llm_agent mode requested")
    process_name: Optional[str] = Field(default="node_run", description="Name of process session")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom contextual metadata")

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Prescription text cannot be empty or whitespace.")
        return clean

    def model_post_init(self, __context: Any) -> None:
        if self.process_id and not self.session_id:
            self.session_id = self.process_id
        if self.fast_mode and self.mode == ExtractionMode.AUTO:
            self.mode = ExtractionMode.DETERMINISTIC_ONLY


class PrescriptionValidationRequest(BaseModel):
    """Request payload to re-validate an existing or edited prescription."""
    raw_text: str = Field(..., min_length=1, description="Verbatim raw input against which to validate")
    items: List[PrescriptionItem] = Field(..., description="Prescription items to audit")


class DrugSearchRequest(BaseModel):
    """Query parameters for master drug formulary search."""
    q: str = Field(..., min_length=1, description="Search query string")
    limit: int = Field(default=30, ge=1, le=100, description="Maximum results to return")


# ============================================================================
# 3. RESPONSE SCHEMAS
# ============================================================================

class PrescriptionExtractionResponse(BaseModel):
    """Canonical response envelope for prescription extraction."""
    success: bool = Field(default=True, description="Indicates whether extraction succeeded")
    prescription: CanonicalPrescription = Field(..., description="Canonical extracted prescription object")
    execution_time_ms: float = Field(..., ge=0.0, description="Total API processing time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or ambiguities")


class AudioTranscriptionResponse(BaseModel):
    """Response payload for audio speech-to-text transcription."""
    success: bool = Field(default=True)
    transcript: str = Field(..., description="Recognized speech text")
    punctuated_transcript: Optional[str] = Field(default=None, description="Punctuation-normalized text")
    stt_model_used: str = Field(..., description="Model key utilized for transcription")
    audio_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    transcription_time_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    device_used: str = Field(default="cpu")


class DrugEntryResponse(BaseModel):
    """Single drug formulation record from the formulary."""
    drug_id: str
    drug_code: str
    drug_name: str
    drug_type: str = Field(..., description="'b' for brand, 'g' for generic")
    base_name: Optional[str] = None
    routes: List[str] = Field(default_factory=list)


class DrugSearchResponse(BaseModel):
    """Response envelope for drug formulary queries."""
    query: str
    total: int = Field(..., ge=0)
    results: List[DrugEntryResponse] = Field(default_factory=list)


class DidYouMeanResponse(BaseModel):
    """Phonetic fuzzy drug name recommendation."""
    query: str
    suggestion: Optional[str] = Field(default=None, description="Suggested drug name if fuzzy match found")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DrugRoutesResponse(BaseModel):
    """Relational anatomical routes permissible for a drug."""
    drug_id: str
    routes: List[str] = Field(default_factory=list)


class SystemStatusResponse(BaseModel):
    """System health, GPU status, and runtime environment metadata."""
    status: str = "online"
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    cuda_available: bool
    device_name: str
    active_process: Optional[Dict[str, Any]] = None
    default_llm: str
    default_stt: str
    stt_models: List[str]
    llm_models: List[str]


# ============================================================================
# 4. VALIDATION SCHEMAS
# ============================================================================

class PrescriptionValidationResponse(BaseModel):
    """QA audit report validating prescription safety and groundedness."""
    is_valid: bool = Field(..., description="True if no blocking clinical errors were identified")
    status: ExtractionStatus = Field(..., description="Overall validation status")
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Ratio of grounded elements")
    findings: List[ValidationFinding] = Field(default_factory=list, description="Clinical findings and discrepancies")
    sanitized_items: List[PrescriptionItem] = Field(default_factory=list, description="Validated and scrubbed items")


# ============================================================================
# 5. ERROR SCHEMAS
# ============================================================================

class ErrorDetail(BaseModel):
    """Field-level or contextual error details."""
    field: Optional[str] = Field(default=None, description="Request field that caused the error")
    code: str = Field(..., description="Error classification code")
    message: str = Field(..., description="Human-readable explanation")
    context: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic parameters")


class ApiErrorResponse(BaseModel):
    """Standardized error envelope returned on 4xx/5xx responses."""
    success: bool = Field(default=False)
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="User-facing error summary")
    details: List[ErrorDetail] = Field(default_factory=list)
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    request_id: Optional[str] = Field(default=None)

    @classmethod
    def create(cls, error_code: str, message: str, details: Optional[List[ErrorDetail]] = None, request_id: Optional[str] = None) -> ApiErrorResponse:
        return cls(
            error_code=error_code,
            message=message,
            details=details or [],
            request_id=request_id,
        )
