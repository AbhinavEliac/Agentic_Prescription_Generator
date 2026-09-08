"""
tests/unit/test_schemas.py
--------------------------
Unit test suite verifying canonical domain models and API contracts.

Tests:
1. Valid objects & field structures
2. Missing values handling (strictly None / null, never fabricated 'NONE')
3. Invalid values & Pydantic constraint validation
4. Conflicting states & lifecycle status transitions
5. Provenance & Evidence binding
6. Intermediate pipeline models (deterministic, LLM, reconciliation, validation)
7. Serialization & deserialization round-trips
8. API compatibility (requests, responses, errors, validation)
"""

import pytest
import datetime
import json
from pydantic import ValidationError

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
from app.api.schemas import (
    ExtractionMode,
    STTModelKey,
    PrescriptionExtractionRequest,
    PrescriptionValidationRequest,
    DrugSearchRequest,
    PrescriptionExtractionResponse,
    AudioTranscriptionResponse,
    DrugEntryResponse,
    DrugSearchResponse,
    DidYouMeanResponse,
    DrugRoutesResponse,
    SystemStatusResponse,
    PrescriptionValidationResponse,
    ErrorDetail,
    ApiErrorResponse,
)


# ============================================================================
# 1. PRESCRIPTION ITEM & EVIDENCE TESTS
# ============================================================================

def test_valid_prescription_item_full():
    """Validates complete prescription item with all fields populated."""
    item = PrescriptionItem(
        item_id=1,
        medicine="Paracetamol 650 mg",
        raw_medicine_name="paracetamol",
        matched_drug_id="16176",
        strength="650 mg",
        dose="650",
        dose_unit="mg",
        frequency="twice daily (1-0-1)",
        duration="5 days",
        route=ClinicalRoute.ORAL,
        instruction="after meals",
        additional_instruction="if fever persists, consult doctor",
        status=ExtractionStatus.VALIDATED,
        confidence=0.98,
        available_routes=["ORAL", "RT"],
    )

    item.attach_evidence(
        field_name="medicine",
        source_text="Take one tablet of Paracetamol 650 mg",
        confidence=0.99,
    )

    assert item.item_id == 1
    assert item.medicine == "Paracetamol 650 mg"
    assert item.route == ClinicalRoute.ORAL
    assert item.status == ExtractionStatus.VALIDATED
    assert "medicine" in item.evidence
    assert item.evidence["medicine"].source_text == "Take one tablet of Paracetamol 650 mg"
    assert item.confidence == 0.98


def test_missing_values_remain_none():
    """
    CRITICAL PRINCIPLE TEST:
    Missing values MUST remain None/null. Never fabricate clinical values.
    """
    item = PrescriptionItem(
        item_id=1,
        medicine="Amoxicillin 500 mg",
    )

    # Optional fields must strictly default to None, route defaults to UNKNOWN
    assert item.strength is None
    assert item.dose is None
    assert item.dose_unit is None
    assert item.frequency is None
    assert item.duration is None
    assert item.instruction is None
    assert item.additional_instruction is None
    assert item.matched_drug_id is None
    assert item.route == ClinicalRoute.UNKNOWN

    # String placeholders like "NONE", "null", "N/A" must be cleaned to None
    dirty_item = PrescriptionItem(
        item_id=2,
        medicine="Atenolol 50 mg",
        strength="NONE",
        frequency="none",
        duration="NULL",
        instruction="N/A",
    )
    assert dirty_item.strength is None
    assert dirty_item.frequency is None
    assert dirty_item.duration is None
    assert dirty_item.instruction is None


def test_invalid_medicine_placeholder_rejected():
    """Verifies that an empty or placeholder medicine name raises ValidationError."""
    with pytest.raises(ValidationError):
        PrescriptionItem(item_id=1, medicine="")

    with pytest.raises(ValidationError):
        PrescriptionItem(item_id=1, medicine="NONE")

    with pytest.raises(ValidationError):
        PrescriptionItem(item_id=1, medicine="UNKNOWN")


def test_invalid_confidence_range():
    """Confidence score must be strictly bounded in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        PrescriptionItem(item_id=1, medicine="Aspirin", confidence=1.5)

    with pytest.raises(ValidationError):
        PrescriptionItem(item_id=1, medicine="Aspirin", confidence=-0.1)


def test_text_span_validation():
    """Validates TextSpan offsets and prevents inverted spans."""
    span = TextSpan(start=5, end=15, verbatim="Paracetamol")
    assert span.start == 5
    assert span.end == 15
    assert span.verbatim == "Paracetamol"

    with pytest.raises(ValidationError):
        TextSpan(start=20, end=10, verbatim="Invalid")


# ============================================================================
# 2. CANONICAL PRESCRIPTION DOCUMENT TESTS
# ============================================================================

def test_canonical_prescription_document():
    """Tests the top-level CanonicalPrescription container model."""
    raw_text = "Take Dolo 650 once daily for 5 days."
    item = PrescriptionItem(
        item_id=1,
        medicine="Dolo 650",
        frequency="once daily",
        duration="5 days",
        route=ClinicalRoute.ORAL,
    )

    rx = CanonicalPrescription(
        raw_input_text=raw_text,
        punctuated_text="Take Dolo 650 once daily for 5 days.",
        items=[item],
        status=ExtractionStatus.NORMALIZED,
        metadata=PipelineMetadata(
            pipeline_version="3.0.0",
            stage_latencies_ms={"deterministic": 4.5, "validator": 1.2},
            device_used="cuda",
        ),
        total_latency_ms=5.7,
    )

    assert rx.prescription_id is not None
    assert rx.total_medicines == 1
    assert rx.is_valid is True
    assert rx.items[0].medicine == "Dolo 650"
    assert rx.metadata.device_used == "cuda"


def test_canonical_prescription_serialization_round_trip():
    """Tests full JSON serialization and deserialization without information loss."""
    item = PrescriptionItem(
        item_id=1,
        medicine="Azithromycin 500 mg",
        frequency="once daily",
        duration="3 days",
        route=ClinicalRoute.ORAL,
        instruction="before food",
        status=ExtractionStatus.VALIDATED,
        evidence={
            "medicine": FieldEvidence(
                field_name="medicine",
                source_text="Azithromycin 500 mg",
                span=TextSpan(start=0, end=20, verbatim="Azithromycin 500 mg"),
                confidence=0.99,
            )
        },
    )

    rx = CanonicalPrescription(
        raw_input_text="Azithromycin 500 mg once daily before food for 3 days",
        items=[item],
        status=ExtractionStatus.VALIDATED,
    )

    json_str = rx.model_dump_json()
    reconstructed = CanonicalPrescription.model_validate_json(json_str)

    assert reconstructed.prescription_id == rx.prescription_id
    assert reconstructed.items[0].medicine == "Azithromycin 500 mg"
    assert reconstructed.items[0].evidence["medicine"].span.verbatim == "Azithromycin 500 mg"
    assert reconstructed.items[0].route == ClinicalRoute.ORAL


# ============================================================================
# 3. CONFLICTING STATES & LIFECYCLE TESTS
# ============================================================================

def test_conflicting_states():
    """Verifies representation of conflicting extraction states."""
    reconciled_field = ReconciledField(
        field_name="frequency",
        selected_value="twice daily (1-0-1)",
        source_stage=ExtractionStage.DETERMINISTIC,
        deterministic_value="twice daily (1-0-1)",
        llm_value="once daily (1-0-0)",
        has_conflict=True,
    )

    rec_item = ReconciliationItem(
        item_id=1,
        medicine="Metformin 500 mg",
        fields={"frequency": reconciled_field},
        overall_status=ExtractionStatus.CONFLICT,
        has_conflict=True,
        conflict_details=["Deterministic frequency 'twice daily' contradicts LLM 'once daily'"],
    )

    assert rec_item.overall_status == ExtractionStatus.CONFLICT
    assert rec_item.has_conflict is True
    assert rec_item.fields["frequency"].has_conflict is True
    assert "twice daily" in rec_item.conflict_details[0]


def test_needs_review_and_error_status():
    """Validates NEEDS_REVIEW and ERROR lifecycle states."""
    item = PrescriptionItem(
        item_id=1,
        medicine="UnknownCompound 100 mg",
        status=ExtractionStatus.NEEDS_REVIEW,
        validation_warnings=["Drug name not found in master formulary"],
    )
    assert item.status == ExtractionStatus.NEEDS_REVIEW
    assert len(item.validation_warnings) == 1

    finding = ValidationFinding(
        item_id=1,
        field_name="route",
        severity=ValidationSeverity.ERROR,
        rule_id="RULE_UNGROUNDED_ROUTE",
        message="Route 'intravenous' was not mentioned in prescription text.",
        rejected_value="intravenous",
    )

    val_res = ValidationResult(
        status=ExtractionStatus.ERROR,
        is_valid=False,
        findings=[finding],
        grounding_score=0.75,
        requires_agent_feedback=True,
        targeted_feedback={"route_agent": "Route 'intravenous' was ungrounded."},
    )

    assert val_res.is_valid is False
    assert val_res.status == ExtractionStatus.ERROR
    assert val_res.findings[0].severity == ValidationSeverity.ERROR
    assert val_res.requires_agent_feedback is True


# ============================================================================
# 4. INTERMEDIATE MODELS TESTS
# ============================================================================

def test_deterministic_extraction_candidate():
    """Validates deterministic stage candidate data contract."""
    cand = DeterministicExtractionCandidate(
        raw_clause="take Paracetamol 650 mg twice daily for 5 days",
        medicine_candidates=["PARACETAMOL"],
        matched_drug_id="16176",
        formulation_strength="650 mg",
        frequency="twice daily (1-0-1)",
        duration="5 days",
        route=ClinicalRoute.ORAL,
        confidence=0.95,
        matched_by_soundex=False,
    )
    assert cand.matched_drug_id == "16176"
    assert cand.route == ClinicalRoute.ORAL
    assert cand.confidence == 0.95


def test_llm_extraction_candidate():
    """Validates LLM multi-agent stage candidate data contract."""
    llm_cand = LLMExtractionCandidate(
        medicine_id=1,
        drug_name="Budecort 200 mcg",
        strength="200 mcg",
        route="inhalation",
        frequency="twice daily",
        duration="10 days",
        instruction="rinse mouth after use",
        confidence=0.92,
    )
    assert llm_cand.drug_name == "Budecort 200 mcg"
    assert llm_cand.route == "inhalation"


# ============================================================================
# 5. API CONTRACT SCHEMAS TESTS
# ============================================================================

def test_prescription_extraction_request_valid():
    """Tests valid extraction request parsing."""
    req = PrescriptionExtractionRequest(
        text="Take Aten 50 mg once daily before food.",
        session_id=12,
        mode=ExtractionMode.AUTO,
        device="cuda",
    )
    assert req.text == "Take Aten 50 mg once daily before food."
    assert req.session_id == 12
    assert req.mode == ExtractionMode.AUTO
    assert req.device == "cuda"


def test_prescription_extraction_request_empty_rejected():
    """Rejects empty prescription extraction text."""
    with pytest.raises(ValidationError):
        PrescriptionExtractionRequest(text="")

    with pytest.raises(ValidationError):
        PrescriptionExtractionRequest(text="   ")


def test_prescription_extraction_response():
    """Tests canonical extraction response envelope."""
    item = PrescriptionItem(item_id=1, medicine="Pan 40")
    rx = CanonicalPrescription(raw_input_text="Take Pan 40", items=[item])

    resp = PrescriptionExtractionResponse(
        success=True,
        prescription=rx,
        execution_time_ms=12.4,
        warnings=[],
    )
    assert resp.success is True
    assert resp.prescription.items[0].medicine == "Pan 40"
    assert resp.execution_time_ms == 12.4


def test_api_error_response_structure():
    """Tests standardized error envelope."""
    err = ApiErrorResponse.create(
        error_code="INVALID_AUDIO_FORMAT",
        message="Uploaded audio file must be WAV or MP3.",
        details=[
            ErrorDetail(
                field="file",
                code="UNSUPPORTED_MIME_TYPE",
                message="Expected audio/wav, got application/pdf",
            )
        ],
        request_id="req-12345",
    )

    assert err.success is False
    assert err.error_code == "INVALID_AUDIO_FORMAT"
    assert err.details[0].field == "file"
    assert err.request_id == "req-12345"
    assert err.timestamp is not None


def test_prescription_validation_schemas():
    """Tests QA validation request and response models."""
    item = PrescriptionItem(item_id=1, medicine="Crocin 650 mg", route=ClinicalRoute.ORAL)
    req = PrescriptionValidationRequest(
        raw_text="Take Crocin 650 mg orally",
        items=[item],
    )
    assert req.items[0].medicine == "Crocin 650 mg"

    resp = PrescriptionValidationResponse(
        is_valid=True,
        status=ExtractionStatus.VALIDATED,
        grounding_score=1.0,
        sanitized_items=[item],
    )
    assert resp.is_valid is True
    assert resp.status == ExtractionStatus.VALIDATED
    assert resp.grounding_score == 1.0


def test_audio_transcription_response():
    """Tests transcription response envelope."""
    resp = AudioTranscriptionResponse(
        success=True,
        transcript="Take Dolo 650 twice daily",
        punctuated_transcript="Take Dolo 650 twice daily.",
        stt_model_used=STTModelKey.WHISPER_AYUSH.value,
        audio_duration_seconds=3.5,
        transcription_time_ms=45.2,
        device_used="cuda",
    )
    assert resp.success is True
    assert resp.stt_model_used == "whisper_ayush"
    assert resp.audio_duration_seconds == 3.5
