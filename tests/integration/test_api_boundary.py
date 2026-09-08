"""
tests/integration/test_api_boundary.py
---------------------------------------
Integration tests verifying the FastAPI canonical boundary layer:
- Endpoint routing and schema compliance for 5 core endpoints:
  1. POST /api/prescription/extract
  2. POST /api/prescription/transcribe
  3. POST /api/prescription/validate
  4. GET  /api/drugs/search
  5. GET  /api/health
- Global X-Request-ID propagation (custom header echoed, generated if absent)
- Standardized error envelopes (ApiErrorResponse) on 400, 422, 500
- Backwards-compatibility aliases (/api/extract, /api/transcribe)
"""

import pytest
from fastapi.testclient import TestClient
from rx_extractor_app.api_server import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ============================================================================
# 1. HEALTH & TELEMETRY ENDPOINTS
# ============================================================================

def test_health_endpoint(client):
    """GET /api/health should return system status, version, uptime, and request ID."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "3.0.0"
    assert "request_id" in data
    assert "uptime_seconds" in data
    assert "default_models" in data
    assert "timestamp" in data

    # Verify X-Request-ID header echoes the payload request_id
    assert response.headers.get("x-request-id") == data["request_id"]


def test_request_id_propagation(client):
    """Client-provided X-Request-ID header must be preserved in headers and body."""
    custom_id = "clinical-trace-999888"
    response = client.get("/api/health", headers={"X-Request-ID": custom_id})

    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id
    assert response.json()["request_id"] == custom_id


# ============================================================================
# 2. CANONICAL PRESCRIPTION EXTRACTION
# ============================================================================

def test_extract_canonical_endpoint(client):
    """POST /api/prescription/extract should extract structured clinical items."""
    payload = {
        "text": "Take Paracetamol 650 mg twice daily for 5 days after food.",
        "session_id": "test-session-boundary",
        "mode": "deterministic_only",
    }
    response = client.post("/api/prescription/extract", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "prescription" in data
    assert "execution_time_ms" in data

    items = data["prescription"]["items"]
    assert len(items) == 1
    assert "Paracetamol" in items[0]["medicine_name"]
    assert ("650 mg" in items[0]["medicine_name"]) or (items[0]["strength"] == "650 mg")
    assert "twice daily" in items[0]["frequency"].lower()
    assert "5 days" in items[0]["duration"].lower()


def test_extract_backwards_compatible_alias(client):
    """POST /api/extract should be preserved as alias with fast_mode support."""
    payload = {
        "text": "Amoxicillin 500 mg thrice daily for 7 days",
        "fast_mode": True,
    }
    response = client.post("/api/extract", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "parsed_records" in data
    assert "prescription" in data
    assert len(data["prescription"]["items"]) >= 1


# ============================================================================
# 3. CLINICAL VALIDATION BOUNDARY
# ============================================================================

def test_validate_endpoint_valid_case(client):
    """POST /api/prescription/validate should pass clinically grounded items."""
    payload = {
        "raw_text": "Paracetamol 650 mg twice daily for 5 days after food",
        "items": [
            {
                "item_id": 1,
                "medicine_name": "Paracetamol",
                "strength": "650 mg",
                "frequency": "twice daily",
                "duration": "5 days",
                "route": "oral",
                "instruction": "after food",
            }
        ],
    }
    response = client.post("/api/prescription/validate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["is_valid"] is True
    assert data["status"] == "VALIDATED"
    assert len(data["findings"]) == 0


def test_validate_endpoint_hallucination_detection(client):
    """POST /api/prescription/validate should flag hallucinated or ungrounded drugs."""
    payload = {
        "raw_text": "Metformin 500 mg once daily with dinner",
        "items": [
            {
                "item_id": 1,
                "medicine_name": "Azithromycin",  # Not in raw text
                "strength": "500 mg",
                "frequency": "daily",
                "route": "intravenous",
            }
        ],
    }
    response = client.post("/api/prescription/validate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["is_valid"] is False
    assert data["status"] == "ERROR"
    assert len(data["findings"]) >= 1
    rule_ids = [f["rule_id"] for f in data["findings"]]
    assert "RULE_GROUNDED_DRUG_NAME" in rule_ids


# ============================================================================
# 4. MASTER DRUG FORMULARY SEARCH
# ============================================================================

def test_drug_search_endpoint(client):
    """GET /api/drugs/search should query the canonical DrugRepository."""
    response = client.get("/api/drugs/search?q=paracetamol&limit=10")
    assert response.status_code == 200

    data = response.json()
    assert "total" in data
    assert "results" in data
    assert data["total"] > 0
    assert len(data["results"]) <= 10

    first = data["results"][0]
    assert "drug_name" in first
    assert "drug_id" in first
    assert "base_name" in first


# ============================================================================
# 5. AUDIO TRANSCRIPTION BOUNDARY
# ============================================================================

def test_transcribe_endpoint_with_mock_engine(client):
    """POST /api/prescription/transcribe should process audio bytes using STTManager."""
    # Minimal valid 16-bit PCM mono RIFF WAV header + 0 samples
    wav_header = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00"
        b"\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    files = {"file": ("recording.wav", wav_header, "audio/wav")}
    data = {"stt_model": "mock"}

    response = client.post("/api/prescription/transcribe", files=files, data=data)
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert "transcript" in body
    assert body["stt_model_used"] == "mock"
    assert "transcription_time_ms" in body


# ============================================================================
# 6. STANDARDIZED ERROR ENVELOPES
# ============================================================================

def test_error_envelope_422_validation(client):
    """Validation errors must return HTTP 422 with the canonical ApiErrorResponse envelope."""
    # Empty text violates min_length / non-empty validator
    response = client.post("/api/prescription/extract", json={"text": "   "})
    assert response.status_code == 422

    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "clinical schema validation" in data["message"].lower()
    assert isinstance(data["details"], list)
    assert len(data["details"]) >= 1
    assert "request_id" in data


def test_error_envelope_400_bad_request(client):
    """Missing required audio file must return HTTP 400 with ApiErrorResponse envelope."""
    response = client.post("/api/prescription/transcribe")
    assert response.status_code == 400

    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "HTTP_400"
    assert "no audio file" in data["message"].lower()
    assert "request_id" in data
