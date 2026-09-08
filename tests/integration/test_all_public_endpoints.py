"""
tests/integration/test_all_public_endpoints.py
---------------------------------------------
Exhaustive verification of all public FastAPI REST and WebSocket endpoints:
- GET  /api/health
- GET  /api/status
- GET  /api/models
- POST /api/prescription/extract
- POST /api/prescription/validate
- POST /api/prescription/transcribe
  - Success with valid audio (Mock STT)
  - Rejection of invalid file extension (415)
  - Rejection of oversized payload > 25MB (413)
  - Rejection of empty file (400)
- GET  /api/drugs/search
- GET  /api/drugs/did-you-mean
- GET  /api/drugs/{drug_id}/routes
- GET  /api/reference-data
- GET  /api/history
- GET  /api/threads
- POST /api/threads
- WS   /ws/transcribe
"""

import io
import pytest
from fastapi.testclient import TestClient
from rx_extractor_app.api_server import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_public_get_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime_seconds" in data
    assert "request_id" in data


def test_public_get_status(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "active_process" in data
    assert "default_llm" in data
    assert "default_stt" in data


def test_public_get_models(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "stt_models" in data
    assert "llm_models" in data
    assert len(data["stt_models"]) >= 5


def test_public_post_prescription_extract(client):
    payload = {
        "text": "Tab Augmentin 625mg 1-0-1 after meals for 5 days",
        "llm_mode": False
    }
    res = client.post("/api/prescription/extract", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "prescription" in data
    assert len(data["prescription"]["items"]) >= 1
    med = data["prescription"]["items"][0]
    assert "augmentin" in med["medicine"].lower()
    assert med["strength"] == "625mg"
    assert "5 days" in med["duration"]



def test_public_post_prescription_validate(client):
    payload = {
        "items": [
            {
                "name": "Paracetamol",
                "strength": "500mg",
                "frequency": "1-0-1",
                "duration": "3 days",
                "route": "oral"
            }
        ],
        "raw_text": "Paracetamol 500mg 1-0-1 oral for 3 days"
    }
    res = client.post("/api/prescription/validate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "is_valid" in data
    assert "findings" in data
    assert "grounding_score" in data



def test_public_transcribe_audio_valid(client):
    # Mock audio bytes (WAV header)
    mock_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("test_sample.wav", io.BytesIO(mock_wav), "audio/wav")}
    res = client.post("/api/prescription/transcribe", files=files, data={"stt_model": "mock"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "transcript" in data


def test_public_transcribe_audio_invalid_extension(client):
    bad_file = b"BINARY_DATA"
    files = {"file": ("malicious.exe", io.BytesIO(bad_file), "application/x-msdownload")}
    res = client.post("/api/prescription/transcribe", files=files)
    assert res.status_code == 415
    assert "Unsupported media type" in res.json()["message"]


def test_public_transcribe_audio_empty(client):
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    res = client.post("/api/prescription/transcribe", files=files)
    assert res.status_code == 400


def test_public_transcribe_audio_oversized(client):
    # 26MB dummy stream (exceeds 25MB limit)
    big_buffer = io.BytesIO(b"0" * (26 * 1024 * 1024))
    files = {"file": ("oversized.wav", big_buffer, "audio/wav")}
    res = client.post("/api/prescription/transcribe", files=files)
    assert res.status_code == 413
    assert "exceeds maximum limit" in res.json()["message"]


def test_public_get_drugs_search(client):
    res = client.get("/api/drugs/search?q=amox&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0


def test_public_get_drugs_did_you_mean(client):
    res = client.get("/api/drugs/did-you-mean?q=Augmentn")
    assert res.status_code == 200
    data = res.json()
    assert "query" in data
    assert "suggestion" in data


def test_public_get_drug_routes(client):
    res = client.get("/api/drugs/1/routes")
    assert res.status_code == 200
    data = res.json()
    assert "drug_id" in data
    assert "routes" in data


def test_public_get_reference_data(client):
    res = client.get("/api/reference-data")
    assert res.status_code == 200
    data = res.json()
    assert "routes" in data
    assert "schedules" in data
    assert "dose_units" in data


def test_public_get_history(client):
    res = client.get("/api/history?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "records" in data
    assert isinstance(data["records"], list)


def test_public_threads_get_and_post(client):
    # Test GET threads
    res = client.get("/api/threads")
    assert res.status_code == 200
    data = res.json()
    assert "threads" in data

    # Test POST thread
    post_res = client.post("/api/threads", json={"name": "Dr. Test Patient Consultation", "device": "cpu"})
    assert post_res.status_code == 200
    thread_data = post_res.json()
    assert thread_data["success"] is True
    assert "process_id" in thread_data



def test_public_ws_transcribe_handshake(client):
    # Test WebSocket connection and clean close
    with client.websocket_connect("/ws/transcribe?engine=mock") as websocket:
        websocket.send_text("PING")
        websocket.close()
