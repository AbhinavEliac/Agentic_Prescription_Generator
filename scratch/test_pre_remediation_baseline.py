"""
scratch/test_pre_remediation_baseline.py
----------------------------------------
Verification of baseline runtime endpoints before any remediation changes:
1. /api/prescription/extract
2. /api/prescription/transcribe (STT endpoint)
3. /ws/transcribe (WebSocket streaming)
4. /api/drugs/search (Drug lookup)
5. /api/health
"""

import io
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

rx_app_path = repo_root / "rx_extractor_app"
if str(rx_app_path) not in sys.path:
    sys.path.append(str(rx_app_path))

from rx_extractor_app.api_server import app
from fastapi.testclient import TestClient

client = TestClient(app)

print("[1/5] Testing /api/health...")
res = client.get("/api/health")
assert res.status_code == 200, f"Health failed: {res.text}"
print(f"  Health OK: status={res.json().get('status')}")

print("[2/5] Testing /api/drugs/search...")
res = client.get("/api/drugs/search?q=amox&limit=5")
assert res.status_code == 200, f"Drug search failed: {res.text}"
print(f"  Drug search OK: {len(res.json().get('results', []))} results")

print("[3/5] Testing /api/prescription/extract...")
res = client.post("/api/prescription/extract", json={"text": "Tab Augmentin 625mg 1-0-1 for 5 days", "fast_mode": True})
assert res.status_code == 200, f"Extract failed: {res.text}"
print(f"  Extract OK: {len(res.json().get('prescription', {}).get('items', []))} items extracted")

print("[4/5] Testing /api/prescription/transcribe...")
mock_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
files = {"file": ("test.wav", io.BytesIO(mock_wav), "audio/wav")}
res = client.post("/api/prescription/transcribe", files=files, data={"stt_model": "mock"})
assert res.status_code == 200, f"Transcribe failed: {res.text}"
print(f"  Transcribe OK: transcript='{res.json().get('transcript')}'")

print("[5/5] Testing /ws/transcribe WebSocket handshake...")
with client.websocket_connect("/ws/transcribe?engine=mock") as ws:
    ws.send_text("PING")
    ws.close()
print("  WebSocket handshake OK")

print("\nALL PRE-REMEDIATION BASELINE ENDPOINTS VERIFIED OPERATIONAL!")
