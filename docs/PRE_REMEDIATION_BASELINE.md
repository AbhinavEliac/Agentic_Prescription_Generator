# 📊 Pre-Remediation Baseline Report

**Execution Timestamp:** September 8, 2026  
**Repository State:** Pre-Remediation (Post-Audit Baseline)  
**Objective:** Record empirical test suite results, endpoint health, and runtime baseline behavior before implementing architectural consolidation.

---

## 1. Test Suite Baseline Execution

```bash
python -m pytest tests/ -v
```

### Empirical Results:
- **Total Tests Collected**: 206
- **Passed**: 206 (100%)
- **Failed**: 0
- **Skipped**: 0
- **Warnings**: 1 (`StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead`)
- **Import Errors**: 0
- **Execution Duration**: 20.91 seconds

---

## 2. Pre-Remediation Endpoint & Runtime Verification

Executed via `scratch/test_pre_remediation_baseline.py`:

| Endpoint / Subsystem | Tested Operation | Status | Runtime Observation / Forensic Note |
| :--- | :--- | :---: | :--- |
| **FastAPI Core Gateway** | `GET /api/health` | 🟢 **OPERATIONAL** | Returns HTTP 200, `status: healthy`, CUDA device name detected. |
| **Drug Repository Lookup** | `GET /api/drugs/search?q=amox&limit=5` | 🟢 **OPERATIONAL** | Returns HTTP 200, 5 drug matches from SQLite formulary. |
| **Prescription Extraction** | `POST /api/prescription/extract` | 🟢 **OPERATIONAL** | Returns HTTP 200, correctly extracts Augmentin 625mg 1-0-1 for 5 days. |
| **STT Transcription Endpoint** | `POST /api/prescription/transcribe` | 🟡 **DEGRADED FALLBACK** | Returns HTTP 200, **BUT** threw `AttributeError: 'TranscriptionResult' object has no attribute 'duration_s'` inside `api_server.py`, which caused a silent fallback to legacy `transcriber.transcribe_audio`. |
| **WebSocket Streaming ASR** | `WS /ws/transcribe` | 🟡 **ARCHITECTURAL BYPASS** | Handshake succeeds, **BUT** runs through `rx_extractor_app/fast_streaming_transcriber.py` rather than canonical `app/stt/streaming.py`. |
| **Node.js Gateway / Frontend** | `GET http://localhost:5000` | 🟢 **OPERATIONAL** | Node.js Express server boots and proxies API endpoints to port 8080. |

---

## 3. Critical Defects to Remediate

1. **Hidden STT Fallback & Attribute Mismatch**:
   - `TranscriptionResult` in `app/stt/schemas.py` lacks `duration_s` and `latency_ms` properties, causing `api_server.py` to fail and secretly call `rx_extractor_app/transcriber.py`.
2. **WebSocket Streaming Bypass**:
   - `/ws/transcribe` invokes `FastLiveTranscriber` from `fast_streaming_transcriber.py` instead of the canonical `app/stt/streaming.py` (`StreamingTranscriber`).
3. **Competing Streamlit App**:
   - `rx_extractor_app/app.py` is still connected to legacy `pipeline.py`, `transcriber.py`, and `vectorstore.py`.
4. **Agentic NameError**:
   - `rx_extractor_app/agents/medicine_strength_agent.py` line 62 crashes on LLM calls due to missing `import prompt`.
5. **Duplicate Drug Data Access**:
   - `medicine_strength_agent.py` independently parses `Drug_databse/drugList.json` with ad-hoc regexes rather than using `DrugRepository`.
