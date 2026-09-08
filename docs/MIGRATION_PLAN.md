# Migration Plan: Agentic Prescription Generator Rehabilitation

**Document**: `docs/MIGRATION_PLAN.md`  
**Version**: 1.0.0  
**Date**: September 8, 2026  
**Status**: Approved Blueprint — Pending Implementation Execution  
**Target Architecture**: [`docs/TARGET_ARCHITECTURE.md`](file:///c:/Users/ADMIN/Downloads/rx_extractor_app_agentic/docs/TARGET_ARCHITECTURE.md)  

---

## Migration Philosophy & Safety Rules

1. **Non-Destructive Transition**: New core modules will be developed and verified in isolation before retiring legacy components.
2. **Atomic Steps**: Each migration phase is self-contained with explicit verification gates.
3. **No Downtime / Broken Commits**: At every step, the repository must remain importable and executable.
4. **Verification First**: Automated tests must validate every newly migrated component before replacing the corresponding legacy code.

---

## Phase 1: Foundation, Configuration & Canonical Data Consolidation

### Step 1.1: Consolidate Configuration & Environment
- **Objective**: Establish single centralized configuration via `pydantic-settings` and `.env`.
- **Actions**:
  1. Create `.env.example` defining `HOST`, `PORT`, `NODE_PORT`, `STREAMLIT_PORT`, `STT_DEVICE`, `DEFAULT_LLM_MODEL`.
  2. Implement `core/config.py` with `AppConfig` inheriting from `pydantic_settings.BaseSettings`.
  3. Ensure dynamic directory resolution based on `Path(__file__)` rather than hardcoded `C:\Users\ADMIN\...`.
- **Verification Gate**:
  - `python -c "from core.config import settings; print(settings.BASE_DIR, settings.PORT)"` executes cleanly without error.

### Step 1.2: Canonical Dataset Normalization & De-duplication
- **Objective**: Consolidate master drug datasets into single canonical path and eliminate redundant copies.
- **Actions**:
  1. Rename directory `Drug_databse/` -> `Drug_database/` (fixing directory name typo).
  2. Create temporary path alias or backwards-compatible symlink/reference if required during migration.
  3. Append common clinical Indian brand formulations (`DOLO 650 TAB`, `DOLO 500 TAB`, `PAN 40 TAB`, `PAN-D CAP`, `TELMA 40 TAB`) directly into `Drug_database/drugList.json`.
  4. Delete redundant duplicate dataset folder `rx_node_app/public/data/` (saving 3.15 MB).
- **Verification Gate**:
  - Verify `Drug_database/drugList.json` parses as valid JSON containing 14,362 entries.
  - Verify `Drug_database/Drug_Route_mapping.csv` parses with 38,622 route mappings.

---

## Phase 2: Canonical Schemas & Drug Repository

### Step 2.1: Define Canonical Pydantic Schemas
- **Objective**: Implement type-safe domain models for clinical items and extractions.
- **Actions**:
  1. Create `core/schemas/prescription.py` with `PrescriptionItem`, `CanonicalPrescription`, `EvidenceSpan`, `RouteType`, and `ValidationStatus`.
  2. Create `core/schemas/drug.py` with `DrugEntry`, `RouteMappingEntry`, and `ScheduleEntry`.
  3. Create `core/schemas/telemetry.py` with latency metrics and agent audit log structures.
- **Verification Gate**:
  - Run `pytest` or Python one-liner validating schema instantiation and serialization to/from JSON.

### Step 2.2: Implement Canonical `DrugRepository`
- **Objective**: Build high-speed Python in-memory formulary and relational route repository.
- **Actions**:
  1. Create `core/drug_db/repository.py` implementing `DrugRepository` singleton:
     - Parses `Drug_database/drugList.json` into `drugs_by_id` and `drugs_by_clean_name`.
     - Builds inverted 4-character Soundex index (`soundex_buckets`) and 3-character prefix tree (`prefix_index`).
     - Parses `Drug_database/Drug_Route_mapping.csv` into `route_mappings`.
     - Implements `search(query, limit)`, `did_you_mean(term)`, `get_routes_for_drug(drug_id)`, and `get_strengths_for_base(base_name)`.
- **Verification Gate**:
  - Benchmark `DrugRepository` initialization time (must be `< 150ms`).
  - Verify `repo.search("paracetamol")` and `repo.did_you_mean("parasitamol")` return exact matches in `< 1ms`.

---

## Phase 3: Speech-to-Text (STT) Subsystem Modernization

### Step 3.1: Define `STTEngine` Abstract Base Class
- **Objective**: Create clean abstraction decoupling transcription clients from underlying inference engines.
- **Actions**:
  1. Create `core/stt/base.py` defining `STTEngine` interface (`load()`, `transcribe()`, `transcribe_chunk()`, `is_cuda_available()`).
  2. Implement input polymorphism: ensure `transcribe()` accepts `bytes`, `str` (filepath), file-like streams, or `np.ndarray` without raising `TypeError`.

### Step 3.2: Implement `WhisperAyushCT2Engine` Adapter
- **Objective**: Encapsulate high-speed CTranslate2 Whisper model with dynamic hardware detection.
- **Actions**:
  1. Create `core/stt/adapters/ct2_whisper.py` implementing `STTEngine`.
  2. Binds directly to `Whisper_Ayush_ct2/model.bin`.
  3. Remove hardcoded paths to external Ollama directories; implement platform-native CUDA DLL resolution via `torch.utils` / `ctranslate2`.
  4. Integrate VAD and sliding window decoders from `core/stt/vad.py` and `core/stt/stream_decoder.py`.
- **Verification Gate**:
  - Run benchmark test transcribing `proc_70_20260828_134911.wav` on CUDA FP16 (verifying latency `< 500ms`).
  - Verify string path, raw bytes, and file-like objects all transcribe cleanly without `TypeError`.

---

## Phase 4: Unified 7-Stage Prescription Extraction Pipeline

### Step 4.1: Implement Normalizer & Segmenter (Stages 1 & 2)
- **Objective**: Unify text cleaning, phonetic correction, and clause segmentation into clean modules.
- **Actions**:
  1. Create `core/prescription/normalizer.py`:
     - Consolidates phonetic replacements (`Aten`, `Metformin`, `Soframycin`).
     - Normalizes punctuation, decimal quantities (`0.5%`, `100 mg`), and schedule notation (`1-0-1`).
     - Decouples conversational filler and speaker timestamp prefixes.
  2. Create `core/prescription/segmenter.py`:
     - Segments continuous transcript into individual medicine prescription clauses based on conjunctions and imperative transitions.
- **Verification Gate**:
  - Unit tests verify segmenting multi-drug continuous speech with zero clause leakage.

### Step 4.2: Implement Deterministic Extractor (Stage 3)
- **Objective**: Provide sub-15ms extraction for unambiguous, standard prescriptions without requiring an LLM.
- **Actions**:
  1. Create `core/prescription/deterministic.py`:
     - Uses `DrugRepository` for base drug identification and formulation dosage extraction.
     - Resolves the dosage partitioner bug: correctly separates secondary formulation dose from trailing titration instructions (fixing Test 2 and Test 20 failures).
     - Resolves anatomical routes via `DrugRepository.get_routes_for_drug()`.
     - Extracts standardized schedules and primary administration instructions.
     - Computes confidence score ($C$).
- **Verification Gate**:
  - Verify execution time is `< 15ms` for a 5-medicine prescription.

### Step 4.3: Refactor LangGraph Agent Pipeline (Stage 4)
- **Objective**: Clean up LangGraph multi-agent orchestration for complex syntax and ambiguous cases.
- **Actions**:
  1. Migrate agents into `core/prescription/agentic/`.
  2. Replace bare `except Exception: pass` blocks with explicit exception logging.
  3. Wire agents into `core/prescription/agentic/graph.py`.
  4. Ensure prompts and state operate directly on canonical schemas.

### Step 4.4: Implement Reconciler, Validator & Formatter (Stages 5, 6, 7)
- **Objective**: Establish strict validation boundary and emit type-safe `CanonicalPrescription`.
- **Actions**:
  1. Create `core/prescription/reconciler.py`: Merges candidates and maps `EvidenceSpan` tokens.
  2. Create `core/prescription/validator.py`: Enforces 100% grounding, rejects non-drug entities and hallucinated advice, cross-checks routes against `DrugRepository`.
  3. Create `core/prescription/formatter.py`: Generates final `CanonicalPrescription` and serializes to SQLite/CSV/Excel.
- **Verification Gate**:
  - Run all 22 clinical benchmark scenarios; verify all 22 pass (100% success rate).

---

## Phase 5: Canonical FastAPI Gateway Modernization

### Step 5.1: Refactor FastAPI REST Routes
- **Objective**: Establish FastAPI as single authoritative API gateway on port `8080`.
- **Actions**:
  1. Implement `api/main.py` with lifespan pre-warming of STT and `DrugRepository`.
  2. Create `api/routes/extract.py`: Exposes `POST /api/v1/extract` routing to the unified 7-stage engine.
  3. Create `api/routes/transcribe.py`: Exposes `POST /api/v1/transcribe` routing to `STTEngine`.
  4. Create `api/routes/drugs.py`: Exposes `GET /api/v1/drugs/search`, `/api/v1/drugs/did-you-mean`, `/api/v1/drugs/{id}/routes`.
  5. Create `api/routes/history.py`: Exposes process/session history backed by SQLite.
  6. Delete legacy inline procedural extraction block (`FastRelationalFlowsheet` in `api_server.py:L212-L456`).

### Step 5.2: Modernize WebSocket Streaming Handler
- **Objective**: Clean up full-duplex binary audio streaming.
- **Actions**:
  1. Create `api/websocket/stream_handler.py` managing `/ws/v1/transcribe`.
  2. Directly pipes binary PCM16 stream into `VADDetector` and `SlidingWindowStreamingDecoder`.
- **Verification Gate**:
  - Run integration tests validating REST endpoints and WebSocket handshakes via `TestClient`.

---

## Phase 6: Presentation Tier Realignment (Node.js & Streamlit)

### Step 6.1: Refactor Node.js Express Server
- **Objective**: Convert Node.js application from an independent extraction engine into a clean static host & proxy.
- **Actions**:
  1. In `rx_node_app/server.js`:
     - Fix error message string port typo (replace `8000` with `8080`).
     - Remove local execution of `drugDbService.executeSqlFlowsheet()`.
     - Proxy `/api/*` and `/ws/*` transparently to FastAPI (`http://127.0.0.1:8080`).
  2. In `rx_node_app/public/app.js`:
     - Remove client-side in-browser extraction engine (`analyzePrescriptionText()`).
     - Remove startup fetch of `drugList.json` and `Drug_Route_mapping.csv`.
     - Route extraction requests directly to `POST /api/v1/extract` on FastAPI.
     - Retain waveform visualizer, audio recorder, and editable table grid.
- **Verification Gate**:
  - Test browser UI at `http://localhost:5000`: speech recording streams to WebSocket, extraction renders within 20ms, table allows inline edits and CSV export.

### Step 6.2: Realign Streamlit Dashboard
- **Objective**: Re-position Streamlit as Developer & Benchmarking Portal on port `8501`.
- **Actions**:
  1. Update `rx_extractor_app/app.py` to import directly from `core.prescription` and `core.stt`.
  2. Remove redundant FAISS prompt retrieval; load system prompts from static templates.
  3. Add model comparison tabs (Whisper Ayush vs Canary vs Moonshine benchmark test harness).
- **Verification Gate**:
  - Launch `streamlit run rx_extractor_app/app.py` and verify multi-model comparative benchmarking.

---

## Phase 7: Automated Test Suite Migration & Regression Fixing

### Step 7.1: Establish Pytest Test Framework
- **Objective**: Migrate from imperative scripts to standard Pytest hierarchy.
- **Actions**:
  1. Create `tests/conftest.py` with shared fixtures.
  2. Port and fix all 22 tests in `test_langgraph_pipeline.py` into `tests/integration/test_prescription_pipeline.py`.
  3. Fix the 5 regressions identified in baseline audit:
     - Test 2 (`test_dual_dose_extraction`): Fix titration dose capture.
     - Test 7 (`test_multi_medicine_with_timestamps_and_conditionals`): Fix Crocin 650 mg drug name formulation binding.
     - Test 19 (`test_complex_multidrug_with_nasal_irrigations_and_precautions`): Fix Oxymetazoline 0.05% route & formulation match.
     - Test 20 (`test_sentence_punctuation_correction_agent`): Fix Disprin 500 mg name match.
     - Test 22 (`test_5_drug_sequential_conditional_advice_attribution`): Fix Disprin 300 mg name match.
  4. Port streaming audio tests into `tests/integration/test_stt_engine.py`.
- **Verification Gate**:
  - Execute `pytest tests/`: 100% of unit and integration tests must pass.

---

## Phase 8: Dead Code Deletion & Final Operational Verification

### Step 8.1: Eliminate Dead Code & Legacy Artifacts
- **Objective**: Clean repository of unneeded files and legacy duplicates.
- **Actions**:
  1. Delete empty checkpoint directory `Whisper_Ayush/Whisper large/saved_models_whisper_large/merged_turbo_rx_v1/`.
  2. Delete duplicate `rx_extractor_app/requirements.txt`.
  3. Deprecate `rx_extractor_app/vectorstore.py`.
  4. Update `start_backend.bat`, `start_frontend.bat`, `start_all.bat` to eliminate hardcoded `C:\Users\ADMIN` paths.
  5. Update `README.md` to reflect unified architecture.

### Step 8.2: Full Stack End-to-End Verification
- **Objective**: Validate complete production workflow across both interfaces.
- **Actions**:
  1. Run `start_all.bat` (or `start_all.sh` on Linux).
  2. Record 10-second voice prescription through Node.js Studio (`http://localhost:5000`).
  3. Verify live waveform display, streaming ASR transcription, <20ms extraction, and CSV export.
  4. Verify SQLite database persistence in `data/app_state.db`.
- **Verification Gate**:
  - Zero terminal errors, zero runtime warnings, clean shutdown on Ctrl+C.
