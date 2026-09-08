# Rehabilitation Status: Agentic Prescription Generator

**Phase**: Phase 0 — Safe Baseline & Repository Inspection  
**Date**: September 8, 2026  
**Status**: Completed (Baseline Verified)

---

## 1. Phase 0 Status

Phase 0 baseline verification is complete. The repository has been thoroughly inspected without modifying source code, altering project architecture, deleting files, or introducing unneeded dependencies.

The core execution pathways across Python (FastAPI, LangGraph multi-agent pipeline, Whisper Ayush STT via CTranslate2 CUDA) and Node.js (Express, In-Memory SQL Flowsheet) are functional, but 5 regressions in `test_langgraph_pipeline.py` and 2 runtime edge cases were documented.

---

## 2. Checks Performed

| Category | Check Description | Method / Command |
| :--- | :--- | :--- |
| **System & Runtimes** | Python version detection | `python.exe --version` |
| **System & Runtimes** | Node.js & npm version detection | `node.exe --version`, `npm.cmd --version` |
| **Dependencies** | Python installed package listing | `pip list` |
| **Dependencies** | Node.js npm package tree | `npm list` in `rx_node_app` |
| **Import Verification** | Core Python ML & framework modules | `import faster_whisper, ctranslate2, fastapi, streamlit, langgraph, langchain, torch, soundfile, gpt4all` |
| **Import Verification** | Application Python modules | `import config, db, vectorstore, graph_state, agents, graph_pipeline, exporter, transcriber, api_server` |
| **Import Verification** | Node.js modules & drug service | `require('./rx_node_app/drugDbService')` |
| **Drug Database Access** | Python dataset parsing | Count indexed drugs from `Drug_databse/drugList.json` (11,525 names) |
| **Drug Database Access** | Node dataset parsing & soundex indexing | Load `Drug_databse` via `drugDbService.js` (14,357 drugs, 38,622 route mappings) |
| **ASR / STT** | Whisper Ayush CTranslate2 execution | Transcribe `proc_70_20260828_134911.wav` on CUDA GPU |
| **Prescription Extraction** | Python LangGraph pipeline execution | `run_graph_extraction()` on transcribed text |
| **Prescription Extraction** | Node.js SQL relational flowsheet | `drugDbService.executeSqlFlowsheet()` on transcribed text |
| **Backend Service** | FastAPI endpoints via TestClient | Test `/api/status`, `/api/models`, `/api/extract` |
| **Frontend Service** | Node.js Express server startup | `node server.js` binding on port 5000 |
| **Frontend Service** | Streamlit CLI availability | `streamlit.exe --help` |
| **Test Suite** | LangGraph extraction regression tests | `rx_extractor_app/test_langgraph_pipeline.py` (22 tests) |
| **Test Suite** | Streaming audio & WebSocket test suite | `rx_extractor_app/test_streaming_engine.py` (4 stages) |
| **Benchmarks** | CTranslate2 CPU benchmark | `test_ct2_benchmark.py` |
| **Benchmarks** | CTranslate2 GPU CUDA benchmark | `test_gpu_benchmark.py` |

---

## 3. Checks Passed

1. **Python Virtual Environment**:
   - Python `3.13.13` active with all core requirements installed in `rx_extractor_app/env`.
2. **Node.js Environment**:
   - Node `v24.19.0` and npm `11.17.0` active; `express`, `cors`, and `multer` installed and operational.
3. **Module Imports**:
   - All Python framework modules (`langgraph`, `langchain`, `fastapi`, `ctranslate2`, `faster_whisper`, `torch`, `soundfile`, `streamlit`) and local modules import cleanly without runtime syntax or import errors.
4. **Drug Database Accessibility**:
   - Node.js `drugDbService` successfully loads and indexes 14,357 drug formulations and 38,622 route mappings in ~90ms.
   - Python `api_server` successfully indexes 11,525 drug tokens from `Drug_databse/drugList.json`.
5. **Speech-to-Text (STT) Execution**:
   - Whisper Ayush CTranslate2 model loads and runs on CUDA (`NVIDIA GeForce RTX 3050 Laptop GPU`) with FP16 compute.
   - Accurately transcribed `proc_70_20260828_134911.wav` in 453ms (warm pass).
6. **Prescription Extraction Execution**:
   - Python LangGraph pipeline extracts 7-column structured records from transcribed input in 1.06s.
   - Node.js SQL Flowsheet extracts structured records with available routes and formulations in 11ms.
7. **FastAPI REST Backend**:
   - Responds on `/api/status` (200 OK) and executes extraction via `/api/extract` (200 OK) in 141ms.
8. **Node.js Express Server**:
   - Boots cleanly on `http://localhost:5000` and serves static files.
9. **Streaming Engine Test Suite**:
   - `test_streaming_engine.py` passed 100% (VAD frame latency ~0.03ms, sliding window consensus, LiveStreamingTranscriber lifecycle, and full-duplex binary WebSocket communication).
10. **Benchmark Scripts**:
    - `test_ct2_benchmark.py` (CPU) and `test_gpu_benchmark.py` (GPU) passed 100%.

---

## 4. Checks Failed

1. **`rx_extractor_app/test_langgraph_pipeline.py` (5 of 22 tests failed)**:
   - `test_dual_dose_extraction`: Failed due to titration dose `100 mg` bleeding into `strength` column instead of formulation secondary dose `20 mg`.
   - `test_multi_medicine_with_timestamps_and_conditionals`: Failed because `Crocin 650 mg` was parsed as `Crocin` (strength split discrepancy).
   - `test_complex_multidrug_with_nasal_irrigations_and_precautions`: Failed on string assertion matching for `Oxymetazoline 0.05%`.
   - `test_sentence_punctuation_correction_agent`: Failed on string assertion matching for `disprin 500 mg`.
   - `test_5_drug_sequential_conditional_advice_attribution`: Failed on string assertion matching for `disprin 300 mg`.
2. **`transcriber.transcribe_audio` String Path Handling**:
   - Calling `transcribe_audio(path_str, ...)` directly with a string path throws `TypeError: string argument without an encoding` because it expects raw bytes or a stream buffer.
3. **Missing Model Weights in `Whisper_Ayush/`**:
   - Hugging Face format directory `Whisper_Ayush/.../merged_turbo_rx_v1` contains only JSON tokenizers and configs without model binaries (e.g. `model.safetensors`).

---

## 5. Blockers for Future Rehabilitation Phases

| Severity | Issue | Impact | Resolution Requirement for Phase 1 |
| :--- | :--- | :--- | :--- |
| **High** | 5 failing tests in `test_langgraph_pipeline.py` | Regressions prevent automated verification of extraction pipeline changes. | Align rule-based dosage partitioner in `medicine_strength_agent.py` and normalize assertions before refactoring. |
| **Medium** | `transcriber.py` TypeError on string paths | Direct invocation with file path fails unless opened as binary. | Add path check (`if isinstance(audio_data, str) and os.path.exists(audio_data): read bytes`) to avoid runtime crashes. |
| **Low** | Folder misspelling `Drug_databse` | Code references hardcode the typo; renaming would break paths without coordinated updates. | Standardize or alias paths in a controlled step. |
| **Low** | Duplicate data folder in `rx_node_app/public/data` | Wasted disk space (3.1 MB) and dual source of truth. | Consolidate data loading to single canonical path. |
| **Low** | Hardcoded Windows paths in `.bat` scripts | Batch scripts fail if run by any user other than `ADMIN`. | Use dynamic environment variables (`%LOCALAPPDATA%`, relative `.\env\Scripts`). |
