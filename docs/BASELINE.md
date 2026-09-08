# System Baseline Report: Agentic Prescription Generator

**Generated**: September 8, 2026  
**Environment**: Windows 11 (OS build), Python 3.13.13, Node.js v24.19.0, npm 11.17.0, CUDA (NVIDIA GeForce RTX 3050 Laptop GPU)  
**Status**: Phase 0 Baseline Established (Pre-refactoring Diagnostic Phase)

---

## 1. Current Project Structure

```text
Agentic_Prescription_Generator/
├── .gitignore                                      # Root git ignore rules
├── convert_to_ct2.py                              # HuggingFace Whisper -> CTranslate2 INT8 converter
├── requirements.txt                               # Root Python requirements (52 lines)
├── start_all.bat                                  # Windows batch script to launch both servers
├── start_all.sh                                   # Linux/Ubuntu bash script to launch both servers
├── start_backend.bat                              # Windows batch script for FastAPI backend (:8080)
├── start_backend.sh                               # Linux bash script for FastAPI backend (:8080)
├── start_frontend.bat                             # Windows batch script for Node.js studio (:5000)
├── start_frontend.sh                              # Linux bash script for Node.js studio (:5000)
├── test_ct2_benchmark.py                          # CPU 6-thread CTranslate2 ASR benchmark
├── test_gpu_benchmark.py                          # CUDA GPU CTranslate2 ASR benchmark
├── README.md                                      # Project overview and run guide
├── MOBILE_DEPLOYMENT_GUIDE.md                     # Guide for mobile mic streaming & Nginx reverse proxy
├── VAD_AND_STREAMING_ARCHITECTURE.md              # Technical architecture for VAD & streaming STT
│
├── Drug_databse/                                  # Master clinical dataset (spelled "Drug_databse")
│   ├── dose_units.csv                             # 31 clinical dosage units (mg, ml, g, etc.)
│   ├── Drug_Route_mapping.csv                     # 38,622 drug-to-route relational mappings
│   ├── drug_routes.csv                            # 49 anatomical route definitions
│   ├── drug_schedule.csv                          # 39 dosage frequency / timing notations
│   └── drugList.json                              # 14,357 pharmaceutical drug formulations (2.52 MB)
│
├── Whisper_Ayush/                                 # Fine-tuned Whisper checkpoint folder
│   └── Whisper large/
│       └── saved_models_whisper_large/
│           └── merged_turbo_rx_v1/                # Tokenizer/config files ONLY (NO model weights file!)
│               ├── added_tokens.json
│               ├── config.json
│               ├── generation_config.json
│               ├── merges.txt
│               ├── normalizer.json
│               ├── preprocessor_config.json
│               ├── special_tokens_map.json
│               ├── tokenizer_config.json
│               └── vocab.json
│
├── Whisper_Ayush_ct2/                             # Quantized CTranslate2 model (functional ASR weights)
│   ├── config.json
│   ├── model.bin                                  # Quantized INT8/float16 model weights (814 MB)
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocabulary.json
│
├── rx_extractor_app/                              # Python Backend & Streamlit Clinical Application
│   ├── .gitignore
│   ├── README.md
│   ├── requirements.txt                           # Duplicate copy of root requirements.txt
│   ├── config.py                                  # Central model labels, paths, device settings
│   ├── db.py                                      # SQLite database persistence layer (processes & history)
│   ├── vectorstore.py                             # FAISS vector store & prompt caching logic
│   ├── prompt.py                                  # Modular multi-agent prompts (Supervisor, Formatter, etc.)
│   ├── pipeline.py                                # Offline GPT4All LLM instantiation & query runner
│   ├── graph_state.py                             # LangGraph TypedDict state schemas
│   ├── graph_pipeline.py                          # LangGraph StateGraph orchestration & feedback loop
│   ├── exporter.py                                # Output parser (7 columns) and CSV/XLSX export
│   ├── transcriber.py                             # Multi-model STT coordinator
│   ├── fast_streaming_transcriber.py              # Low-latency streaming transcriber with GPU affinity
│   ├── streaming_transcriber.py                   # Live sliding-window streaming coordinator
│   ├── api_server.py                              # FastAPI REST & WebSocket server (port 8080)
│   ├── app.py                                     # Streamlit analytical clinical dashboard (port 8501)
│   ├── naturalized_prescriptions_final.jsonl      # Synthetic benchmark prescriptions dataset (57.1 MB)
│   ├── test_langgraph_pipeline.py                 # Automated test suite (22 tests)
│   ├── test_streaming_engine.py                   # Streaming audio, VAD, and WebSocket tests
│   │
│   ├── env/                                       # Active Python virtual environment (Python 3.13.13)
│   │   └── pyvenv.cfg
│   │
│   ├── agents.py                                  # Top-level re-exporter for agents package
│   ├── agents/                                    # Modular LangGraph agent nodes
│   │   ├── __init__.py
│   │   ├── supervisor_agent.py                    # Supervisor coordination & noise filtering
│   │   ├── punctuation_agent.py                   # Speech transcript segmentation & punctuation
│   │   ├── medicine_strength_agent.py             # Drug name & strength extractor
│   │   ├── route_agent.py                         # Anatomical route extractor & specificity resolver
│   │   ├── duration_frequency_agent.py            # Frequency & duration extractor with coreference
│   │   ├── instruction_agent.py                   # Primary and additional clinical instruction extractor
│   │   ├── aggregator_agent.py                    # Multi-agent record merger
│   │   ├── validator_agent.py                     # Anti-hallucination QA audit & feedback loop
│   │   ├── formatter_agent.py                     # Structured 7-column formatter
│   │   └── utils.py                               # Shared NLP regex, noise filter, JSON parser
│   │
│   ├── streaming_engine/                          # Real-time DSP audio processing
│   │   ├── __init__.py
│   │   ├── vad_detector.py                        # Frame-level Voice Activity Detector (energy/zero-crossing)
│   │   └── sliding_window_decoder.py              # Overlap-add audio buffer & deduplication
│   │
│   └── data/                                      # Local runtime storage (created dynamically)
│       ├── app_state.db                           # SQLite persistence database (290 KB)
│       ├── audio_files/                           # Sample & recorded audio WAVs
│       ├── faiss_index/                           # FAISS index files
│       └── outputs/                               # Generated CSV and Excel exports
│
└── rx_node_app/                                   # Parallel Node.js Studio Web Application
    ├── .gitignore
    ├── README.md
    ├── package.json                               # Express, CORS, Multer
    ├── package-lock.json
    ├── server.js                                  # Express HTTP & proxy server (port 5000)
    ├── drugDbService.js                           # In-memory Soundex & SQL relational flowsheet (<15ms)
    ├── node_modules/                              # Installed npm dependencies
    │
    └── public/                                    # Frontend SPA static client
        ├── index.html                             # Glassmorphic clinical UI
        ├── app.js                                 # Client state, audio recorder, inline table editor (79 KB)
        ├── style.css                              # Responsive CSS styles (12 KB)
        └── data/                                  # Exact duplicate copy of Drug_databse/ (3.1 MB)
            ├── dose_units.csv
            ├── Drug_Route_mapping.csv
            ├── drug_routes.csv
            ├── drug_schedule.csv
            └── drugList.json
```

---

## 2. Current Entry Points

The project contains three distinct entry points:

| Service / Interface | Primary Entry File | Default Port | Role |
| :--- | :--- | :--- | :--- |
| **FastAPI REST & WebSocket Gateway** | `rx_extractor_app/api_server.py` | `8080` | High-speed REST backend serving LangGraph multi-agent extraction, Whisper Ayush GPU STT, and WebSocket streaming. |
| **Node.js RxAgent Studio** | `rx_node_app/server.js` | `5000` | Express application serving glassmorphic web UI (`public/index.html`), in-memory sub-second SQL flowsheet matcher, and proxying to FastAPI. |
| **Streamlit Clinical Dashboard** | `rx_extractor_app/app.py` | `8501` | Multi-tab Python analytical UI with process thread manager, vector store inspector, and SQLite database explorer. |

---

## 3. Current Startup Commands

### Windows (PowerShell / Command Prompt)

1. **Full Stack (Both Services concurrently)**:
   ```cmd
   start_all.bat
   ```
2. **FastAPI Backend (Port 8080)**:
   ```powershell
   .\rx_extractor_app\env\Scripts\python.exe rx_extractor_app\api_server.py
   # Or using uvicorn:
   .\rx_extractor_app\env\Scripts\python.exe -m uvicorn api_server:app --app-dir rx_extractor_app --host 127.0.0.1 --port 8080
   ```
3. **Node.js Frontend Studio (Port 5000)**:
   ```powershell
   cd rx_node_app
   & "C:\Program Files\nodejs\node.exe" server.js
   ```
4. **Streamlit Clinical Dashboard (Port 8501)**:
   ```powershell
   .\rx_extractor_app\env\Scripts\streamlit.exe run rx_extractor_app\app.py
   ```

### Linux / Ubuntu (Bash)

1. **Full Stack**:
   ```bash
   chmod +x start_all.sh start_backend.sh start_frontend.sh
   ./start_all.sh
   ```
2. **Backend**:
   ```bash
   ./start_backend.sh
   ```
3. **Frontend**:
   ```bash
   ./start_frontend.sh
   ```

---

## 4. Current Dependencies

### Python Environment (`rx_extractor_app/env`)
- **Python Version**: `3.13.13` (64-bit Windows)
- **Virtualenv Location**: `rx_extractor_app/env`
- **Key Installed Packages & Versions**:
  - `langchain`: 1.3.17
  - `langchain-core`: 1.6.0
  - `langchain-community`: 0.4.2
  - `langgraph`: 1.2.11
  - `fastapi`: 0.110.0 (via starlette 1.6.0)
  - `uvicorn`: 0.52.4
  - `websockets`: 16.1.1
  - `pydantic`: 2.13.4
  - `torch`: 2.13.0
  - `torchaudio`: 2.11.0+cu126
  - `torchvision`: 0.28.0+cu126
  - `ctranslate2`: 4.6.0
  - `faster-whisper`: 1.2.1
  - `openai-whisper`: 20250625
  - `gpt4all`: 2.8.2
  - `soundfile`: 0.14.0
  - `streamlit`: 1.62.0
  - `pandas`: 3.0.5
  - `openpyxl`: 3.1.5

### Node.js Environment (`rx_node_app`)
- **Node Version**: `v24.19.0`
- **npm Version**: `11.17.0`
- **Dependencies (`rx_node_app/package.json`)**:
  - `express`: `^4.21.2` (installed: 4.22.2)
  - `cors`: `^2.8.5` (installed: 2.8.6)
  - `multer`: `^1.4.5-lts.1` (installed: 1.4.5-lts.2)

---

## 5. Current Test Status

| Test Suite / Script | Command | Total Tests | Passed | Failed | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LangGraph Pipeline Tests** | `python rx_extractor_app/test_langgraph_pipeline.py` | 22 | 17 | 5 | ❌ **FAILING** |
| **Streaming Audio & WebSocket Tests** | `python rx_extractor_app/test_streaming_engine.py` | 4 test stages | 4 | 0 | ✅ **PASSED** |
| **CT2 Benchmark (CPU)** | `python test_ct2_benchmark.py` | 3 passes | 3 | 0 | ✅ **PASSED** |
| **GPU Benchmark (CUDA)** | `python test_gpu_benchmark.py` | 3 passes | 3 | 0 | ✅ **PASSED** |

### Detailed LangGraph Test Breakdown:
- **Passed (17)**:
  - `test_multi_drug_diet_instructions`
  - `test_route_specificity`
  - `test_no_moralizing_or_extra_cautions`
  - `test_validator_loop_and_cap`
  - `test_complex_5_drug_prescription`
  - `test_topicals_sprays_drops`
  - `test_cross_sentence_and_class_instructions`
  - `test_additional_instructions_column`
  - `test_natural_language_time_and_evaluation`
  - `test_punctuation_free_continuous_voice_speech`
  - `test_complex_multidrug_decimal_and_advice_guards`
  - `test_noisy_transcript_and_conversational_chatter_filtering`
  - `test_single_med_hydration_and_visit_doctor_advice`
  - `test_cross_sentence_coreference_frequency_resolution`
  - `test_dosage_titration_instruction_capture`
  - `test_faulty_grammar_duration_and_comma_titration`
  - `test_chronological_instruction_order_and_conditional_punctuation`
- **Failed (5)**:
  - `test_dual_dose_extraction`: `AssertionError: Unexpected strength: 100 mg` (Clause-level regex matched trailing titration integer "100 mg" into strength instead of formulation strength "20 mg").
  - `test_multi_medicine_with_timestamps_and_conditionals`: `AssertionError: Crocin 650 mg missing!` (Extracted as `Crocin` because Crocin dosage is split or not present in `drugList.json`).
  - `test_complex_multidrug_with_nasal_irrigations_and_precautions`: `AssertionError: assert 'Oxymetazoline 0.05%' in parsed[-1]['Drug_name']`.
  - `test_sentence_punctuation_correction_agent`: `AssertionError: assert 'disprin 500 mg' in parsed[0]['Drug_name'].lower()`.
  - `test_5_drug_sequential_conditional_advice_attribution`: `AssertionError: assert 'disprin 300 mg' in parsed[1]['Drug_name'].lower()`.

---

## 6. Current Runtime Errors

1. **`transcriber.transcribe_audio` TypeError on String Path**:
   - Calling `transcribe_audio(audio_path, ...)` with a file path string crashes at line 285 (`tmp_file.write(bytes(audio_data))`) with `TypeError: string argument without an encoding`.
   - The function assumes `audio_data` is `bytes` or has `.read()`. If given raw file bytes, it succeeds.
2. **Streamlit ScriptRunContext Warning**:
   - Top-level `import streamlit as st` inside `rx_extractor_app/pipeline.py` causes:
     `WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext!`
     when any CLI, API server, or unit test imports `pipeline`.
3. **Windows Python 3.13 Torchvision DLL Conflict**:
   - `api_server.py` lines 9-11 require `sys.modules.setdefault('torchvision', None)` and `sys.modules.setdefault('torchaudio', None)` to prevent a C++ DLL loader crash in PyTorch under Python 3.13 on Windows.

---

## 7. Current Missing Files / Configuration

1. **Missing Model Weights in `Whisper_Ayush`**:
   - Directory `Whisper_Ayush/Whisper large/saved_models_whisper_large/merged_turbo_rx_v1/` contains tokenizer and model JSON configurations, but no weight files (`model.safetensors` or `pytorch_model.bin` are absent).
   - *Mitigation*: The project relies instead on `Whisper_Ayush_ct2/model.bin` (814 MB), which is present and fully functional.
2. **Missing Environment (`.env`) Files**:
   - No `.env` or `.env.example` file exists in the project root or subdirectories. Environment variables such as `PYTHON_API_BASE` and `PORT` are handled with fallback defaults.
3. **Missing Formal Testing Framework**:
   - No `pytest.ini` or test runner configuration. Tests are invoked as imperative Python scripts.

---

## 8. Current Known Issues

1. **Directory Naming Typo**:
   - `Drug_databse/` is misspelled (`databse` vs `database`). Multiple files in Python (`api_server.py`, `medicine_strength_agent.py`) and Node.js (`drugDbService.js`) reference this misspelled path via relative paths.
2. **Data Duplication**:
   - `rx_node_app/public/data/` contains an exact duplicate of all files in `Drug_databse/` (consuming ~3.1 MB redundantly).
3. **Hardcoded User Paths in Batch Scripts**:
   - `start_backend.bat` contains hardcoded paths referencing `C:\Users\ADMIN\...` instead of dynamic environment resolution `%LOCALAPPDATA%` or relative virtualenv activation.
4. **Port Inconsistency in Error String**:
   - In `rx_node_app/server.js` line 52, the error message states:
     `error: 'Python Agentic API service is unreachable. Ensure the FastAPI server is running on port 8000.'`
     However, the backend actually runs on port `8080`.
5. **Incomplete Exports in `agents.py`**:
   - `rx_extractor_app/agents.py` omits `punctuation_agent` and `correct_sentence_punctuation` from its imports and `__all__`, whereas `rx_extractor_app/agents/__init__.py` properly includes them.
6. **Rule-Based Extraction Discrepancy with `drugList.json`**:
   - The fallback rule-based engine partitions medicine dosage based on whether an integer appears in `drugList.json` for that brand name. If a brand (e.g. `Disprin`, `Crocin`) is missing a formulation row in `drugList.json`, the dosage is treated as an order strength rather than binding to `Drug_name`, causing regression test assertions to fail.
