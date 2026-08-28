# 🩺 LangGraph Agentic Prescription Extractor (Offline & Multi-Agent)

A state-of-the-art, 100% **offline agentic medical prescription extraction system** built with **LangGraph**, **LangChain**, local **Whisper** speech-to-text, **FAISS Vector Store**, **SQLite**, and local **LLMs** (Llama 3 8B, Qwen 1.5B, Qwen3 0.6B, DeepSeek-R1 Distill).

The application ingests raw doctor-patient voice recordings or text notes and uses a coordinated network of specialized parallel agents to extract structured clinical prescription blocks into a validated 7-column schema with **zero hallucinations** and **no unsolicited commentary**.

---

## 🏗️ Multi-Agent Architecture (LangGraph StateGraph)

```
                            ┌─────────────────────────────────────────┐
                            │           User Voice / Text Input       │
                            │      (Whisper Speech-to-Text / UI)      │
                            └────────────────────┬────────────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │    Supervisor Agent    │
                                    │ (Initializes State &   │
                                    │  Coordinates Fan-Out)  │
                                    └────────────┬───────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   │                             │                             │
                   ▼                             ▼                             ▼
       ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
       │ Medicine & Strength  │      │     Route Agent      │      │ Duration & Frequency │
       │        Agent         │      │ (Direct Mention &    │      │        Agent         │
       │ (Generics, Dual-Dose,│      │  Form Specificity)   │      │ (Verbatim Intervals  │
       │  Companion Splitting)│      │                      │      │  & Duration Spans)   │
       └───────────┬──────────┘      └───────────┬──────────┘      └───────────┬──────────┘
                   │                             │                             │
                   │                             ▼                             │
                   │                 ┌──────────────────────┐                  │
                   │                 │  Instruction Agent   │                  │
                   │                 │ (Primary Admin vs.   │                  │
                   │                 │  Secondary Advice)   │                  │
                   │                 └───────────┬──────────┘                  │
                   │                             │                             │
                   └─────────────────────────────┼─────────────────────────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │    Aggregator Agent    │
                                    │  (Merges Parallel Node │
                                    │   Outputs into Records)│
                                    └────────────┬───────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │    Validator Agent     │
                                    │ (Groundedness & Anti-  │
                                    │   Hallucination QA)    │
                                    └────────────┬───────────┘
                                                 │
                       ┌─────────────────────────┴─────────────────────────┐
                       │                                                   │
        [NEEDS_CORRECTION & Iterations < 3]                          [VALID / Max Reps Capped]
                       │                                                   │
                       ▼                                                   ▼
       ┌───────────────────────────────┐                  ┌─────────────────────────────────┐
       │ Targeted Feedback Dispatched  │                  │         Formatter Agent         │
       │ to Responsible Agents & Retry │                  │ (Strict Structured Text Blocks) │
       └───────────────────────────────┘                  └────────────────┬────────────────┘
                                                                           │
                                                                           ▼
                                                          ┌─────────────────────────────────┐
                                                          │   SQLite DB + CSV/XLSX Export   │
                                                          │   + Streamlit Data Table UI     │
                                                          └─────────────────────────────────┘
```

---

## 📊 Structured 7-Column Clinical Schema

| Column | Description | Real-World Example |
| :--- | :--- | :--- |
| **`Drug_name`** | Brand name or generic name + primary dose | `PHEXIN DT 250 mg`, `Paracetamol 650 mg`, `Ofloxacin-Ornidazole` |
| **`strength`** | Secondary dose value if dual-dosed (or `NONE`) | `20 mg`, `NONE` |
| **`frequency`** | Schedule notation, clinical timing, or interval | `twice daily(1-0-1)`, `every 6 hours`, `TID`, `once daily` |
| **`duration`** | Duration span | `10 days`, `5 days`, `2 weeks`, `30 days` |
| **`route`** | Anatomical route of administration | `oral`, `inhalation`, `topical`, `nasal`, `ophthalmic`, `otic` |
| **`instruction`** | Primary administration timing, meal rules, devices, PRN | `1. before breakfast`, `1. using Revolizer device 2. rinse mouth after use` |
| **`additional_instruction`** | Secondary clinical monitoring, follow-up, warnings, diet/lifestyle | `1. Seek reassessment if adverse effects develop`, `1. Stick to a bland diet` |

---

## 🚀 Key Features

1. **Doctor-First Precision**:
   - Strictly grounded in the doctor's spoken/written words. No unsolicited advice, no moralizing, no disclaimers.
2. **Parallel Agent Extraction**:
   - Independent extraction nodes concurrently process drug names, strengths, anatomical routes, dosage schedules, and primary vs. secondary instructions.
3. **Dual Instruction Separation**:
   - `instruction`: Primary preparation, meal timing, ingestion methods, device care.
   - `additional_instruction`: Clinical follow-up visits, reassessment conditions, adverse effect cautions, dose titrations, and dietary/lifestyle guidance.
4. **Drift-Proof Natural Language Clause Decomposition**:
   - Automatically handles non-standard times of day (`Morning Night`), cross-sentence instructions, and standalone clinical sentences without brittle hardcoding.
5. **Quality Assurance Validator Loop**:
   - Anti-hallucination auditor verifies 100% groundedness against raw input text and loops targeted feedback up to a strict cap of 3 iterations.
6. **Multi-Format Export & Thread History**:
   - Complete per-process SQLite persistence with live downloads for **CSV** and **Excel (.xlsx)** spreadsheets.
7. **100% Offline & Private**:
   - Zero external API dependencies or cloud leakage. Local Whisper speech transcription + local LLM backends.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13)
- **Git**

### 2. Clone Repository & Setup Environment
```powershell
# Clone the repository
git clone <repo-url>
cd rx_extractor_app_agentic

# Create and activate virtual environment
python -m venv env
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Application
```powershell
streamlit run rx_extractor_app/app.py
```

---

## 🧪 Automated Verification Suite

Run the comprehensive test suite covering companion drugs, inhalation routes, topicals, ophthalmic drops, otic drops, interval schedules, speaker headers, and cross-sentence linkages:

```powershell
python rx_extractor_app/test_langgraph_pipeline.py
```

```text
========================================================
ALL 11 LANGGRAPH DRIFT-PROOF MULTI-AGENT TESTS PASSED!
========================================================
```

---

## 📂 Repository Structure

```text
rx_extractor_app_agentic/
├── .gitignore                      # Git ignore file (excluding training datasets, DBs, audio, logs)
├── requirements.txt                # Production dependencies
├── README.md                       # Architecture & usage documentation
└── rx_extractor_app/
    ├── app.py                      # Streamlit interactive UI
    ├── config.py                   # Paths, model definitions, device configs
    ├── db.py                       # SQLite persistence manager
    ├── exporter.py                 # 7-column CSV & XLSX exporters
    ├── graph_pipeline.py           # LangGraph StateGraph engine & runner
    ├── graph_state.py              # Pydantic/TypedDict state schemas
    ├── prompt.py                   # Multi-agent specialized clinical prompts
    ├── transcriber.py              # Offline Whisper voice transcription
    ├── vectorstore.py              # FAISS prompt retrieval store
    ├── test_langgraph_pipeline.py  # 11-test automated verification suite
    └── agents/                     # Modular Multi-Agent Node Implementations
        ├── __init__.py             # Agent node exports
        ├── supervisor_agent.py     # Supervisor coordinating agent
        ├── medicine_strength_agent.py # Medicine & strength extractor
        ├── route_agent.py          # Anatomical route extractor
        ├── duration_frequency_agent.py # Schedule & duration extractor
        ├── instruction_agent.py    # Primary & additional instruction agent
        ├── aggregator_agent.py     # Parallel state aggregator
        ├── validator_agent.py      # Quality assurance validator
        ├── formatter_agent.py      # Structured block formatter
        └── utils.py                # Clause segmenter & clinical regex helpers
```
