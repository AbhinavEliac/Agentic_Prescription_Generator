# 🩺 LangGraph Agentic Prescription Extractor (Offline & Multi-Agent)

A state-of-the-art, 100% **offline agentic medical prescription extraction system** built with **LangGraph**, **LangChain**, local **Whisper** speech-to-text, **FAISS Vector Store**, **SQLite**, and local **LLMs** (Llama 3 8B, Qwen 1.5B, Qwen3 0.6B, DeepSeek-R1 Distill).

The application ingests raw doctor-patient voice recordings or text notes and uses a coordinated network of specialized parallel agents to extract structured clinical prescription blocks into a validated 7-column schema with **zero hallucinations**, **conversational noise filtering**, and **no unsolicited commentary**.

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
                                    │ (Noise Filtering, STT  │
                                    │  Cleaning & Fan-Out)   │
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
                                    │ (Unifies Records and   │
                                    │  Filters Non-Drug Data)│
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

## 🛠️ Installation & Setup

```powershell
# Create and activate virtual environment
python -m venv env
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch Application
streamlit run app.py
```

---

## 🧪 Automated Verification Suite

```powershell
python test_langgraph_pipeline.py
```

```text
========================================================
ALL 14 LANGGRAPH DRIFT-PROOF MULTI-AGENT TESTS PASSED!
========================================================
```
