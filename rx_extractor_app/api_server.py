"""
api_server.py
-------------
FastAPI backend service exposing the Agentic Prescription Extractor and STT engines.
Enables parallel Node.js, Web, and mobile applications to interface with the LangGraph pipeline.
"""
import os
import sys
# Prevent torchvision / torchaudio DLL binary incompatibility from crashing transformers on Windows Python 3.13
sys.modules.setdefault('torchvision', None)
sys.modules.setdefault('torchaudio', None)

import re
import io
import time
import asyncio
import datetime
import threading
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

import config
import db
import vectorstore
import pipeline
import exporter
import transcriber
from graph_pipeline import run_graph_extraction
from exporter import parse_output_fields
from fast_streaming_transcriber import FastLiveTranscriber

# Initialize database
db.init_db()


@asynccontextmanager
async def lifespan(app_instance):
    """Pre-warm the Whisper Ayush model on server startup to eliminate cold-start latency."""
    print("[Startup] Pre-warming Whisper Ayush (CTranslate2 Turbo GPU)...")
    def _prewarm():
        try:
            import numpy as np
            from fast_streaming_transcriber import _get_ayush_model, _transcribe_ayush_pcm
            m = _get_ayush_model()
            # Run 0.1s warm-up pass on GPU
            dummy = np.zeros(1600, dtype=np.float32)
            _transcribe_ayush_pcm(dummy)
            print("[Startup] Whisper Ayush GPU pre-warmed and ready.")
        except Exception as e:
            print(f"[Startup] Whisper Ayush pre-warming notice: {e}")
    threading.Thread(target=_prewarm, daemon=True).start()
    yield
    print("[Shutdown] FastAPI server shutting down.")


app = FastAPI(
    title="Agentic Prescription Extractor API",
    description="REST API bridging LangGraph Multi-Agent Architecture and Multi-Engine STT to parallel Node.js applications.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Node.js app (Port 3000 / 5173 / any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for LLM and Vector Store
_CACHED_MODELS: Dict[str, Any] = {}
_CACHED_STORE = None
_DRUG_NAMES_SET = None


def get_drug_names_set():
    global _DRUG_NAMES_SET
    if _DRUG_NAMES_SET is None:
        _DRUG_NAMES_SET = set()
        db_path = os.path.join(os.path.dirname(__file__), "..", "Drug_databse", "drugList.json")
        if os.path.exists(db_path):
            try:
                import re
                with open(db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("drugData", []):
                        name = item.get("drug_name", "").upper()
                        name = re.sub(r"\b(TAB|TABS|TABLET|TABLETS|CAP|CAPS|CAPSULE|CAPSULES|SYP|SYRUP|INJ|INJECTION|DROPS)\b", "", name)
                        name = re.sub(r"\b\d+(\.\d+)?\s*(MG|G|MCG|ML|L|IU|%)\b", "", name)
                        clean = re.sub(r"[^\w\s]", "", name).strip()
                        for w in clean.split():
                            if len(w) >= 3:
                                _DRUG_NAMES_SET.add(w.upper())
            except Exception as e:
                print(f"[Backend] Error loading drugList: {e}")
    return _DRUG_NAMES_SET


def get_cached_chat(device: str = "cpu", model_name: str = None):
    global _CACHED_MODELS
    target_model = model_name or config.MODEL_NAME
    key = f"{device}_{target_model}"
    if key not in _CACHED_MODELS:
        _CACHED_MODELS[key] = pipeline.build_chat(device, target_model)
    return _CACHED_MODELS[key]


def get_cached_vector_store():
    global _CACHED_STORE
    if _CACHED_STORE is None:
        _CACHED_STORE = vectorstore.load_or_create_index()
    return _CACHED_STORE


class ExtractRequest(BaseModel):
    text: str
    process_id: Optional[int] = None
    llm_model: Optional[str] = None
    device: Optional[str] = "cpu"
    process_name: Optional[str] = "node_run"
    fast_mode: Optional[bool] = False


class ThreadCreateRequest(BaseModel):
    name: str
    device: Optional[str] = "cpu"
    model_label: Optional[str] = config.DEFAULT_MODEL_LABEL


@app.get("/api/status")
def get_system_status():
    """Returns system status, device configurations, and model options."""
    active_proc = db.get_active_process()
    return {
        "status": "online",
        "active_process": active_proc,
        "default_llm": config.DEFAULT_MODEL_LABEL,
        "default_stt": config.DEFAULT_STT_MODEL_LABEL,
        "available_llm_models": config.MODEL_OPTIONS,
        "available_stt_models": config.STT_MODEL_OPTIONS,
        "device_options": config.DEVICE_OPTIONS,
    }


@app.get("/api/models")
def get_models():
    """Lists all available LLM and STT models."""
    return {
        "llm_models": [
            {"label": k, "filename": v, "is_default": (k == config.DEFAULT_MODEL_LABEL)}
            for k, v in config.MODEL_OPTIONS.items()
        ],
        "stt_models": [
            {"label": k, "key": v, "is_default": (k == config.DEFAULT_STT_MODEL_LABEL)}
            for k, v in config.STT_MODEL_OPTIONS.items()
        ],
    }


def normalize_asr_phonetics(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?i)\b(?:8|eight)\s+(?:and|&)\s+(?:50|fifty)\b", "Aten 50", text)
    text = re.sub(r"(?i)\b(?:8|eight)\s+(?:and|&)\s+(?:25|twenty\s*five)\b", "Aten 25", text)
    text = re.sub(r"(?i)\b(?:8|eight)\s+(?:and|&)\s+(?:100|one\s*hundred)\b", "Aten 100", text)
    text = re.sub(r"(?i)\b(?:8|eight)\s*(?:ten|10)\s+(?:50|fifty)\b", "Aten 50", text)
    text = re.sub(r"(?i)\b(?:8|eight)\s*(?:ten|10)\s+(?:25|twenty\s*five)\b", "Aten 25", text)
    text = re.sub(r"(?i)\b(?:8|eight)\s*(?:ten|10)\b", "Aten", text)
    text = re.sub(r"(?i)\bMetaforamine\b", "Metformin", text)
    text = re.sub(r"(?i)\bMetaformine\b", "Metformin", text)
    text = re.sub(r"(?i)\bMetaphormine\b", "Metformin", text)
    text = re.sub(r"(?i)\bSophramycin\b", "Soframycin", text)
    return text


@app.post("/api/extract")
def extract_prescription(req: ExtractRequest):
    """
    Executes the LangGraph Multi-Agent extraction workflow on text prescription.
    Returns validated structured medication blocks and logs to SQLite + CSV/XLSX.
    """
    query = req.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Prescription text cannot be empty.")

    # 1. Process / Thread handling
    proc_id = req.process_id
    csv_path = None
    xlsx_path = None
    if not proc_id:
        active = db.get_active_process()
        if active:
            proc_id = active["process_id"]
            csv_path = active.get("csv_path")
            xlsx_path = active.get("xlsx_path")
        else:
            p_name = req.process_name or "node_run"
            csv_path, xlsx_path = exporter.new_output_paths(p_name)
            target_model = req.llm_model or config.DEFAULT_MODEL_LABEL
            model_file = config.MODEL_OPTIONS.get(target_model, config.MODEL_NAME)
            proc_id = db.create_process(p_name, req.device or "cpu", csv_path, xlsx_path, model_name=model_file, model_label=target_model)

    # 2. Run LangGraph Multi-Agent pipeline OR Fast Relational Flowsheet
    target_llm_label = req.llm_model or config.DEFAULT_MODEL_LABEL

    if req.fast_mode or req.llm_model == "fast_relational":
        t0 = time.perf_counter()
        query = normalize_asr_phonetics(query)
        parsed_records = []
        raw_segments = [s.strip() for s in re.split(r"(?:[\r\n;]+|(?:\.|\?|!)(?:\s+|$)|(?:\s*,\s*|\s+)(?=(?:and\s+then|then|next\s+(?:medicine|drug)|second\s+medicine|third\s+medicine|also\s+(?:give|take|start|prescribe|add|apply)|plus)\b))+", query, flags=re.IGNORECASE) if s.strip()]
        
        segments = []
        current_order = ""
        stopwords = {
            "take", "give", "tab", "tablet", "tabs", "tablets", "cap", "capsule", "caps", "capsules",
            "syrup", "syp", "inj", "injection", "drops", "orally", "oral", "daily", "twice", "thrice",
            "times", "days", "day", "weeks", "months", "for", "after", "before", "meals", "meal",
            "food", "eating", "breakfast", "lunch", "dinner", "with", "water", "milk", "warm", "cold",
            "regular", "hot", "fluids", "liquid", "empty", "stomach", "fasting", "bedtime", "night",
            "sleep", "morning", "afternoon", "evening", "avoid", "spicy", "oily", "alcohol", "driving",
            "smoking", "rest", "walk", "diet", "bland", "sugar", "sweets", "salt", "steam", "rinse",
            "mouth", "swallow", "whole", "chew", "dissolve", "shake", "well", "complete", "course",
            "strictly", "plenty", "gargle", "consult", "doctor", "confer", "conferred", "patient",
            "recover", "immediately", "urgently", "okay", "alright", "last", "lasts", "more", "than",
            "come", "visit", "blood", "the", "a", "an", "this", "that", "if", "when", "whenever",
            "fever", "pain", "headache", "cough", "cold", "rash", "vomiting", "nausea"
        }
        
        drug_db = get_drug_names_set()
        
        for sent in raw_segments:
            words = [w.upper() for w in re.findall(r"[A-Za-z]{3,}", sent)]
            has_drug = any(w in drug_db for w in words if w.lower() not in stopwords)
            if has_drug:
                if current_order.strip():
                    segments.append(current_order.strip())
                current_order = sent
            else:
                if current_order:
                    current_order += ". " + sent
                else:
                    current_order = sent
        if current_order.strip():
            segments.append(current_order.strip())

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            core_med = re.split(r"(?i)\b(?:for\s+\d+\s*(?:days?|weeks?|months?)|if\s+|in\s+case|whenever|when\s+|as\s+needed|avoid\s+|consult\s+|contact\s+)\b", seg)[0]
            unit_matches = [(m.group(1), m.group(2).lower(), m.start()) for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|l|iu|drops?|puffs?|units?|%|tabs?|caps?|tablets?|capsules?|spoonfuls?|spoons?|sachets?|vials?)\b", core_med, re.IGNORECASE)]
            captured_indices = {m[2] for m in unit_matches}
            all_cands = [(m[0], m[1], m[2]) for m in unit_matches]
            for nm in re.finditer(r"\b(\d+(?:\.\d+)?)\b", core_med):
                num_val = nm.group(1)
                idx = nm.start()
                following = core_med[idx + len(num_val):idx + len(num_val) + 15]
                is_dur = bool(re.match(r"^\s*(?:days?|weeks?|months?)\b", following, re.IGNORECASE))
                if idx not in captured_indices and not is_dur and not re.match(r"^(?:101|111|100|010|001|110|011|1111)$", num_val):
                    all_cands.append((num_val, "", idx))
            all_cands.sort(key=lambda x: x[2])
            all_dose_matches = [(c[0], c[1]) for c in all_cands]
            strength = ""
            
            freq = ""
            if re.search(r"\b(101|1-0-1|1\s*0\s*1)\b", seg):
                freq = "Twice a day (1-0-1)"
            elif re.search(r"\b(111|1-1-1|1\s*1\s*1)\b", seg):
                freq = "Thrice a day (1-1-1)"
            elif re.search(r"\b(100|1-0-0|1\s*0\s*0)\b", seg):
                freq = "Once a day (1-0-0)"
            elif re.search(r"\b(010|0-1-0)\b", seg):
                freq = "Once a day (0-1-0)"
            elif re.search(r"\b(001|0-0-1)\b", seg):
                freq = "Once a day (bedtime)"
            elif re.search(r"\b(110|1-1-0)\b", seg):
                freq = "Twice a day (1-1-0)"
            elif re.search(r"\b(011|0-1-1)\b", seg):
                freq = "Twice a day (0-1-1)"
            elif re.search(r"\b(1111|1-1-1-1)\b", seg):
                freq = "Four times a day (1-1-1-1)"
            elif re.search(r"\b(twice|two times|bid|b\.i\.d)\b", seg, re.IGNORECASE):
                freq = "Twice a day (1-0-1)"
            elif re.search(r"\b(thrice|three times|tid|t\.i\.d)\b", seg, re.IGNORECASE):
                freq = "Thrice a day (1-1-1)"
            elif re.search(r"\b(once\s*(?:a\s*)?day|once daily|od|o\.d)\b", seg, re.IGNORECASE):
                freq = "Once a day (1-0-0)"
            elif re.search(r"\b(four times|qid|q\.i\.d)\b", seg, re.IGNORECASE):
                freq = "Four times a day (1-1-1-1)"
            elif re.search(r"\b(sos|if needed|if required)\b", seg, re.IGNORECASE):
                freq = "If Required (SOS)"
            
            dur_match = re.search(r"\b(?:for\s*)?(\d+)\s*(days?|weeks?|months?)\b", seg, re.IGNORECASE)
            duration = f"{dur_match.group(1)} {dur_match.group(2)}" if dur_match else ""
            
            # Multi-instruction extraction (Primary + Secondary combined with semicolon)
            instructions = []
            if re.search(r"\bafter\s*breakfast\b", seg, re.IGNORECASE):
                instructions.append("after breakfast")
            elif re.search(r"\bafter\s*lunch\b", seg, re.IGNORECASE):
                instructions.append("after lunch")
            elif re.search(r"\bafter\s*dinner\b", seg, re.IGNORECASE):
                instructions.append("after dinner")
            elif re.search(r"\bafter\s*(?:meals?|food|eating)\b", seg, re.IGNORECASE):
                instructions.append("after meals")

            if re.search(r"\bbefore\s*breakfast\b", seg, re.IGNORECASE):
                instructions.append("before breakfast")
            elif re.search(r"\bbefore\s*lunch\b", seg, re.IGNORECASE):
                instructions.append("before lunch")
            elif re.search(r"\bbefore\s*dinner\b", seg, re.IGNORECASE):
                instructions.append("before dinner")
            elif re.search(r"\bbefore\s*(?:meals?|food)\b", seg, re.IGNORECASE):
                instructions.append("before meals")

            if re.search(r"\bwith\s*(?:meals?|food)\b", seg, re.IGNORECASE):
                instructions.append("with meals")
            if re.search(r"\b(?:on\s*(?:an?\s*)?empty\s*stomach|empty\s*stomach)\b", seg, re.IGNORECASE):
                instructions.append("on empty stomach")
            if re.search(r"\b(?:at\s*bedtime|before\s*sleep|at\s*night)\b", seg, re.IGNORECASE):
                instructions.append("at bedtime")
            if re.search(r"\bwith\s*(?:warm|hot)\s*water\b", seg, re.IGNORECASE):
                instructions.append("with warm water")
            elif re.search(r"\bwith\s*(?:cold|regular|clean)?\s*water\b", seg, re.IGNORECASE) and "with warm water" not in instructions:
                instructions.append("with water")
            if re.search(r"\bwith\s*milk\b", seg, re.IGNORECASE):
                instructions.append("with milk")
            if re.search(r"\brinse\s*mouth(?:\s*after\s*use)?\b", seg, re.IGNORECASE):
                instructions.append("rinse mouth after use")
            if re.search(r"\b(?:swallow\s*whole|do\s*not\s*chew)\b", seg, re.IGNORECASE):
                instructions.append("swallow whole (do not chew)")
            if re.search(r"\bavoid\s*(?:oily\s*(?:and\s*)?spicy\s*food|oily\s*food|spicy\s*food)\b", seg, re.IGNORECASE):
                instructions.append("avoid oily and spicy food")
            if re.search(r"\bavoid\s*alcohol\b", seg, re.IGNORECASE):
                instructions.append("avoid alcohol")
            if re.search(r"\b(?:drink\s*plenty\s*of\s*water|plenty\s*of\s*fluids)\b", seg, re.IGNORECASE):
                instructions.append("drink plenty of fluids")
            if re.search(r"\bcomplete\s*(?:the\s*)?(?:full\s*)?course\b", seg, re.IGNORECASE):
                instructions.append("complete full course")

            # Contingency & Doctor consultation
            contingency_m = re.search(r"\bif\s+(?:the\s+)?(?:patient|fever|pain|condition|symptoms?|cough|headache|vomiting|infection|rash)?\s*(?:does\s*not|doesn't|not|fails\s*to|persists?|worsens?|increases?|lasts?)\s*(?:recover|improve|subside|go\s*away|decrease|reduce|respond|get\s*better|for\s+more\s+than\s+\d+\s*days?)?\b", seg, re.IGNORECASE)
            consult_m = re.search(r"\b(?:consult|confer(?:red)?|confirm|contact|visit|see|call|inform|report\s*to|meet|come\s+visit)\s*(?:with\s*)?(?:the\s*)?(?:doctor|physician|hospital|clinic|emergency)(?:\s*immediately|\s*urgently|\s*sos|\s*asap)?\b", seg, re.IGNORECASE)

            if contingency_m and consult_m:
                cond_text = re.sub(r"\bthe\s+", "", contingency_m.group(0), flags=re.IGNORECASE).strip()
                doc_text = re.sub(r"\bconfer(?:red)?\b", "consult", consult_m.group(0), flags=re.IGNORECASE)
                doc_text = re.sub(r"\bthe\s+doctor\b", "doctor", doc_text, flags=re.IGNORECASE).strip()
                instructions.append(f"{cond_text}, {doc_text}")
            elif contingency_m:
                instructions.append(f"{contingency_m.group(0).strip()}, review with doctor")
            elif consult_m:
                doc_text = re.sub(r"\bconfer(?:red)?\b", "consult", consult_m.group(0), flags=re.IGNORECASE)
                doc_text = re.sub(r"\bthe\s+doctor\b", "doctor", doc_text, flags=re.IGNORECASE).strip()
                instructions.append(doc_text)

            # Tail advice fallback
            if not instructions:
                dur_tail_m = re.search(r"\b(?:for\s*)?\d+\s*(?:days?|weeks?|months?)\s+(.+)$", seg, re.IGNORECASE)
                if dur_tail_m:
                    tail = re.sub(r"\b(?:okay|ok|alright|fine|thank\s*you|thanks|please|next|done)\s*$", "", dur_tail_m.group(1), flags=re.IGNORECASE).strip()
                    tail = re.sub(r"\bconfer(?:red)?\s*(?:with\s*)?(?:the\s*)?doctor\b", "consult doctor", tail, flags=re.IGNORECASE).strip()
                    if len(tail) > 5:
                        instructions.append(tail)

            inst = "; ".join(dict.fromkeys(instructions)) if instructions else ""

            clean_words = re.findall(r"[A-Za-z]{3,}", seg)
            stopwords = {
                "take", "give", "tab", "tablet", "tabs", "tablets", "cap", "capsule", "caps", "capsules",
                "syrup", "syp", "inj", "injection", "drops", "orally", "oral", "daily", "twice", "thrice",
                "times", "days", "day", "weeks", "months", "for", "after", "before", "meals", "meal",
                "food", "eating", "breakfast", "lunch", "dinner", "with", "water", "milk", "warm", "cold",
                "regular", "hot", "fluids", "liquid", "empty", "stomach", "fasting", "bedtime", "night",
                "sleep", "morning", "afternoon", "evening", "avoid", "spicy", "oily", "alcohol", "driving",
                "smoking", "rest", "walk", "diet", "bland", "sugar", "sweets", "salt", "steam", "rinse",
                "mouth", "swallow", "whole", "chew", "dissolve", "shake", "well", "complete", "course",
                "strictly", "plenty", "gargle", "consult", "doctor", "confer", "conferred", "patient",
                "recover", "immediately", "urgently", "okay", "alright", "last", "lasts", "more", "than",
                "come", "visit", "blood"
            }
            drug_cands = [w for w in clean_words if w.lower() not in stopwords and (not drug_db or w.upper() in drug_db)]

            if not drug_cands and parsed_records:
                if inst:
                    existing = parsed_records[-1]["instruction"]
                    if not existing or existing == "NONE":
                        parsed_records[-1]["instruction"] = inst
                    else:
                        combined = list(dict.fromkeys(existing.split("; ") + instructions))
                        parsed_records[-1]["instruction"] = "; ".join(combined)
                continue

            if not drug_cands:
                continue

            cand_name = drug_cands[0].upper()

            # Clinical Dosage Rules 1 & 2
            from agents.medicine_strength_agent import _get_drug_strengths_map
            db_map = _get_drug_strengths_map()
            db_strengths = set()
            for k, v in db_map.items():
                if cand_name in k or k in cand_name:
                    db_strengths.update(v)

            if len(all_dose_matches) >= 2:
                # Rule 1: medicine + (int + unit) + (int + unit) == final int + unit is dose
                final_cand = all_dose_matches[-1]
                strength = f"{final_cand[0]} {final_cand[1]}".strip()
                if all_dose_matches[0][0] in db_strengths:
                    cand_name = f"{cand_name} {all_dose_matches[0][0]}"
            elif len(all_dose_matches) == 1:
                cand_num = all_dose_matches[0][0]
                if cand_num in db_strengths:
                    # Rule 2: medicine + (int + unit) == in DB -> no dose, just medicine name with integer
                    strength = ""
                    cand_name = f"{cand_name} {cand_num}"
                else:
                    strength = f"{all_dose_matches[0][0]} {all_dose_matches[0][1]}".strip()
            else:
                strength = ""

            parsed_records.append({
                "Drug_name": cand_name,
                "strength": strength,
                "frequency": freq,
                "duration": duration,
                "route": "ORAL",
                "instruction": inst,
                "additional_instruction": ""
            })

        # Deduplicate only identical records
        deduped = []
        for r in parsed_records:
            existing = next((d for d in deduped if d["Drug_name"].split()[0] == r["Drug_name"].split()[0] and d["strength"] == r["strength"] and d["duration"] == r["duration"] and d["frequency"] == r["frequency"]), None)
            if not existing:
                deduped.append(r)
            else:
                if r["instruction"] and r["instruction"] not in existing["instruction"]:
                    existing["instruction"] = f"{existing['instruction']}; {r['instruction']}".strip("; ")
        parsed_records = deduped

        t1 = time.perf_counter()
        generation_time = round(t1 - t0, 4)
        output = "\n\n".join([f"Drug_name: {r['Drug_name']}\nstrength: {r['strength']}\nfrequency: {r['frequency']}\nduration: {r['duration']}\nroute: {r['route']}\ninstruction: {r['instruction']}" for r in parsed_records])
        agent_logs = [{"agent": "FastRelationalFlowsheet", "status": "completed", "time": generation_time}]
        aggregated_blocks = parsed_records
    else:
        t0 = time.perf_counter()
        output, generation_time, agent_logs, aggregated_blocks = run_graph_extraction(None, query)
        t1 = time.perf_counter()
        # 3. Parse output fields
        parsed_records = parse_output_fields(output, query=query)

    # 4. Save to DB and export files
    db.add_history(proc_id, query, output, generation_time)
    if not csv_path or not xlsx_path:
        proc_data = db.get_process(proc_id)
        if proc_data:
            csv_path = proc_data.get("csv_path")
            xlsx_path = proc_data.get("xlsx_path")

    if csv_path and xlsx_path:
        exporter.append_generation(
            csv_path,
            xlsx_path,
            query,
            output,
            generation_time=generation_time,
            llm_model_used=target_llm_label,
        )

    return {
        "success": True,
        "process_id": proc_id,
        "raw_query": query,
        "raw_output": output,
        "parsed_records": parsed_records,
        "generation_time": generation_time,
        "agent_logs": agent_logs,
        "total_medicines": len(parsed_records),
    }


@app.post("/api/transcribe")
async def transcribe_speech(
    file: Optional[UploadFile] = File(None),
    stt_model: Optional[str] = Form("whisper_ayush"),
):
    """
    Transcribes uploaded audio speech note using the selected STT engine (Whisper Ayush, Canary, Parakeet, Moonshine).
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    t0 = time.perf_counter()
    try:
        transcript = transcriber.transcribe_audio(audio_bytes, model_key=stt_model)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"STT transcription error: {str(ex)}")
    t1 = time.perf_counter()

    return {
        "success": True,
        "transcript": transcript,
        "stt_model_used": stt_model,
        "transcription_time": round(t1 - t0, 3),
    }


@app.get("/api/history")
def get_history(process_id: Optional[int] = None, limit: int = 50):
    """Retrieves prescription extraction history from SQLite."""
    records = db.get_history(process_id, limit=limit)
    formatted = []
    for r in records:
        parsed = parse_output_fields(r.get("output", ""), query=r.get("query", ""))
        formatted.append({
            "id": r.get("id"),
            "process_id": r.get("process_id"),
            "timestamp": r.get("created_at"),
            "input_text": r.get("query"),
            "output_text": r.get("output"),
            "generation_time": r.get("generation_time"),
            "audio_path": r.get("audio_path"),
            "parsed_records": parsed,
        })
    return {"total": len(formatted), "records": formatted}


@app.get("/api/threads")
def get_threads():
    """Lists all active and archived process threads."""
    threads = db.list_processes()
    return {"threads": threads}


@app.post("/api/threads")
def create_thread(req: ThreadCreateRequest):
    """Creates a new process thread."""
    csv_path, xlsx_path = exporter.new_output_paths(req.name)
    model_name = config.MODEL_OPTIONS.get(req.model_label, config.MODEL_NAME)
    proc_id = db.create_process(req.name, req.device, csv_path, xlsx_path, model_name=model_name, model_label=req.model_label)
    return {
        "success": True,
        "process_id": proc_id,
        "name": req.name,
        "device": req.device,
        "model_label": req.model_label,
        "csv_path": csv_path,
        "xlsx_path": xlsx_path,
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    Bidirectional WebSocket endpoint for sub-second streaming audio transcription.
    Accepts raw PCM16 mono audio frames from Web Audio API.
    Uses FastLiveTranscriber with proactive push queue and instant finalize.
    """
    await websocket.accept()
    stt_model = websocket.query_params.get("stt_model") or "whisper_ayush"
    try:
        sample_rate = int(websocket.query_params.get("sample_rate", 16000))
    except Exception:
        sample_rate = 16000

    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue = asyncio.Queue()

    def _on_partial(data):
        loop.call_soon_threadsafe(out_queue.put_nowait, data)

    # Use FastLiveTranscriber with selected STT model (defaults to whisper_ayush)
    streamer = FastLiveTranscriber(sample_rate=sample_rate, model_key=stt_model, on_partial_callback=_on_partial)
    print(f"[WS] Client connected. Active ASR engine: {stt_model} | Sample Rate: {sample_rate}Hz")

    # Send connection acknowledgment
    await websocket.send_text(json.dumps({
        "type": "connected",
        "stt_model": stt_model,
        "sample_rate": sample_rate,
        "status": "ready"
    }))

    last_sent_speech_state = None

    async def sender_worker():
        while True:
            msg = await out_queue.get()
            if msg is None:
                break
            try:
                await websocket.send_text(json.dumps(msg))
            except Exception:
                break

    sender_task = asyncio.create_task(sender_worker())

    try:
        while True:
            message = await websocket.receive()

            # Handle binary audio frame (PCM16 raw samples from Web Audio API)
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                partial_res = streamer.feed_pcm16(audio_bytes)
                # Only push on VAD state transitions or non-empty speech text to avoid flooding
                curr_speech = partial_res.get("is_speech", False)
                if curr_speech != last_sent_speech_state:
                    last_sent_speech_state = curr_speech
                    await out_queue.put(partial_res)

            # Handle text/control JSON command
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    data = {"action": message["text"]}

                action = data.get("action", "").lower()
                if action in ("stop", "finalize"):
                    final_res = await asyncio.to_thread(streamer.finalize)
                    print(f"[WS] Finalized: '{final_res.get('punctuated_text', '')}' ({final_res.get('final_latency_ms')}ms)")
                    await out_queue.put(final_res)
                elif action == "reset":
                    streamer.reset()
                    last_sent_speech_state = None
                    await out_queue.put({"type": "reset", "status": "cleared"})

    except (WebSocketDisconnect, RuntimeError):
        print("[WS] Client disconnected cleanly.")
    except Exception as e:
        print(f"[WS] WebSocket error: {e}")
    finally:
        await out_queue.put(None)
        sender_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"[INFO] Starting Agentic Prescription Extractor FastAPI Server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
