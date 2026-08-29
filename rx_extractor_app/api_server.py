"""
api_server.py
-------------
FastAPI backend service exposing the Agentic Prescription Extractor and STT engines.
Enables parallel Node.js, Web, and mobile applications to interface with the LangGraph pipeline.
"""
import os
import io
import time
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
from fast_streaming_transcriber import FastLiveTranscriber, _get_whisper_model

# Initialize database
db.init_db()


@asynccontextmanager
async def lifespan(app_instance):
    """Pre-warm the whisper tiny model on server startup to eliminate cold-start latency."""
    print("[Startup] Pre-warming whisper tiny model for sub-second streaming...")
    threading.Thread(target=_get_whisper_model, args=("tiny",), daemon=True).start()
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

    # 2. Run LangGraph Multi-Agent pipeline
    target_llm_label = req.llm_model or config.DEFAULT_MODEL_LABEL
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
    Uses FastLiveTranscriber (whisper tiny, ~200-400ms CPU latency).
    Emits real-time partial JSON updates and finalized punctuated results.
    """
    await websocket.accept()
    stt_model = websocket.query_params.get("stt_model") or "whisper_ayush"
    try:
        sample_rate = int(websocket.query_params.get("sample_rate", 16000))
    except Exception:
        sample_rate = 16000

    # Use FastLiveTranscriber (whisper tiny) for sub-second latency
    streamer = FastLiveTranscriber(sample_rate=sample_rate, model_key=stt_model)
    print(f"[WS] Client connected. FastASR engine | Sample Rate: {sample_rate}Hz")

    # Send connection acknowledgment
    await websocket.send_text(json.dumps({
        "type": "connected",
        "stt_model": "whisper_tiny_fast",
        "sample_rate": sample_rate,
        "status": "ready"
    }))

    try:
        while True:
            message = await websocket.receive()
            
            # Handle binary audio frame (PCM16 raw samples from Web Audio API)
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                partial_res = streamer.feed_pcm16(audio_bytes)
                await websocket.send_text(json.dumps(partial_res))
                
            # Handle text/control JSON command
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    data = {"action": message["text"]}

                action = data.get("action", "").lower()
                if action in ("stop", "finalize"):
                    final_res = streamer.finalize()
                    print(f"[WS] Finalized: '{final_res.get('punctuated_text', '')}' ({final_res.get('final_latency_ms')}ms)")
                    await websocket.send_text(json.dumps(final_res))
                elif action == "reset":
                    streamer.reset()
                    await websocket.send_text(json.dumps({"type": "reset", "status": "cleared"}))

    except (WebSocketDisconnect, RuntimeError):
        print("[WS] Client disconnected cleanly.")
    except Exception as e:
        print(f"[WS] WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"[INFO] Starting Agentic Prescription Extractor FastAPI Server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
