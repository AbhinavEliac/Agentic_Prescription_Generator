"""
api_server.py
-------------
FastAPI backend service exposing the Agentic Prescription Extractor and STT engines.
Enables parallel Node.js, Web, and mobile applications to interface with the LangGraph pipeline.
"""
import os
import sys
_APP_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.append(_APP_DIR)
if 'app' in sys.modules and not hasattr(sys.modules['app'], '__path__'):
    del sys.modules['app']

# Prevent torchvision / torchaudio DLL binary incompatibility from crashing transformers on Windows Python 3.13
sys.modules.setdefault('torchvision', None)
sys.modules.setdefault('torchaudio', None)

import re
import io
import time
import uuid
import asyncio
import datetime
import threading
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

logger = logging.getLogger("api_server")

from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

import config
import db
import exporter
from exporter import parse_output_fields
from app.stt import get_stt_manager
from app.stt.streaming import StreamingTranscriber
from app.prescription.pipeline import PrescriptionPipeline, PipelineMode
from app.prescription.validator import ClinicalValidator
from app.drugs import get_drug_repository
from app.api.schemas import (
    PrescriptionExtractionRequest,
    PrescriptionExtractionResponse,
    PrescriptionValidationRequest,
    PrescriptionValidationResponse,
    AudioTranscriptionResponse,
    DrugSearchResponse,
    DrugEntryResponse,
    ApiErrorResponse,
    ErrorDetail,
    ExtractionMode,
)

# Global server initialization metrics
_SERVER_START_TIME = time.time()

# Initialize canonical pipeline, drug repository, and database
canonical_pipeline = PrescriptionPipeline()
drug_repo = get_drug_repository()
db.init_db()


@asynccontextmanager
async def lifespan(app_instance):
    """Pre-warm the Whisper Ayush model on server startup to eliminate cold-start latency."""
    print("[Startup] Pre-warming Whisper Ayush (CTranslate2 Turbo GPU)...")
    def _prewarm():
        try:
            import numpy as np
            from app.stt import get_stt_manager
            mgr = get_stt_manager()
            engine = mgr.get_engine("whisper_ayush")
            # Run 0.1s warm-up pass
            dummy = np.zeros(1600, dtype=np.float32)
            engine.transcribe(dummy)
            print("[Startup] Whisper Ayush canonical engine pre-warmed and ready.")
        except Exception as e:
            print(f"[Startup] Whisper Ayush pre-warming notice: {e}")
    threading.Thread(target=_prewarm, daemon=True).start()
    yield
    print("[Shutdown] FastAPI server shutting down.")


app = FastAPI(
    title="Agentic Prescription Extractor API",
    description="Canonical REST API bridging Clinical Extraction Pipeline and Multi-Engine STT to Node.js and Web clients.",
    version="3.0.0",
    lifespan=lifespan,
)

# Configure CORS securely from environment or safe local developer origins
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:5000,http://127.0.0.1:8080"
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
_allow_credentials = False if "*" in _allowed_origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assigns and propagates a unique X-Request-ID across all incoming requests and responses."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardized HTTP error envelope."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    err = ApiErrorResponse.create(
        error_code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        request_id=req_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=err.model_dump(mode="json"),
        headers={"X-Request-ID": req_id}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Standardized 422 schema validation error envelope."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    details = []
    for error in exc.errors():
        loc = ".".join(str(x) for x in error.get("loc", []))
        details.append(ErrorDetail(
            field=loc,
            code=error.get("type", "value_error"),
            message=error.get("msg", "Validation error"),
        ))
    err = ApiErrorResponse.create(
        error_code="VALIDATION_ERROR",
        message="Request payload failed clinical schema validation.",
        details=details,
        request_id=req_id,
    )
    return JSONResponse(
        status_code=422,
        content=err.model_dump(mode="json"),
        headers={"X-Request-ID": req_id}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Standardized 500 internal server error envelope."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    err = ApiErrorResponse.create(
        error_code="INTERNAL_SERVER_ERROR",
        message=f"An unexpected internal error occurred: {str(exc)}",
        request_id=req_id,
    )
    return JSONResponse(
        status_code=500,
        content=err.model_dump(mode="json"),
        headers={"X-Request-ID": req_id}
    )


@app.get("/api/health")
def get_health(request: Request):
    """Canonical health probe returning service status, hardware, models, and request ID."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    cuda_avail = False
    device_name = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            cuda_avail = True
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status": "healthy",
        "version": "3.0.0",
        "request_id": req_id,
        "uptime_seconds": round(time.time() - _SERVER_START_TIME, 2),
        "environment": getattr(config, "ENVIRONMENT", "development"),
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "default_models": {
            "llm": getattr(config, "MODEL_NAME", "Meta-Llama-3-8B-Instruct.Q4_0.gguf"),
            "stt": getattr(config, "WHISPER_MODEL", "whisper_ayush"),
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }




ExtractRequest = PrescriptionExtractionRequest


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


@app.get("/api/drugs/search")
def search_drugs(
    q: str = Query("", description="Drug search query"),
    limit: int = Query(30, description="Max results"),
    request: Request = None,
):
    """Fast Soundex and prefix search across the master formulary."""
    results = drug_repo.search_drugs(q, limit=limit)
    return {
        "query": q,
        "total": len(results),
        "results": [
            {
                "drug_id": d.drug_id,
                "drug_code": d.drug_code,
                "drug_name": d.drug_name,
                "drug_type": d.drug_type,
                "routes": d.routes,
                "base_name": d.base_name,
            }
            for d in results
        ],
    }


@app.get("/api/drugs/did-you-mean")
def drug_did_you_mean(q: str = Query("", description="Misspelled drug name query")):
    """Phonetic fuzzy drug name recommendation using Soundex + Levenshtein distance."""
    suggestion = drug_repo.find_did_you_mean(q)
    return {"query": q, "suggestion": suggestion}


@app.get("/api/drugs/{drug_id}/routes")
def get_drug_routes(drug_id: str):
    """Permissible anatomical routes for a given formulary drug ID."""
    routes = drug_repo.find_route(drug_id)
    return {"drug_id": drug_id, "routes": routes}


@app.get("/api/drugs/by-brand")
def get_drugs_by_brand(brand: str = Query("", description="Brand name query"), limit: int = Query(10, description="Max results")):
    """Retrieves brand formulations matching a given brand name."""
    entries = drug_repo.find_by_brand(brand, limit=limit)
    return {
        "brand": brand,
        "total": len(entries),
        "results": [e.to_summary_dict() for e in entries],
    }


@app.get("/api/drugs/by-generic")
def get_drugs_by_generic(generic: str = Query("", description="Generic substance query"), limit: int = Query(10, description="Max results")):
    """Retrieves generic formulations and associated brands for an active substance."""
    entries = drug_repo.find_by_generic(generic, limit=limit)
    return {
        "generic": generic,
        "total": len(entries),
        "results": [e.to_summary_dict() for e in entries],
    }


@app.get("/api/drugs/strength")
def get_drug_strength(drug: str = Query("", description="Drug name or ID"), strength: Optional[str] = Query(None, description="Optional strength to validate")):
    """Retrieves or validates recognized formulation strengths for a drug."""
    valid_strengths = drug_repo.find_strength(drug, strength_query=strength)
    return {
        "drug": drug,
        "is_valid": len(valid_strengths) > 0 if strength else None,
        "strengths": valid_strengths,
    }


@app.get("/api/reference-data")
def get_reference_data():
    """Returns clinical reference data for schedules, dose units, and routes."""
    ref = drug_repo.get_reference_data()
    return {
        "schedules": ref.schedules,
        "dose_units": ref.dose_units,
        "routes": ref.routes,
    }


@app.post("/api/prescription/extract")
@app.post("/api/extract")
def extract_prescription(req: PrescriptionExtractionRequest, request: Request = None):
    """
    Canonical endpoint executing the Prescription Extraction Pipeline.
    Returns strongly typed canonical prescription, structured items, and telemetry.
    """
    query = req.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Prescription text cannot be empty.")

    # 1. Process / Thread handling
    proc_id = req.session_id or req.process_id
    if isinstance(proc_id, str):
        try:
            proc_id = int(proc_id)
        except ValueError:
            proc_id = None
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

    # 2. Resolve execution mode
    pipeline_mode = PipelineMode.FAST if (
        req.fast_mode or 
        req.mode == ExtractionMode.DETERMINISTIC_ONLY or 
        req.llm_model == "fast_relational"
    ) else PipelineMode.STANDARD

    t0 = time.perf_counter()
    canonical_rx = canonical_pipeline.extract(query, mode=pipeline_mode)
    t1 = time.perf_counter()
    generation_time = round(t1 - t0, 4)

    # 3. Format structured blocks and string output for backwards-compatible consumers
    parsed_records = canonical_pipeline.extract_legacy_blocks(query, mode=pipeline_mode)
    output = "\n\n".join([
        f"Drug_name: {r['Drug_name']}\nstrength: {r['strength']}\nfrequency: {r['frequency']}\nduration: {r['duration']}\nroute: {r['route']}\ninstruction: {r['instruction']}"
        + (f"\nadditional_instruction: {r['additional_instruction']}" if r.get('additional_instruction') and r['additional_instruction'] != 'NONE' else "")
        for r in parsed_records
    ])
    agent_logs = [{"agent": "CanonicalPrescriptionPipeline", "mode": pipeline_mode.value, "status": "completed", "time": generation_time}]

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
            llm_model_used=req.llm_model or config.DEFAULT_MODEL_LABEL,
        )

    return {
        "success": True,
        "prescription": canonical_rx.model_dump(),
        "execution_time_ms": round(generation_time * 1000, 2),
        "warnings": canonical_rx.validation_warnings,
        # Legacy compatibility fields
        "process_id": proc_id,
        "raw_query": query,
        "raw_output": output,
        "parsed_records": parsed_records,
        "generation_time": generation_time,
        "agent_logs": agent_logs,
        "total_medicines": len(parsed_records),
        "canonical": canonical_rx.model_dump(),
    }


@app.post("/api/prescription/validate")
def validate_prescription(req: PrescriptionValidationRequest, request: Request = None):
    """
    Validates arbitrary prescription items against verbatim raw input.
    Enforces strict clinical groundedness, anti-hallucination, and valid pharmaceutical entity checks.
    """
    validator = ClinicalValidator()
    report = validator.validate_prescription_items(req.items, req.raw_text)
    return report.model_dump(mode="json")


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB DoS protection limit


@app.post("/api/prescription/transcribe")
@app.post("/api/transcribe")
async def transcribe_speech(
    file: Optional[UploadFile] = File(None),
    stt_model: Optional[str] = Form("whisper_ayush"),
    request: Request = None,
):
    """
    Transcribes uploaded audio speech note using the canonical STT subsystem
    (Whisper Ayush CT2, OpenAI Whisper, Canary, Parakeet, Moonshine, Mock).
    Enforces 25MB upload limit and audio file format validation.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    filename = file.filename or "audio.wav"
    ext = os.path.splitext(filename)[1].lower()
    content_type = (file.content_type or "").lower()

    # Validate audio extension or content-type
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS and not content_type.startswith("audio/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type for '{filename}'. Allowed audio formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded audio ({len(audio_bytes)} bytes) exceeds maximum limit of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    t0 = time.perf_counter()
    try:
        mgr = get_stt_manager()
        res = mgr.transcribe(audio_bytes, model_key=stt_model)
        t1 = time.perf_counter()
        return {
            "success": True,
            "transcript": res.text,
            "punctuated_transcript": res.text,
            "stt_model_used": res.model,
            "audio_duration_seconds": res.duration_s,
            "transcription_time_ms": res.latency_ms,
            "device_used": res.model,
            # Legacy compatibility fields
            "transcription_time": round(t1 - t0, 3),
            "latency_ms": res.latency_ms,
            "segments": [s.model_dump() for s in res.segments],
            "fallback": res.fallback_info.model_dump() if res.fallback_info else None,
        }
    except Exception as ex:
        logger.error(f"[API] STT transcription failed: {ex}")
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {str(ex)}")



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
@app.websocket("/ws/v1/transcribe")
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

    # Use canonical StreamingTranscriber with selected STT model from STTManager
    mgr = get_stt_manager()
    try:
        engine = mgr.get_engine(stt_model)
    except Exception as e:
        logger.warning(f"[WS] Failed to load engine '{stt_model}', using default: {e}")
        engine = mgr.get_engine()

    streamer = StreamingTranscriber(
        engine=engine,
        sample_rate=sample_rate,
        on_partial_callback=_on_partial,
    )
    print(f"[WS] Client connected. Active ASR engine: {engine.name} | Sample Rate: {sample_rate}Hz")

    # Send connection acknowledgment
    await websocket.send_text(json.dumps({
        "type": "connected",
        "stt_model": engine.name,
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
                    final_res = await asyncio.to_thread(streamer.finalize_dict)
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
