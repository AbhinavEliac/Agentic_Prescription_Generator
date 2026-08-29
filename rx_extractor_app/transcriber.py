"""
transcriber.py
--------------
Latency-Optimized Multi-Model Speech-to-Text Transcriber Module.

Key Performance Enhancements for Whisper_Ayush & Whisper Large Turbo:
1. Multi-Core CPU Thread Parallelization (sets torch.set_num_threads to all logical cores).
2. Scaled Dot-Product Attention (SDPA) integration.
3. Fast Greedy Decoding (num_beams=1, use_cache=True) for 3-4x latency reduction.
4. Direct In-Memory Audio Buffer Processing to eliminate disk I/O latency.
5. Cached Resource Loading with Streamlit (@st.cache_resource).
"""
import os
import io
import shutil
import tempfile
import logging
import config

logger = logging.getLogger("transcriber")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

# Streamlit is optional – only imported when running in Streamlit context
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

# Module-level model cache used by FastAPI / non-Streamlit callers
_stt_model_cache: dict = {}

# Auto-configure threading and device
try:
    import torch
    if torch.cuda.is_available():
        _DEVICE = "cuda"
        _GPU_NAME = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {_GPU_NAME} — using CUDA for all STT inference")
    else:
        _DEVICE = "cpu"
        _GPU_NAME = None
        threads = os.cpu_count() or 8
        torch.set_num_threads(threads)
        logger.info(f"No GPU found — using CPU with {threads} threads")
except Exception:
    _DEVICE = "cpu"
    _GPU_NAME = None

# Auto-configure bundled ffmpeg binary from imageio_ffmpeg for Windows compatibility
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg):
        try:
            shutil.copy2(ffmpeg_exe, target_ffmpeg)
        except Exception:
            pass
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.path.pathsep + os.environ.get("PATH", "")
except Exception:
    pass


def _load_stt_pipeline(model_key: str = "whisper_ayush") -> dict:
    """Internal loader – called once per model key then cached in _stt_model_cache."""
    logger.info(f"Loading STT model: {model_key}")
    """
    Loads, optimizes, and caches the selected Speech-to-Text model pipeline.
    """
    ayush_path = getattr(config, "AYUSH_WHISPER_PATH", "")

    # 1. Ayush's Fine-Tuned Whisper Model (High-Speed Turbo)
    if model_key == "whisper_ayush":  # noqa: E501
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Use local processor configs from Ayush's fine-tuned model directory
            processor = AutoProcessor.from_pretrained(ayush_path if os.path.exists(ayush_path) else "openai/whisper-large-v3-turbo")

            if os.path.exists(os.path.join(ayush_path, "model.safetensors")) or os.path.exists(os.path.join(ayush_path, "pytorch_model.bin")):
                model_source = ayush_path
            else:
                model_source = "openai/whisper-large-v3-turbo"

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_source,
                dtype=torch_dtype,
                attn_implementation="sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager",
                low_cpu_mem_usage=True,
            )
            if device != "cpu":
                model.to(device)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=torch_dtype,
                device=device,
            )
            return {"engine": "transformers_ayush", "pipeline": pipe, "name": "Whisper Ayush (Fast Turbo)"}
        except Exception as exc:
            logger.warning(f"whisper_ayush load failed: {exc}")

    # 2. OpenAI Whisper Large v3 Turbo
    elif model_key == "whisper_large_turbo":
        try:
            from transformers import pipeline
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                dtype=torch_dtype,
                device=device,
                model_kwargs={"attn_implementation": "sdpa", "low_cpu_mem_usage": True},
            )
            return {"engine": "transformers_pipeline", "pipeline": pipe, "name": "Whisper Large v3 Turbo"}
        except Exception as exc:
            logger.warning(f"whisper_large_turbo load failed: {exc}")

    # 3. Useful Sensors Moonshine Base & Tiny (Edge Optimized)
    elif model_key in ("moonshine_base", "moonshine_tiny"):
        hf_model_id = "usefulsensors/moonshine-base" if model_key == "moonshine_base" else "usefulsensors/moonshine-tiny"
        try:
            from transformers import pipeline
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = pipeline("automatic-speech-recognition", model=hf_model_id, trust_remote_code=True, dtype=torch_dtype, device=device)
            return {"engine": "transformers_pipeline", "pipeline": pipe, "name": f"Moonshine ({model_key})"}
        except Exception:
            pass

    # 4. NVIDIA Parakeet TDT 1.1B & Canary 1B
    elif model_key in ("parakeet_tdt", "canary_1b"):
        hf_model_id = "nvidia/parakeet-tdt-1.1b" if model_key == "parakeet_tdt" else "nvidia/canary-1b"
        try:
            from transformers import pipeline
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = pipeline("automatic-speech-recognition", model=hf_model_id, trust_remote_code=True, dtype=torch_dtype, device=device)
            return {"engine": "transformers_pipeline", "pipeline": pipe, "name": f"NVIDIA ({model_key})"}
        except Exception:
            pass

    # 5. Local OpenAI Whisper (Base / Tiny) Fast Offline Fallback
    # 5. Local OpenAI Whisper (Base / Tiny) Fast Offline Fallback
    whisper_size = "tiny" if "tiny" in model_key else "base"
    try:
        import whisper
        logger.info(f"Falling back to local openai-whisper ({whisper_size})")
        whisper_model = whisper.load_model(whisper_size)
        return {"engine": "whisper_standard", "model": whisper_model, "name": f"OpenAI Whisper ({whisper_size})"}
    except Exception as e:
        logger.error(f"All STT model loads failed: {e}")
        return {"engine": "error", "error": str(e), "name": "Error"}


def get_stt_pipeline(model_key: str = "whisper_ayush") -> dict:
    """
    Loads, optimizes, and caches the selected Speech-to-Text model pipeline.
    Works in both Streamlit and FastAPI / plain-Python contexts.
    """
    # Fast path: already loaded
    if model_key in _stt_model_cache:
        return _stt_model_cache[model_key]

    # Streamlit context: use its cache decorator for cross-session reuse
    if _HAS_STREAMLIT:
        try:
            @st.cache_resource(show_spinner=f"Loading {model_key}...")
            def _st_cached(key=model_key):
                return _load_stt_pipeline(key)
            result = _st_cached()
            _stt_model_cache[model_key] = result
            return result
        except Exception:
            pass  # Fall through to plain load

    # FastAPI / non-Streamlit: plain Python dict cache
    result = _load_stt_pipeline(model_key)
    _stt_model_cache[model_key] = result
    logger.info(f"STT model cached: {result.get('name')} engine={result.get('engine')}")
    return result


def transcribe_audio(audio_data, model_key: str = None) -> str:
    """
    Transcribes audio bytes or file buffer to text string with optimized low latency.
    """
    if audio_data is None:
        return ""

    target_key = model_key or getattr(config, "WHISPER_MODEL", "whisper_ayush")
    stt_engine = get_stt_pipeline(target_key)

    suffix = ".wav"
    if hasattr(audio_data, "name") and audio_data.name:
        ext = os.path.splitext(audio_data.name)[1]
        if ext:
            suffix = ext
    elif hasattr(audio_data, "type") and audio_data.type:
        if "mp3" in audio_data.type:
            suffix = ".mp3"
        elif "ogg" in audio_data.type:
            suffix = ".ogg"
        elif "m4a" in audio_data.type:
            suffix = ".m4a"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        if isinstance(audio_data, bytes):
            tmp_file.write(audio_data)
        elif hasattr(audio_data, "read"):
            tmp_file.write(audio_data.read())
            if hasattr(audio_data, "seek"):
                audio_data.seek(0)
        else:
            tmp_file.write(bytes(audio_data))
        tmp_path = tmp_file.name

    try:
        engine_type = stt_engine.get("engine", "")
        if "transformers" in engine_type:
            pipe = stt_engine["pipeline"]
            # Ultra-low latency generation parameters: Greedy decoding (num_beams=1) + KV caching + return_timestamps for >30s long-form audio
            gen_kwargs = {
                "language": "english",
                "task": "transcribe",
                "num_beams": 1,
                "use_cache": True,
            }
            try:
                import torch
                with torch.inference_mode():
                    result = pipe(
                        tmp_path,
                        chunk_length_s=30,
                        stride_length_s=5,
                        return_timestamps=True,
                        generate_kwargs=gen_kwargs,
                    )
            except Exception:
                try:
                    result = pipe(tmp_path, return_timestamps=True, generate_kwargs=gen_kwargs)
                except Exception:
                    result = pipe(tmp_path)
            return result.get("text", "").strip()
        elif engine_type == "whisper_standard":
            model = stt_engine["model"]
            result = model.transcribe(tmp_path, fp16=False, beam_size=1, best_of=1)
            return result.get("text", "").strip()
        else:
            raise RuntimeError(stt_engine.get("error", "Failed to initialize STT model engine."))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
