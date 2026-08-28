"""
transcriber.py
--------------
Multi-Model Speech-to-Text Transcriber Module.

Supports dynamic toggling across state-of-the-art open-source STT models:
- Whisper Ayush (Fine-Tuned Turbo Rx v1)
- NVIDIA Canary 1B (High-Accuracy Multilingual)
- NVIDIA Parakeet TDT 1.1B (Ultra-Low Latency Streaming)
- Useful Sensors Moonshine (Base & Tiny for Edge Devices)
- OpenAI Whisper (Large v3 Turbo, Base, Tiny)

Cached with @st.cache_resource for instant zero-latency subsequent inference.
"""
import os
import shutil
import tempfile
import streamlit as st
import config

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


@st.cache_resource
def get_stt_pipeline(model_key: str = "whisper_ayush"):
    """
    Loads and caches the selected Speech-to-Text model pipeline.
    """
    ayush_path = getattr(config, "AYUSH_WHISPER_PATH", "")

    # 1. Ayush's Fine-Tuned Whisper Model
    if model_key == "whisper_ayush":
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            if os.path.exists(os.path.join(ayush_path, "model.safetensors")) or os.path.exists(os.path.join(ayush_path, "pytorch_model.bin")):
                processor = AutoProcessor.from_pretrained(ayush_path)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(ayush_path, dtype=torch_dtype)
                model.to(device)
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    dtype=torch_dtype,
                    device=device,
                )
                return {"engine": "transformers_ayush_local", "pipeline": pipe, "name": "Whisper Ayush"}
            elif os.path.exists(ayush_path):
                processor = AutoProcessor.from_pretrained(ayush_path)
                model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo", dtype=torch_dtype)
                model.to(device)
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    dtype=torch_dtype,
                    device=device,
                )
                return {"engine": "transformers_ayush_turbo", "pipeline": pipe, "name": "Whisper Ayush"}
        except Exception:
            pass

    # 2. OpenAI Whisper Large v3 Turbo (HuggingFace Transformers)
    elif model_key == "whisper_large_turbo":
        try:
            from transformers import pipeline
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=torch_dtype, device=device)
            return {"engine": "transformers_pipeline", "pipeline": pipe, "name": "Whisper Large v3 Turbo"}
        except Exception:
            pass

    # 3. Useful Sensors Moonshine Base & Tiny (Edge / Mobile Optimized)
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

    # 5. Local OpenAI Whisper (Base / Tiny) Fast Fallback
    whisper_size = "tiny" if "tiny" in model_key else "base"
    try:
        import whisper
        whisper_model = whisper.load_model(whisper_size)
        return {"engine": "whisper_standard", "model": whisper_model, "name": f"OpenAI Whisper ({whisper_size})"}
    except Exception as e:
        return {"engine": "error", "error": str(e), "name": "Error"}


def transcribe_audio(audio_data, model_key: str = None) -> str:
    """
    Transcribes audio bytes or file buffer to text string using the selected STT model.
    Returns the transcribed text string.
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
            try:
                result = pipe(tmp_path, generate_kwargs={"language": "english"})
            except Exception:
                result = pipe(tmp_path)
            return result.get("text", "").strip()
        elif engine_type == "whisper_standard":
            model = stt_engine["model"]
            result = model.transcribe(tmp_path, fp16=False)
            return result.get("text", "").strip()
        else:
            raise RuntimeError(stt_engine.get("error", "Failed to initialize STT model engine."))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
