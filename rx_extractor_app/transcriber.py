"""
transcriber.py
--------------
Local offline Whisper speech-to-text transcriber module.

Loads the local Whisper model ('base' or 'tiny') using @st.cache_resource
so model initialization happens once and stays in memory.
Accepts raw audio bytes or Streamlit UploadedFile / AudioInput buffer,
saves to a temporary audio file, runs whisper.transcribe(), and returns text.
"""
import os
import shutil
import tempfile
import streamlit as st
import whisper
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
def get_whisper_model(model_name: str = None):
    target = model_name or getattr(config, "WHISPER_MODEL", "base")
    return whisper.load_model(target)


def transcribe_audio(audio_data, model_name: str = None) -> str:
    """
    Transcribes audio bytes or file buffer to text string using local Whisper.
    Returns the transcript string.
    """
    if audio_data is None:
        return ""

    model = get_whisper_model(model_name)

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
        result = model.transcribe(tmp_path, fp16=False)
        transcript = result.get("text", "").strip()
        return transcript
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
