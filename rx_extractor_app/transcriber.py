"""
transcriber.py [DEPRECATED]
---------------------------
DEPRECATED: Active STT inference and model lifecycle management has migrated
to the canonical STT subsystem in `app.stt`.

This module is retained strictly as a backward-compatibility stub that delegates
directly to `app.stt.get_stt_manager()`.
"""
import warnings
warnings.warn(
    "rx_extractor_app.transcriber is deprecated. Use app.stt.get_stt_manager() instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Union, Any, Optional
from app.stt import get_stt_manager


def transcribe_audio(
    audio: Union[bytes, str, Any],
    model_key: str = "whisper_ayush",
    **kwargs,
) -> str:
    """
    Backwards-compatible wrapper delegating to canonical STTManager.
    """
    mgr = get_stt_manager()
    res = mgr.transcribe(audio, model_key=model_key, **kwargs)
    return res.text


def get_stt_pipeline(model_key: str = "whisper_ayush"):
    """
    Backwards-compatible loader delegating to canonical STTManager engine.
    """
    mgr = get_stt_manager()
    engine = mgr.get_engine(model_key)
    return {"engine": engine, "model_key": model_key}
