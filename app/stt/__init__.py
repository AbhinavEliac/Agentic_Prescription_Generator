"""
app.stt package
---------------
Canonical Speech-to-Text (STT) Subsystem.
Provides lazy-loaded multi-model ASR engines, reusable VAD, decoupled streaming,
and standardized transcription schemas with explicit fallback tracking.
"""

from app.stt.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    STTFallbackRecord,
    STTConfig,
)
from app.stt.base import (
    STTEngine,
    STTError,
    ModelNotFoundError,
    ModelLoadError,
    TranscriptionError,
    AudioFormatError,
    prepare_audio_input,
)
from app.stt.vad import VADDetector
from app.stt.streaming import StreamingTranscriber
from app.stt.manager import STTManager, get_stt_manager

__all__ = [
    "TranscriptionResult",
    "TranscriptionSegment",
    "STTFallbackRecord",
    "STTConfig",
    "STTEngine",
    "STTError",
    "ModelNotFoundError",
    "ModelLoadError",
    "TranscriptionError",
    "AudioFormatError",
    "prepare_audio_input",
    "VADDetector",
    "StreamingTranscriber",
    "STTManager",
    "get_stt_manager",
]
