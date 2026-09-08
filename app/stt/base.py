"""
app/stt/base.py
---------------
Abstract base classes, standardized exceptions, and polymorphic audio input handlers.
"""

from __future__ import annotations
import os
import io
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Tuple, Any, BinaryIO
import numpy as np

from app.stt.schemas import TranscriptionResult


class STTError(Exception):
    """Base exception for all Speech-to-Text subsystem errors."""
    def __init__(self, message: str, model_name: str = "unknown", details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.model_name = model_name
        self.details = details

    def __str__(self) -> str:
        return f"[{self.model_name}] {self.message}"


class ModelNotFoundError(STTError):
    """Raised when a requested STT model key or weights directory is not found."""
    pass


class ModelLoadError(STTError):
    """Raised when an STT engine fails to load weights or initialize hardware context."""
    pass


class TranscriptionError(STTError):
    """Raised when speech inference fails on audio data."""
    pass


class AudioFormatError(STTError):
    """Raised when audio data is corrupt, empty, or unreadable."""
    pass


def prepare_audio_input(
    audio: Union[bytes, str, Path, np.ndarray, BinaryIO, Any],
    target_sample_rate: int = 16000,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    Polymorphic audio input normalizer.
    Accepts:
      - bytes / bytearray (raw audio bytes)
      - str / Path (existing file path on disk)
      - np.ndarray (float32 or int16 PCM audio array)
      - File-like objects with .read()
    Returns:
      (numpy_pcm_array, file_path_on_disk, cleanup_path_if_temp)
    """
    if audio is None:
        raise AudioFormatError("Audio input cannot be None.")

    temp_path: Optional[str] = None
    file_path: Optional[str] = None
    pcm_data: Optional[np.ndarray] = None

    # 1. Input is already a string or Path pointing to an existing file
    if isinstance(audio, (str, Path)):
        p_str = str(audio)
        if not os.path.exists(p_str):
            raise AudioFormatError(f"Audio file not found at: {p_str}")
        if os.path.getsize(p_str) == 0:
            raise AudioFormatError(f"Audio file is empty (0 bytes): {p_str}")
        file_path = p_str

    # 2. Input is a NumPy array
    elif isinstance(audio, np.ndarray):
        if audio.size == 0:
            raise AudioFormatError("NumPy audio array is empty.")
        # Normalize to float32 between -1.0 and 1.0
        if np.issubdtype(audio.dtype, np.integer):
            pcm_data = (audio / 32768.0).astype(np.float32)
        else:
            pcm_data = audio.astype(np.float32)

    # 3. Input is bytes, bytearray, or file-like object
    else:
        raw_bytes: bytes
        if hasattr(audio, "read"):
            raw_bytes = audio.read()
            if hasattr(audio, "seek"):
                audio.seek(0)
        elif isinstance(audio, (bytes, bytearray)):
            raw_bytes = bytes(audio)
        else:
            raw_bytes = bytes(audio)

        if len(raw_bytes) == 0:
            raise AudioFormatError("Audio byte stream is empty (0 bytes).")

        # Write to temporary file for engines that require file paths or soundfile decoding
        suffix = ".wav"
        if hasattr(audio, "name") and audio.name:
            ext = os.path.splitext(audio.name)[1]
            if ext:
                suffix = ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            temp_path = tmp.name
            file_path = temp_path

    # Try to load numpy PCM data if not already present
    if pcm_data is None and file_path is not None:
        try:
            import soundfile as sf
            data, sr = sf.read(file_path, dtype="float32")
            # If stereo, average to mono
            if data.ndim > 1:
                data = data.mean(axis=1)
            # If sample rate differs, simple down/upsample if needed or pass as is
            pcm_data = data
        except Exception:
            # Not all files can be read by soundfile (e.g. some mp3/aac), engines will use file_path
            pcm_data = None

    return pcm_data, file_path, temp_path


class STTEngine(ABC):
    """Abstract base class for all Speech-to-Text inference engines."""

    def __init__(self, name: str, device: str = "cpu", compute_type: str = "auto"):
        self.name = name
        self.device = device
        self.compute_type = compute_type
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded in memory."""
        return self._is_loaded

    @property
    def is_cuda_available(self) -> bool:
        """Checks if CUDA GPU is available for this engine."""
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @abstractmethod
    def load(self) -> None:
        """Explicitly loads model weights into RAM/VRAM. Separate from instantiation."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Explicitly frees model weights and caches from RAM/VRAM."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: Union[bytes, str, Path, np.ndarray, Any],
        **kwargs,
    ) -> TranscriptionResult:
        """
        Executes speech-to-text inference on polymorphic audio input.
        Returns standardized TranscriptionResult.
        """
        pass
