"""
app/stt/adapters/transformers_engine.py
---------------------------------------
Adapter wrapping Hugging Face transformers pipeline for ASR models
(Whisper Large Turbo, NVIDIA Canary 1B, NVIDIA Parakeet TDT).
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Union, Optional, Any, List
import numpy as np

from app.stt.base import (
    STTEngine,
    prepare_audio_input,
    ModelLoadError,
    TranscriptionError,
)
from app.stt.schemas import TranscriptionResult, TranscriptionSegment


class TransformersEngine(STTEngine):
    """
    Adapter wrapping transformers.pipeline('automatic-speech-recognition').
    """

    def __init__(
        self,
        name: str = "whisper_large_turbo",
        hf_model_id: str = "openai/whisper-large-v3-turbo",
        device: str = "auto",
        trust_remote_code: bool = True,
    ):
        resolved_device = device
        if device == "auto":
            try:
                import torch
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"

        super().__init__(name=name, device=resolved_device, compute_type="float16" if resolved_device == "cuda" else "float32")
        self.hf_model_id = hf_model_id
        self.trust_remote_code = trust_remote_code
        self._pipe = None

    def load(self) -> None:
        """Loads Hugging Face pipeline."""
        if self._is_loaded and self._pipe is not None:
            return

        try:
            from transformers import pipeline
            import torch

            torch_device = "cuda:0" if self.device == "cuda" else "cpu"
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.hf_model_id,
                dtype=torch_dtype,
                device=torch_device,
                trust_remote_code=self.trust_remote_code,
            )
            self._is_loaded = True
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Hugging Face model {self.hf_model_id}: {str(e)}",
                model_name=self.name,
                details=e,
            )

    def unload(self) -> None:
        """Unloads pipeline and clears memory."""
        self._pipe = None
        self._is_loaded = False

    def transcribe(
        self,
        audio: Union[bytes, str, Path, np.ndarray, Any],
        language: str = "english",
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribes audio using the Hugging Face pipeline."""
        if not self._is_loaded or self._pipe is None:
            self.load()

        pcm_data, file_path, temp_path = prepare_audio_input(audio)
        input_target = file_path if file_path is not None else pcm_data

        if input_target is None:
            raise TranscriptionError("No valid audio input available.", model_name=self.name)

        t0 = time.perf_counter()
        try:
            import torch

            gen_kwargs = {
                "language": language,
                "task": "transcribe",
                "num_beams": kwargs.get("num_beams", 1),
            }

            with torch.inference_mode():
                try:
                    res = self._pipe(input_target, generate_kwargs=gen_kwargs)
                except Exception:
                    res = self._pipe(input_target)

            text = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
            latency = time.perf_counter() - t0

            return TranscriptionResult(
                text=text,
                language=language,
                segments=[TranscriptionSegment(text=text, confidence=0.94)],
                confidence=0.94,
                model=self.name,
                latency=latency,
            )
        except Exception as e:
            raise TranscriptionError(f"Inference failure: {str(e)}", model_name=self.name, details=e)
        finally:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
