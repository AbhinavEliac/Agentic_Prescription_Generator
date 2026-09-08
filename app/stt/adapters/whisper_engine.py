"""
app/stt/adapters/whisper_engine.py
----------------------------------
Adapter for standard OpenAI Whisper models (e.g. whisper-base, whisper-tiny).
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


class OpenAIWhisperEngine(STTEngine):
    """
    Adapter wrapping official openai-whisper library.
    """

    def __init__(
        self,
        name: str = "whisper_base",
        model_size: str = "base",
        device: str = "cpu",
    ):
        super().__init__(name=name, device=device, compute_type="float32")
        self.model_size = model_size
        self._model = None

    def load(self) -> None:
        """Loads OpenAI Whisper model."""
        if self._is_loaded and self._model is not None:
            return

        try:
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            self._is_loaded = True
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load OpenAI Whisper ({self.model_size}): {str(e)}",
                model_name=self.name,
                details=e,
            )

    def unload(self) -> None:
        """Unloads model."""
        self._model = None
        self._is_loaded = False

    def transcribe(
        self,
        audio: Union[bytes, str, Path, np.ndarray, Any],
        language: str = "en",
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribes audio using OpenAI Whisper."""
        if not self._is_loaded or self._model is None:
            self.load()

        pcm_data, file_path, temp_path = prepare_audio_input(audio)
        input_target = file_path if file_path is not None else pcm_data

        if input_target is None:
            raise TranscriptionError("No valid audio data could be extracted.", model_name=self.name)

        t0 = time.perf_counter()
        try:
            fp16 = (self.device == "cuda")
            result = self._model.transcribe(
                input_target,
                fp16=fp16,
                language=language,
                beam_size=kwargs.get("beam_size", 1),
                best_of=1,
            )

            text = result.get("text", "").strip()
            segments_raw = result.get("segments", [])
            segments: List[TranscriptionSegment] = []

            for s in segments_raw:
                segments.append(
                    TranscriptionSegment(
                        text=s.get("text", "").strip(),
                        start=round(s.get("start", 0.0), 2),
                        end=round(s.get("end", 0.0), 2),
                        confidence=round(np.exp(s.get("avg_logprob", 0.0)), 3) if "avg_logprob" in s else 0.9,
                    )
                )

            latency = time.perf_counter() - t0

            return TranscriptionResult(
                text=text,
                language=result.get("language", language),
                segments=segments,
                timestamps=[(s.start or 0.0, s.end or 0.0) for s in segments],
                confidence=0.92,
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
