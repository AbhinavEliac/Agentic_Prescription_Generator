"""
app/stt/adapters/mock_engine.py
-------------------------------
Deterministic, zero-weight mock STT engine for fast automated testing and CI.
"""

from __future__ import annotations
import time
from typing import Union, Any, Optional
from pathlib import Path
import numpy as np

from app.stt.base import STTEngine, prepare_audio_input, TranscriptionError
from app.stt.schemas import TranscriptionResult, TranscriptionSegment


class MockSTTEngine(STTEngine):
    """
    Mock STT engine that returns predetermined or synthetic transcriptions without loading weights.
    """

    def __init__(
        self,
        name: str = "mock",
        device: str = "cpu",
        default_transcript: str = "Take Paracetamol 650 mg twice daily for 5 days.",
        simulate_latency: float = 0.005,
        should_fail: bool = False,
    ):
        super().__init__(name=name, device=device, compute_type="mock")
        self.default_transcript = default_transcript
        self.simulate_latency = simulate_latency
        self.should_fail = should_fail
        self.call_count: int = 0

    def load(self) -> None:
        """Simulates loading weights."""
        if self.should_fail:
            raise RuntimeError("Simulated model load failure.")
        self._is_loaded = True

    def unload(self) -> None:
        """Simulates unloading weights."""
        self._is_loaded = False

    def transcribe(
        self,
        audio: Union[bytes, str, Path, np.ndarray, Any],
        override_text: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Simulates transcription on polymorphic audio input."""
        if not self._is_loaded:
            self.load()

        if self.should_fail:
            raise TranscriptionError("Simulated inference error.", model_name=self.name)

        t0 = time.perf_counter()
        pcm_data, file_path, temp_path = prepare_audio_input(audio)

        # Simulate brief latency if configured
        if self.simulate_latency > 0:
            time.sleep(self.simulate_latency)

        self.call_count += 1
        text = override_text or self.default_transcript

        # Generate realistic segments
        words = text.split()
        segments = []
        if len(words) > 0:
            seg_len = 1.5
            for i, word in enumerate(words):
                segments.append(
                    TranscriptionSegment(
                        text=word,
                        start=round(i * 0.3, 2),
                        end=round((i + 1) * 0.3, 2),
                        confidence=0.98,
                    )
                )

        latency = time.perf_counter() - t0

        # Clean up temp file if created
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

        return TranscriptionResult(
            text=text,
            language="en",
            segments=segments,
            timestamps=[(s.start or 0.0, s.end or 0.0) for s in segments],
            confidence=0.98,
            model=self.name,
            latency=latency,
        )
