"""
app/stt/streaming.py
--------------------
Decoupled streaming audio processor and rolling window coordinator.
Separates real-time audio chunk buffering and VAD from model inference adapters.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, Optional, Any, Union, Callable

from app.stt.base import STTEngine
from app.stt.vad import VADDetector
from app.stt.schemas import TranscriptionResult


class StreamingTranscriber:
    """
    Coordinator for real-time audio streams (e.g. Web Audio API, WebSocket).
    Decoupled from underlying inference engines.
    """

    def __init__(
        self,
        engine: STTEngine,
        sample_rate: int = 16000,
        window_duration_s: float = 3.0,
        step_duration_s: float = 0.5,
        vad_energy_threshold: float = 0.012,
        on_partial_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.engine = engine
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration_s * sample_rate)
        self.step_samples = int(step_duration_s * sample_rate)
        self.vad = VADDetector(sample_rate=sample_rate, energy_threshold=vad_energy_threshold)
        self.on_partial_callback = on_partial_callback

        self.audio_buffer = np.array([], dtype=np.float32)
        self.confirmed_text: str = ""
        self.interim_text: str = ""
        self.samples_since_last_decode: int = 0
        self.last_speech_time: float = time.time()
        self.is_active: bool = False

    def feed_chunk(self, chunk: Union[bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Processes an incoming chunk of audio samples (PCM16 bytes or float32 array).
        Returns current state including VAD status and interim transcription.
        """
        # Convert PCM16 bytes to float32 if needed
        if isinstance(chunk, (bytes, bytearray)):
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            pcm = chunk.astype(np.float32)

        dur = round(len(self.audio_buffer) / self.sample_rate, 2)
        device_tag = getattr(self.engine, "device", "cpu")

        if len(pcm) == 0:
            return {
                "type": "partial",
                "is_speech": False,
                "interim_text": self.interim_text,
                "text": self.interim_text,
                "final_text": self.confirmed_text,
                "decoded": False,
                "duration": dur,
                "latency_ms": 0.0,
                "device": device_tag,
            }

        self.audio_buffer = np.concatenate([self.audio_buffer, pcm])
        self.samples_since_last_decode += len(pcm)
        dur = round(len(self.audio_buffer) / self.sample_rate, 2)

        is_speech = self.vad.is_speech_frame(pcm)
        if is_speech:
            self.last_speech_time = time.time()

        decoded = False
        latency_ms = 0.0
        # Trigger rolling decode when enough new audio has arrived
        if self.samples_since_last_decode >= self.step_samples and is_speech:
            window = self.audio_buffer[-self.window_samples :]
            t0 = time.perf_counter()
            try:
                res = self.engine.transcribe(window)
                self.interim_text = res.text
                decoded = True
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            except Exception:
                pass
            self.samples_since_last_decode = 0

        state = {
            "type": "partial",
            "is_speech": is_speech,
            "interim_text": self.interim_text,
            "text": self.interim_text,
            "final_text": self.confirmed_text,
            "decoded": decoded,
            "duration": dur,
            "latency_ms": latency_ms,
            "device": device_tag,
        }

        if decoded and self.on_partial_callback:
            try:
                self.on_partial_callback(state)
            except Exception:
                pass

        return state

    def feed_pcm16(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Convenience method for PCM16 audio frames."""
        return self.feed_chunk(audio_bytes)

    def finalize(self) -> TranscriptionResult:
        """
        Finalizes the streaming session, running full inference on all buffered speech audio.
        """
        if len(self.audio_buffer) == 0:
            return TranscriptionResult(text="", model=self.engine.name, latency=0.0)

        t0 = time.perf_counter()
        filtered = self.vad.filter_speech(self.audio_buffer)
        if len(filtered) == 0:
            filtered = self.audio_buffer

        result = self.engine.transcribe(filtered)
        self.confirmed_text = result.text
        self.interim_text = ""
        result.latency = time.perf_counter() - t0
        return result

    def finalize_dict(self) -> Dict[str, Any]:
        """
        Finalizes session and formats payload specifically for WebSocket and UI streaming consumers.
        """
        t0 = time.perf_counter()
        res = self.finalize()
        dur = round(len(self.audio_buffer) / self.sample_rate, 2)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        punctuated = res.text
        try:
            from agents.punctuation_agent import correct_sentence_punctuation
            punctuated = correct_sentence_punctuation(res.text)
        except Exception:
            pass

        return {
            "type": "final",
            "raw_text": res.text,
            "punctuated_text": punctuated,
            "duration": dur,
            "final_latency_ms": latency_ms,
            "model_used": self.engine.name,
            "confidence": res.confidence,
        }

    def reset(self) -> None:
        """Resets stream buffer and VAD."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.confirmed_text = ""
        self.interim_text = ""
        self.samples_since_last_decode = 0
        self.vad.reset()
