"""
streaming_transcriber.py
-------------------------
Real-Time Low-Latency Streaming Speech-to-Text Engine for Clinical Dictation.
Processes incremental audio frames over WebSockets with sub-second (<1s) latency.
Powered by SlidingWindowStreamingDecoder and Sub-10ms VAD Engine.
"""
import os
import io
import time
import tempfile
import numpy as np
from typing import Optional, Dict, Any, Tuple
import soundfile as sf

import config
import transcriber
from agents.punctuation_agent import correct_sentence_punctuation
from streaming_engine.vad_detector import VADDetector
from streaming_engine.sliding_window_decoder import SlidingWindowStreamingDecoder


class StreamingAudioBuffer:
    """
    In-memory continuous audio stream buffer with Voice Activity Energy thresholding.
    """
    def __init__(self, sample_rate: int = 16000, max_seconds: float = 120.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_speech_time = time.time()
        self.total_received_bytes = 0
        self.vad = VADDetector(sample_rate=sample_rate)

    def append_pcm16(self, pcm_bytes: bytes):
        """Appends raw 16-bit 16kHz mono PCM bytes directly to the buffer."""
        if not pcm_bytes:
            return
        self.total_received_bytes += len(pcm_bytes)
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, samples])
        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]

    def append_audio_data(self, data: bytes):
        """Appends arbitrary audio bytes (WAV/WebM/PCM) by decoding with soundfile/fallback."""
        if not data:
            return
        try:
            with io.BytesIO(data) as buf:
                samples, sr = sf.read(buf, dtype="float32")
                if len(samples.shape) > 1:
                    samples = samples.mean(axis=1) # Mono mix
                if sr != self.sample_rate:
                    num_target = int(len(samples) * self.sample_rate / sr)
                    samples = np.interp(
                        np.linspace(0, len(samples), num_target),
                        np.arange(len(samples)),
                        samples
                    ).astype(np.float32)
                self.buffer = np.concatenate([self.buffer, samples])
        except Exception:
            self.append_pcm16(data)

        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]

    def is_speech_active(self, energy_threshold: float = 0.008) -> bool:
        """Returns True if the recent 300ms of audio contains human speech."""
        if len(self.buffer) < int(self.sample_rate * 0.1):
            return False
        recent_window = self.buffer[-int(self.sample_rate * 0.3):]
        return self.vad.is_speech_frame(recent_window)

    def get_wav_bytes(self) -> bytes:
        """Encodes current buffer as a standard 16kHz mono WAV byte string."""
        if len(self.buffer) == 0:
            return b""
        bio = io.BytesIO()
        sf.write(bio, self.buffer, self.sample_rate, format="WAV", subtype="PCM_16")
        return bio.getvalue()

    def duration_seconds(self) -> float:
        return len(self.buffer) / self.sample_rate

    def clear(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.total_received_bytes = 0
        self.vad.reset()


class LiveStreamingTranscriber:
    """
    Orchestrates live streaming audio decoding with real-time text emission,
    leveraging sliding-window decoder and sub-10ms VAD segmentation.
    """
    def __init__(self, model_key: Optional[str] = None, sample_rate: int = 16000):
        self.model_key = model_key or getattr(config, "WHISPER_MODEL", "whisper_ayush")
        self.sample_rate = sample_rate
        self.decoder = SlidingWindowStreamingDecoder(
            model_key=self.model_key,
            sample_rate=self.sample_rate,
            window_duration_s=6.0,
            step_duration_s=0.35,
        )
        self.last_transcribed_text = ""
        self.last_inference_time = 0.0
        self.chunks_count = 0

    def feed_audio_chunk(self, chunk_bytes: bytes) -> Dict[str, Any]:
        """
        Ingests an incremental audio slice (PCM16, WebM, or WAV chunk),
        updates sliding buffer, runs greedy ASR decoding, and returns partial transcript payload.
        """
        self.decoder.append_audio_bytes(chunk_bytes)
        self.chunks_count += 1
        
        result = self.decoder.decode_active_window()
        self.last_transcribed_text = result.get("text", "")
        self.last_inference_time = result.get("latency_ms", 0.0)

        return result

    def finalize(self) -> Dict[str, Any]:
        """
        Finalizes the audio stream, runs a complete transcription pass,
        and generates punctuated sentence boundaries for LangGraph ingestion.
        """
        t0 = time.perf_counter()
        final_raw = self.decoder.finalize()

        if not final_raw:
            final_raw = self.last_transcribed_text

        punctuated = correct_sentence_punctuation(final_raw)
        t1 = time.perf_counter()

        return {
            "type": "final",
            "raw_text": final_raw,
            "punctuated_text": punctuated,
            "duration": round(self.decoder.total_audio_received_s, 2),
            "final_latency_ms": round((t1 - t0) * 1000, 2),
        }

    def reset(self):
        """Resets the streaming transcriber state."""
        self.decoder.reset()
        self.last_transcribed_text = ""
        self.last_inference_time = 0.0
        self.chunks_count = 0
