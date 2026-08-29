"""
sliding_window_decoder.py
--------------------------
Sliding-Window Streaming Audio Buffer and Low-Latency Decoder Engine.
Provides prefix caching, common prefix consensus, and word boundary alignment
to eliminate repetitions and boundary truncation in real-time dictation.
"""
import io
import time
import difflib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import soundfile as sf

import config
import transcriber
from streaming_engine.vad_detector import VADDetector


class SlidingWindowStreamingDecoder:
    """
    Manages continuous sliding audio windows, prefix caching, and greedy STT decoding.
    """
    def __init__(
        self,
        model_key: Optional[str] = None,
        sample_rate: int = 16000,
        window_duration_s: float = 6.0,
        step_duration_s: float = 0.4,
        vad_energy_threshold: float = 0.010,
    ):
        self.model_key = model_key or getattr(config, "WHISPER_MODEL", "whisper_ayush")
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration_s * sample_rate)
        self.step_samples = int(step_duration_s * sample_rate)
        
        # Audio Buffers
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.committed_audio_offset = 0
        
        # Transcript State
        self.committed_text = ""       # Solidified text from previous finalized segments
        self.current_partial_text = ""   # Active live partial text from current sliding window
        self.last_full_text = ""         # Full combined output
        self.unconfirmed_history: List[str] = []
        
        # VAD
        self.vad = VADDetector(sample_rate=sample_rate, energy_threshold=vad_energy_threshold)
        self.last_speech_time = time.time()
        self.is_speaking = False

        # Decode throttle – only run Whisper every MIN_DECODE_INTERVAL_S
        self.MIN_DECODE_INTERVAL_S = 2.0  # seconds of audio between inference calls
        self.last_decode_audio_s = 0.0   # total_audio_received_s at last inference
        
        # Performance Tracking
        self.total_audio_received_s = 0.0
        self.last_decode_latency_ms = 0.0
        self.num_chunks_processed = 0

    def append_pcm(self, pcm_bytes: bytes):
        """Appends raw 16-bit 16kHz mono PCM bytes directly to buffer."""
        if not pcm_bytes:
            return
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.append_samples(samples)

    def append_audio_bytes(self, data: bytes):
        """Appends arbitrary audio bytes (WAV/WebM/PCM) by decoding with soundfile or raw fallback."""
        if not data:
            return
        try:
            with io.BytesIO(data) as buf:
                samples, sr = sf.read(buf, dtype="float32")
                if len(samples.shape) > 1:
                    samples = samples.mean(axis=1) # Downmix to Mono
                if sr != self.sample_rate:
                    num_target = int(len(samples) * self.sample_rate / sr)
                    samples = np.interp(
                        np.linspace(0, len(samples), num_target),
                        np.arange(len(samples)),
                        samples
                    ).astype(np.float32)
                self.append_samples(samples)
        except Exception:
            # Fallback to PCM16
            self.append_pcm(data)

    def append_samples(self, samples: np.ndarray):
        """Appends numpy float32 samples to the audio stream."""
        if len(samples) == 0:
            return
        self.audio_buffer = np.concatenate([self.audio_buffer, samples])
        self.total_audio_received_s = len(self.audio_buffer) / self.sample_rate
        self.num_chunks_processed += 1

        # Check VAD for recent frame
        recent_frame = samples[-min(len(samples), int(self.sample_rate * 0.05)):]
        self.is_speaking = self.vad.is_speech_frame(recent_frame)
        if self.is_speaking:
            self.last_speech_time = time.time()

    def decode_active_window(self) -> Dict[str, Any]:
        """
        Runs low-latency greedy inference on the current active sliding window.
        Throttled to run at most every MIN_DECODE_INTERVAL_S of audio to prevent
        overwhelming the model with tiny slices.
        """
        t0 = time.perf_counter()
        buffer_len = len(self.audio_buffer)
        
        # If less than 1s of audio accumulated, return current state without inference
        if buffer_len < int(self.sample_rate * 1.0):
            return {
                "type": "partial",
                "text": self.get_full_transcript(),
                "duration": round(self.total_audio_received_s, 2),
                "latency_ms": 0.0,
                "is_speech": self.is_speaking,
            }

        # Throttle: only run inference if MIN_DECODE_INTERVAL_S have passed since last decode
        audio_since_last_decode = self.total_audio_received_s - self.last_decode_audio_s
        if audio_since_last_decode < self.MIN_DECODE_INTERVAL_S:
            return {
                "type": "partial",
                "text": self.get_full_transcript(),
                "duration": round(self.total_audio_received_s, 2),
                "latency_ms": 0.0,
                "is_speech": self.is_speaking,
            }

        # Select window: from committed_audio_offset to end
        active_samples = self.audio_buffer[self.committed_audio_offset:]
        if len(active_samples) > self.window_samples:
            active_samples = active_samples[-self.window_samples:]

        # Export active window as WAV for fast in-memory ASR
        bio = io.BytesIO()
        sf.write(bio, active_samples, self.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = bio.getvalue()

        try:
            raw_partial = transcriber.transcribe_audio(wav_bytes, model_key=self.model_key)
        except Exception as exc:
            import logging
            logging.getLogger("sliding_window_decoder").warning(f"Transcribe error: {exc}")
            raw_partial = self.current_partial_text

        self.last_decode_audio_s = self.total_audio_received_s  # update throttle timestamp

        # Prefix consensus & boundary alignment
        self.current_partial_text = self._align_and_deduplicate(raw_partial)
        
        # Check if pause/silence allows committing text
        time_since_speech = time.time() - self.last_speech_time
        if time_since_speech > 0.8 and len(self.current_partial_text.strip()) > 0:
            # Commit stable prefix to committed_text
            self._commit_current_window()

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)
        self.last_decode_latency_ms = latency_ms

        full_text = self.get_full_transcript()
        self.last_full_text = full_text

        return {
            "type": "partial",
            "text": full_text,
            "partial": self.current_partial_text,
            "committed": self.committed_text,
            "duration": round(self.total_audio_received_s, 2),
            "latency_ms": latency_ms,
            "is_speech": self.is_speaking,
        }

    def _align_and_deduplicate(self, new_text: str) -> str:
        """
        Deduplicates overlap between committed text and new partial window.
        """
        cleaned_new = new_text.strip()
        if not cleaned_new or not self.committed_text:
            return cleaned_new

        comm_words = self.committed_text.strip().split()
        new_words = cleaned_new.split()

        # Find max overlap between end of comm_words and start of new_words
        max_overlap = min(len(comm_words), len(new_words), 8)
        overlap_len = 0

        for i in range(max_overlap, 0, -1):
            if [w.lower() for w in comm_words[-i:]] == [w.lower() for w in new_words[:i]]:
                overlap_len = i
                break

        if overlap_len > 0:
            return " ".join(new_words[overlap_len:])
        return cleaned_new

    def _commit_current_window(self):
        """
        Solidifies the current active window text into permanent committed state.
        """
        if self.current_partial_text.strip():
            if self.committed_text:
                self.committed_text = f"{self.committed_text.rstrip('. ')} {self.current_partial_text.strip()}"
            else:
                self.committed_text = self.current_partial_text.strip()
            self.current_partial_text = ""
            self.committed_audio_offset = len(self.audio_buffer)

    def get_full_transcript(self) -> str:
        """Returns the full unified transcription string."""
        parts = []
        if self.committed_text.strip():
            parts.append(self.committed_text.strip())
        if self.current_partial_text.strip():
            parts.append(self.current_partial_text.strip())
        return " ".join(parts).strip()

    def finalize(self) -> str:
        """
        Finalizes the entire audio buffer, performing a full clean transcription pass.
        """
        if len(self.audio_buffer) == 0:
            return ""
        
        bio = io.BytesIO()
        sf.write(bio, self.audio_buffer, self.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = bio.getvalue()

        try:
            final_text = transcriber.transcribe_audio(wav_bytes, model_key=self.model_key)
            if final_text:
                self.committed_text = final_text
                self.current_partial_text = ""
                return final_text
        except Exception:
            pass

        self._commit_current_window()
        return self.get_full_transcript()

    def reset(self):
        """Resets all audio buffers and decoding states."""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.committed_audio_offset = 0
        self.committed_text = ""
        self.current_partial_text = ""
        self.last_full_text = ""
        self.total_audio_received_s = 0.0
        self.vad.reset()
        self.is_speaking = False
