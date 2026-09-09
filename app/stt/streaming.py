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
    Coordinator for real-time audio streams (Web Audio API, WebSocket).
    Uses cumulative utterance tracking so earlier words (such as medication names)
    are never discarded mid-sentence.
    """

    def __init__(
        self,
        engine: STTEngine,
        sample_rate: int = 16000,
        window_duration_s: float = 12.0,  # Max utterance span before boundary commit
        step_duration_s: float = 0.6,    # Interval between partial decodes
        vad_energy_threshold: float = 0.008,
        on_partial_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.engine = engine
        self.sample_rate = sample_rate
        self.max_utterance_samples = int(window_duration_s * sample_rate)
        self.step_samples = int(step_duration_s * sample_rate)
        self.vad = VADDetector(
            sample_rate=sample_rate,
            energy_threshold=vad_energy_threshold,
            min_speech_duration_ms=90.0,
            min_silence_duration_ms=300.0,
            hangover_frames=3,
        )
        self.on_partial_callback = on_partial_callback

        # Cumulative audio buffers
        self.audio_buffer = np.array([], dtype=np.float32)            # Complete session audio
        self.current_utterance = np.array([], dtype=np.float32)       # Active uncommitted sentence
        self.confirmed_sentences: List[str] = []                      # Solidified previous sentences

        # Pre-roll onset buffer (200ms) to capture initial syllables without clipping
        self.pre_roll_buffer = np.array([], dtype=np.float32)
        self.pre_roll_max_samples = int(0.2 * sample_rate)
        self.hangover_max_samples = int(0.2 * sample_rate)            # Trailing margin
        self.silence_boundary_samples = int(0.35 * sample_rate)       # 350ms pause commits sentence
        
        self.interim_text: str = ""
        self.confirmed_text: str = ""
        self.samples_since_last_decode: int = 0
        self.silence_samples: int = 0
        self.last_speech_time: float = time.time()
        self.is_speaking: bool = False
        self.was_speaking: bool = False
        self.is_active: bool = False

    def feed_chunk(self, chunk: Union[bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Processes an incoming chunk of audio samples (PCM16 bytes or float32 array).
        Breaks chunks into 30ms frames for sub-millisecond accurate VAD.
        Returns current state including VAD status and cumulative transcription.
        """
        if isinstance(chunk, (bytes, bytearray)):
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            pcm = chunk.astype(np.float32)

        dur = round(len(self.audio_buffer) / self.sample_rate, 2)
        device_tag = getattr(self.engine, "device", "cpu")

        if len(pcm) == 0:
            full_display = self._get_full_display_text()
            return {
                "type": "partial",
                "is_speech": False,
                "interim_text": self.interim_text,
                "text": full_display,
                "full_transcript": full_display,
                "final_text": self.confirmed_text,
                "decoded": False,
                "duration": dur,
                "latency_ms": 0.0,
                "device": device_tag,
            }

        # Track session audio
        self.audio_buffer = np.concatenate([self.audio_buffer, pcm])
        dur = round(len(self.audio_buffer) / self.sample_rate, 2)

        # Sub-chunk 30ms frame-accurate VAD processing
        is_speech = self.vad.process_samples(pcm)

        decoded = False
        latency_ms = 0.0
        boundary_detected = False

        if is_speech:
            self.last_speech_time = time.time()
            if not self.is_speaking:
                # Speech started: prepend pre-roll buffer to prevent onset clipping
                self.is_speaking = True
                if len(self.pre_roll_buffer) > 0:
                    self.current_utterance = np.concatenate([self.pre_roll_buffer, pcm])
                    self.pre_roll_buffer = np.array([], dtype=np.float32)
                else:
                    self.current_utterance = pcm
            else:
                self.current_utterance = np.concatenate([self.current_utterance, pcm])

            self.samples_since_last_decode += len(pcm)

            # Periodic interim decode of current active speech (requires >= 0.8s to avoid sub-syllable hallucinations)
            if self.samples_since_last_decode >= self.step_samples:
                if len(self.current_utterance) >= int(self.sample_rate * 0.8):
                    window = self.current_utterance[-self.max_utterance_samples :]
                    t0 = time.perf_counter()
                    try:
                        res = self.engine.transcribe(window)
                        candidate = res.text.strip()
                        if candidate and any(c.isalnum() for c in candidate):
                            self.interim_text = candidate
                            decoded = True
                            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                    except Exception as ex:
                        import logging
                        logging.getLogger("StreamingTranscriber").warning(f"Inference error: {ex}")
                self.samples_since_last_decode = 0

        else:
            # Silence / no active speech detected by VAD
            if self.is_speaking:
                # VAD transitioned to silence (speaker finished sentence and paused):
                # Cleanly filter active utterance to strip leading/trailing silence and verify genuine speech
                filtered_speech = self.vad.filter_speech(self.current_utterance)

                # Require at least 300ms of genuine speech to commit a sentence (rejects breaths, mic taps)
                if len(filtered_speech) >= int(self.sample_rate * 0.3):
                    t0 = time.perf_counter()
                    try:
                        res = self.engine.transcribe(filtered_speech)
                        final_sentence = res.text.strip()
                        if final_sentence and any(c.isalnum() for c in final_sentence):
                            try:
                                from agents.punctuation_agent import correct_sentence_punctuation
                                final_sentence = correct_sentence_punctuation(final_sentence)
                            except Exception:
                                try:
                                    from rx_extractor_app.agents.punctuation_agent import correct_sentence_punctuation
                                    final_sentence = correct_sentence_punctuation(final_sentence)
                                except Exception:
                                    pass
                            self.confirmed_sentences.append(final_sentence)
                            self.confirmed_text = " ".join(self.confirmed_sentences).strip()
                            boundary_detected = True
                            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                    except Exception as ex:
                        import logging
                        logging.getLogger("StreamingTranscriber").warning(f"Boundary decode error: {ex}")

                # Cleanly clear current utterance so idle pause silence is NEVER fed to Whisper
                self.current_utterance = np.array([], dtype=np.float32)
                self.interim_text = ""
                self.is_speaking = False
                self.silence_samples = 0
                self.samples_since_last_decode = 0

            else:
                # Idle background silence: store only short rolling pre-roll buffer; never transcribe
                self.pre_roll_buffer = np.concatenate([self.pre_roll_buffer, pcm])[-self.pre_roll_max_samples :]
                self.silence_samples = 0
                self.samples_since_last_decode = 0

        self.was_speaking = is_speech
        full_display = self._get_full_display_text()

        state = {
            "type": "boundary" if boundary_detected else "partial",
            "is_speech": is_speech,
            "boundary": boundary_detected,
            "interim_text": self.interim_text,
            "sentence": final_sentence if boundary_detected else "",
            "text": full_display,
            "full_transcript": full_display,
            "final_text": self.confirmed_text,
            "decoded": decoded,
            "duration": dur,
            "latency_ms": latency_ms,
            "device": device_tag,
        }

        if (decoded or boundary_detected) and self.on_partial_callback:
            try:
                self.on_partial_callback(state)
            except Exception:
                pass

        return state

    def _get_full_display_text(self) -> str:
        """Assembles cumulative transcript combining solidified sentences and active interim text."""
        parts = [s for s in self.confirmed_sentences if s]
        if self.interim_text:
            parts.append(self.interim_text)
        return " ".join(parts).strip()

    def feed_pcm16(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Convenience method for PCM16 audio frames."""
        return self.feed_chunk(audio_bytes)

    def finalize(self) -> TranscriptionResult:
        """
        Finalizes the streaming session, running full inference on all buffered speech audio.
        """
        self.last_finalize_new_sentence_committed = False
        if len(self.audio_buffer) == 0:
            return TranscriptionResult(text="", model=self.engine.name, latency=0.0)

        # If speech was actively ongoing when finalize was triggered, commit pending utterance cleanly
        if self.is_speaking and len(self.current_utterance) >= int(self.sample_rate * 0.25):
            filtered = self.vad.filter_speech(self.current_utterance)
            if len(filtered) >= int(self.sample_rate * 0.25):
                try:
                    res = self.engine.transcribe(filtered)
                    cand = res.text.strip()
                    if cand and any(c.isalnum() for c in cand):
                        try:
                            from agents.punctuation_agent import correct_sentence_punctuation
                            cand = correct_sentence_punctuation(cand)
                        except Exception:
                            try:
                                from rx_extractor_app.agents.punctuation_agent import correct_sentence_punctuation
                                cand = correct_sentence_punctuation(cand)
                            except Exception:
                                pass
                        self.confirmed_sentences.append(cand)
                        self.confirmed_text = " ".join(self.confirmed_sentences).strip()
                        self.last_finalize_new_sentence_committed = True
                except Exception:
                    pass
            self.current_utterance = np.array([], dtype=np.float32)
            self.interim_text = ""
            self.is_speaking = False

        # If sentences were confirmed during the stream, return the assembled transcript instantly (<1ms)
        if self.confirmed_sentences:
            full_text = " ".join(self.confirmed_sentences).strip()
            self.confirmed_text = full_text
            self.interim_text = ""
            return TranscriptionResult(
                text=full_text,
                model=self.engine.name,
                latency=0.0,
            )

        # Fallback: check if complete session audio buffer has any verified speech
        if not self.vad.has_speech(self.audio_buffer):
            # Pure silence or background noise: return empty transcript without invoking Whisper
            self.confirmed_text = ""
            self.interim_text = ""
            return TranscriptionResult(text="", model=self.engine.name, latency=0.0)

        t0 = time.perf_counter()
        filtered = self.vad.filter_speech(self.audio_buffer)
        if len(filtered) < int(self.sample_rate * 0.15):
            self.confirmed_text = ""
            self.interim_text = ""
            return TranscriptionResult(text="", model=self.engine.name, latency=time.perf_counter() - t0)

        result = self.engine.transcribe(filtered)
        cand = result.text.strip()
        if cand and any(c.isalnum() for c in cand):
            try:
                from agents.punctuation_agent import correct_sentence_punctuation
                cand = correct_sentence_punctuation(cand)
            except Exception:
                try:
                    from rx_extractor_app.agents.punctuation_agent import correct_sentence_punctuation
                    cand = correct_sentence_punctuation(cand)
                except Exception:
                    pass
            self.confirmed_text = cand
            self.last_finalize_new_sentence_committed = True
        else:
            self.confirmed_text = ""
        self.interim_text = ""
        self.confirmed_sentences = [self.confirmed_text] if self.confirmed_text else []
        result.text = self.confirmed_text
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

        return {
            "type": "final",
            "raw_text": res.text,
            "punctuated_text": res.text,
            "text": res.text,
            "new_sentence_committed": getattr(self, "last_finalize_new_sentence_committed", False),
            "duration": dur,
            "final_latency_ms": latency_ms,
            "model_used": self.engine.name,
            "confidence": res.confidence,
        }

    def reset(self) -> None:
        """Resets stream buffer, cumulative utterances, and VAD."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_utterance = np.array([], dtype=np.float32)
        self.pre_roll_buffer = np.array([], dtype=np.float32)
        self.confirmed_sentences = []
        self.confirmed_text = ""
        self.interim_text = ""
        self.samples_since_last_decode = 0
        self.silence_samples = 0
        self.is_speaking = False
        self.was_speaking = False
        self.vad.reset()

