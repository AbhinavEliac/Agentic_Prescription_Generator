"""
app/stt/vad.py
--------------
Sub-10ms standalone Voice Activity Detection (VAD) Engine.
Optimized for real-time clinical speech dictation streams with zero external heavy model dependencies.
Combines adaptive energy thresholding, zero-crossing rate (ZCR), and hangover smoothing.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


class VADDetector:
    """
    Sub-10ms Voice Activity Detector with adaptive noise floor and hangover smoothing.
    Pure NumPy implementation for instant cross-platform execution.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.012,
        min_speech_duration_ms: float = 180.0,
        min_silence_duration_ms: float = 400.0,
        hangover_frames: int = 3,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold

        self.frame_size = int(sample_rate * 0.03)  # 30ms frames
        self.min_speech_frames = max(1, int(min_speech_duration_ms / 30.0))
        self.min_silence_frames = max(1, int(min_silence_duration_ms / 30.0))
        self.hangover_frames = hangover_frames

        # Adaptive background noise tracking
        self.noise_floor = 0.004
        self.adaptation_rate = 0.05

        # State tracking
        self.speech_counter = 0
        self.silence_counter = 0
        self.is_currently_speech = False
        self.hangover_remaining = 0

    def compute_energy_and_zcr(self, frame: np.ndarray) -> Tuple[float, float]:
        """Computes Root Mean Square (RMS) energy and Zero-Crossing Rate (ZCR)."""
        if len(frame) == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

        # Zero crossing rate
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr = float(np.mean(np.abs(signs[1:] - signs[:-1])) / 2.0) if len(frame) > 1 else 0.0
        return rms, zcr

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """Determines if a short audio frame contains active human speech."""
        if len(frame) < 64:
            return self.is_currently_speech

        rms, zcr = self.compute_energy_and_zcr(frame)

        # Update noise floor during quiet moments
        if rms < self.energy_threshold * 0.8:
            self.noise_floor = (1.0 - self.adaptation_rate) * self.noise_floor + self.adaptation_rate * rms

        dynamic_thresh = max(self.energy_threshold, self.noise_floor * 2.8)
        raw_speech = rms > dynamic_thresh and zcr < 0.45

        if raw_speech:
            self.speech_counter += 1
            self.silence_counter = 0
            if self.speech_counter >= self.min_speech_frames:
                self.is_currently_speech = True
                self.hangover_remaining = self.hangover_frames
        else:
            self.silence_counter += 1
            self.speech_counter = 0
            if self.silence_counter >= self.min_silence_frames:
                if self.hangover_remaining > 0:
                    self.hangover_remaining -= 1
                else:
                    self.is_currently_speech = False

        return self.is_currently_speech or (self.hangover_remaining > 0)

    def process_samples(self, samples: np.ndarray) -> bool:
        """
        Processes an arbitrary-length audio buffer by breaking it into standard 30ms frames.
        Returns whether speech is active at the end of the buffer.
        """
        if len(samples) == 0:
            return self.is_currently_speech

        step = self.frame_size
        n_frames = len(samples) // step
        if n_frames == 0:
            return self.is_speech_frame(samples)

        is_speech = False
        for i in range(n_frames):
            frame = samples[i * step : (i + 1) * step]
            is_speech = self.is_speech_frame(frame)

        # Handle remainder if any
        remainder = samples[n_frames * step :]
        if len(remainder) >= 64:
            is_speech = self.is_speech_frame(remainder)

        return is_speech

    def filter_speech(self, audio: np.ndarray) -> np.ndarray:
        """
        Strips leading and trailing silence from continuous audio without mutating
        the detector's active streaming state.
        """
        if len(audio) == 0:
            return audio

        step = self.frame_size
        n_frames = len(audio) // step
        if n_frames == 0:
            return audio

        # Use an isolated detector instance to avoid polluting live streaming VAD state
        isolated_vad = VADDetector(
            sample_rate=self.sample_rate,
            energy_threshold=self.energy_threshold,
            min_speech_duration_ms=60.0,
            min_silence_duration_ms=250.0,
            hangover_frames=self.hangover_frames,
        )
        isolated_vad.noise_floor = self.noise_floor

        speech_flags = [isolated_vad.is_speech_frame(audio[i * step : (i + 1) * step]) for i in range(n_frames)]

        speech_indices = [i for i, f in enumerate(speech_flags) if f]
        if not speech_indices:
            # No verified speech detected: return empty array so silence is not decoded
            return np.array([], dtype=np.float32)

        first_idx = speech_indices[0]
        last_idx = speech_indices[-1]

        # Add 120ms safety pad before and after speech
        pad = step * 4
        start_sample = max(0, first_idx * step - pad)
        end_sample = min(len(audio), (last_idx + 1) * step + pad)

        return audio[start_sample:end_sample]

    def has_speech(self, audio: np.ndarray) -> bool:
        """Determines if the audio buffer contains any verified speech segments."""
        if len(audio) == 0:
            return False
        step = self.frame_size
        n_frames = len(audio) // step
        if n_frames == 0:
            return False
        isolated_vad = VADDetector(
            sample_rate=self.sample_rate,
            energy_threshold=self.energy_threshold,
            min_speech_duration_ms=60.0,
            min_silence_duration_ms=250.0,
            hangover_frames=self.hangover_frames,
        )
        isolated_vad.noise_floor = self.noise_floor
        return any(isolated_vad.is_speech_frame(audio[i * step : (i + 1) * step]) for i in range(n_frames))

    def reset(self) -> None:
        """Resets VAD tracking state."""
        self.speech_counter = 0
        self.silence_counter = 0
        self.is_currently_speech = False
        self.hangover_remaining = 0

