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
        min_speech_duration_ms: float = 120.0,
        min_silence_duration_ms: float = 350.0,
        hangover_frames: int = 4,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.frame_size = int(sample_rate * 0.03)  # 30ms frames
        self.min_speech_frames = max(1, int(min_speech_duration_ms / 30.0))
        self.min_silence_frames = max(1, int(min_silence_duration_ms / 30.0))
        self.hangover_frames = hangover_frames

        # Adaptive background noise tracking
        self.noise_floor = 0.003
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

        dynamic_thresh = max(self.energy_threshold, self.noise_floor * 2.5)
        raw_speech = rms > dynamic_thresh and zcr < 0.60

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

    def filter_speech(self, audio: np.ndarray) -> np.ndarray:
        """Strips leading and trailing silence from continuous audio."""
        if len(audio) == 0:
            return audio

        step = self.frame_size
        n_frames = len(audio) // step
        if n_frames == 0:
            return audio

        speech_flags = [self.is_speech_frame(audio[i * step : (i + 1) * step]) for i in range(n_frames)]

        # Find first and last speech frame
        first_idx = next((i for i, f in enumerate(speech_flags) if f), 0)
        last_idx = next((i for i in range(n_frames - 1, -1, -1) if speech_flags[i]), n_frames - 1)

        start_sample = max(0, first_idx * step - step * 2)
        end_sample = min(len(audio), (last_idx + 1) * step + step * 2)

        return audio[start_sample:end_sample]

    def reset(self) -> None:
        """Resets VAD tracking state."""
        self.speech_counter = 0
        self.silence_counter = 0
        self.is_currently_speech = False
        self.hangover_remaining = 0
