"""
app/stt/schemas.py
------------------
Canonical data contracts and schemas for the Speech-to-Text (STT) subsystem.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict


class TranscriptionSegment(BaseModel):
    """Timestamped segment of transcribed speech."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None


class STTFallbackRecord(BaseModel):
    """Audit record capturing an explicit model fallback event."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    primary_model: str
    fallback_model: str
    reason: str


class TranscriptionResult(BaseModel):
    """Standardized output schema for all Speech-to-Text inference engines."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    text: str
    language: Optional[str] = "en"
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    timestamps: Optional[List[Tuple[float, float]]] = None
    confidence: Optional[float] = None
    model: str
    latency: float = 0.0
    error_metadata: Optional[Dict[str, Any]] = None
    fallback_info: Optional[STTFallbackRecord] = None

    @property
    def duration_s(self) -> float:
        """Calculated audio duration in seconds from segments or timestamps."""
        if self.segments:
            ends = [s.end for s in self.segments if s.end is not None]
            starts = [s.start for s in self.segments if s.start is not None]
            if ends and starts:
                return round(max(ends) - min(starts), 2)
            if ends:
                return round(max(ends), 2)
        if self.timestamps:
            return round(max(t[1] for t in self.timestamps) - min(t[0] for t in self.timestamps), 2)
        return 0.0

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return round(self.latency * 1000.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to a clean dictionary for API and UI consumers."""
        return {
            "text": self.text,
            "language": self.language,
            "segments": [s.model_dump() for s in self.segments],
            "timestamps": self.timestamps,
            "confidence": round(self.confidence, 3) if self.confidence is not None else None,
            "model": self.model,
            "latency": round(self.latency, 3),
            "latency_ms": self.latency_ms,
            "duration_s": self.duration_s,
            "error_metadata": self.error_metadata,
            "fallback_info": self.fallback_info.model_dump() if self.fallback_info else None,
        }


class STTConfig(BaseModel):
    """Configuration options for STT engines and managers."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    default_model: str = "whisper_ayush"
    device: str = "auto"  # auto, cuda, cpu
    compute_type: str = "auto"  # auto, float16, int8
    fallback_chain: List[str] = Field(default_factory=lambda: ["whisper_ayush", "whisper_base", "mock"])
    enable_fallback: bool = True
    vad_energy_threshold: float = 0.012
    sample_rate: int = 16000
