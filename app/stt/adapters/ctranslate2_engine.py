"""
app/stt/adapters/ctranslate2_engine.py
--------------------------------------
High-performance CTranslate2 adapter binding to local Whisper_Ayush_ct2.
Supports CUDA FP16 and multi-threaded CPU INT8 execution.
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path
from typing import Union, Optional, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# Configure CUDA 12 DLL search path for Windows CTranslate2
for _cp in [
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12"),
    r"C:\Users\ADMIN\AppData\Local\Programs\Ollama\lib\ollama\cuda_v12",
    r"C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\lib",
]:
    if os.path.exists(_cp):
        if _cp not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _cp + os.path.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_cp)
            except Exception:
                pass

from app.stt.base import (
    STTEngine,
    prepare_audio_input,
    ModelNotFoundError,
    ModelLoadError,
    TranscriptionError,
)
from app.stt.schemas import TranscriptionResult, TranscriptionSegment

# Medical priming vocabulary
MEDICAL_PROMPT = (
    "Medical prescription dictation. Drug names, dosages, frequencies, routes: "
    "Paracetamol, Ibuprofen, Aspirin, Amoxicillin, Amoxicillin-Clavulanate, Azithromycin, "
    "Cefpodoxime, Cefixime, Cefuroxime, Ciprofloxacin, Levofloxacin, Metronidazole, "
    "Omeprazole, Pantoprazole, Rabeprazole, Esomeprazole, Ranitidine, Domperidone, "
    "Metformin, Glibenclamide, Glipizide, Sitagliptin, Insulin, "
    "Atorvastatin, Rosuvastatin, Amlodipine, Enalapril, Losartan, Telmisartan, "
    "Metoprolol, Atenolol, Furosemide, Spironolactone, "
    "Oxymetazoline, Betamethasone, Prednisolone, Dexamethasone, "
    "Diclofenac, Aceclofenac, Tramadol, Gabapentin, Pregabalin, "
    "mg, mcg, ml, tablet, capsule, syrup, drops, spray, "
    "twice daily, once daily, three times daily, after food, before food, for 5 days."
)


class CTranslate2Engine(STTEngine):
    """
    Adapter for CTranslate2-compiled Whisper models (e.g. Whisper_Ayush_ct2).
    """

    def __init__(
        self,
        name: str = "whisper_ayush",
        model_dir: Optional[Union[str, Path]] = None,
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int = 6,
    ):
        # Resolve device
        resolved_device = device
        if device == "auto":
            try:
                import ctranslate2
                resolved_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                resolved_device = "cpu"

        # Resolve compute type
        resolved_compute = compute_type
        if compute_type == "auto":
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"

        super().__init__(name=name, device=resolved_device, compute_type=resolved_compute)
        self.model_dir = Path(model_dir) if model_dir else self._resolve_model_dir()
        self.cpu_threads = cpu_threads
        self._model = None

    def _resolve_model_dir(self) -> Path:
        """Finds Whisper_Ayush_ct2 directory."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        candidates = [
            repo_root / "Whisper_Ayush_ct2",
            Path("Whisper_Ayush_ct2"),
        ]
        for c in candidates:
            if c.exists() and (c / "model.bin").exists():
                return c
        return candidates[0]

    def load(self) -> None:
        """Loads CTranslate2 model into memory."""
        if self._is_loaded and self._model is not None:
            return

        if not self.model_dir.exists() or not (self.model_dir / "model.bin").exists():
            raise ModelNotFoundError(
                f"CTranslate2 model files not found at {self.model_dir}",
                model_name=self.name,
            )

        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                str(self.model_dir),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            self._is_loaded = True
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load CTranslate2 model: {str(e)}",
                model_name=self.name,
                details=e,
            )

    def unload(self) -> None:
        """Frees model from RAM/VRAM."""
        self._model = None
        self._is_loaded = False

    def transcribe(
        self,
        audio: Union[bytes, str, Path, np.ndarray, Any],
        initial_prompt: Optional[str] = None,
        language: str = "en",
        beam_size: int = 1,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribes audio using CTranslate2."""
        if not self._is_loaded or self._model is None:
            self.load()

        pcm_data, file_path, temp_path = prepare_audio_input(audio)
        input_target = pcm_data if pcm_data is not None else file_path

        if input_target is None:
            raise TranscriptionError("No valid audio data could be extracted.", model_name=self.name)

        t0 = time.perf_counter()
        try:
            prompt = initial_prompt or MEDICAL_PROMPT
            segments_gen, info = self._model.transcribe(
                input_target,
                beam_size=beam_size,
                best_of=1,
                temperature=0.0,
                language=language,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                vad_filter=kwargs.get("vad_filter", True),
            )

            segments: List[TranscriptionSegment] = []
            text_parts = []
            for s in segments_gen:
                text_parts.append(s.text)
                segments.append(
                    TranscriptionSegment(
                        text=s.text.strip(),
                        start=round(s.start, 2),
                        end=round(s.end, 2),
                        confidence=round(getattr(s, "avg_logprob", 0.0), 3),
                    )
                )

            full_text = " ".join(text_parts).strip()
            latency = time.perf_counter() - t0

            return TranscriptionResult(
                text=full_text,
                language=info.language if info else language,
                segments=segments,
                timestamps=[(s.start or 0.0, s.end or 0.0) for s in segments],
                confidence=round(np.exp(info.avg_logprob), 3) if hasattr(info, "avg_logprob") else 0.95,
                model=self.name,
                latency=latency,
            )
        except Exception as e:
            if "cublas" in str(e).lower() and self.device == "cuda":
                try:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(
                        str(self.model_dir),
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=self.cpu_threads,
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    return self.transcribe(audio, initial_prompt=initial_prompt, language=language, beam_size=beam_size, **kwargs)
                except Exception as cpu_err:
                    logger.warning(f"[CTranslate2Engine] Fallback to CPU execution also failed: {cpu_err}")
            raise TranscriptionError(f"Inference failure: {str(e)}", model_name=self.name, details=e)
        finally:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception as unlink_err:
                    logger.debug(f"[CTranslate2Engine] Failed to delete temp audio file {temp_path}: {unlink_err}")

