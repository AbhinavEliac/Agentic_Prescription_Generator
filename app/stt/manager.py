"""
app/stt/manager.py
------------------
Central STT Subsystem Coordinator and Model Lifecycle Manager.

Enforces:
1. Lazy-loading: zero models are loaded at application startup.
2. Configuration-driven engine selection.
3. Separation of model loading from transcription.
4. Explicit fallback auditing (never silently switches models).
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Type, Union, Any
from pathlib import Path
import numpy as np

from app.stt.base import (
    STTEngine,
    STTError,
    ModelNotFoundError,
    ModelLoadError,
    TranscriptionError,
)
from app.stt.schemas import (
    TranscriptionResult,
    STTFallbackRecord,
    STTConfig,
)
from app.stt.adapters import (
    MockSTTEngine,
    CTranslate2Engine,
    OpenAIWhisperEngine,
    TransformersEngine,
    MoonshineEngine,
)

logger = logging.getLogger("stt_manager")


class STTManager:
    """
    Singleton coordinator managing the lifecycle, registration, lazy-loading,
    and explicit fallback of Speech-to-Text inference engines.
    """

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._active_engines: Dict[str, STTEngine] = {}
        self._engine_factories: Dict[str, Any] = {}
        self._register_default_factories()

    def _register_default_factories(self) -> None:
        """Registers factory functions for all supported STT model architectures."""
        self._engine_factories["whisper_ayush"] = lambda: CTranslate2Engine(
            name="whisper_ayush",
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self._engine_factories["whisper_large_turbo"] = lambda: TransformersEngine(
            name="whisper_large_turbo",
            hf_model_id="openai/whisper-large-v3-turbo",
            device=self.config.device,
        )
        self._engine_factories["moonshine_base"] = lambda: MoonshineEngine(
            name="moonshine_base",
            model_size="base",
            device=self.config.device,
        )
        self._engine_factories["moonshine_tiny"] = lambda: MoonshineEngine(
            name="moonshine_tiny",
            model_size="tiny",
            device=self.config.device,
        )
        self._engine_factories["canary_1b"] = lambda: TransformersEngine(
            name="canary_1b",
            hf_model_id="nvidia/canary-1b",
            device=self.config.device,
        )
        self._engine_factories["parakeet_tdt"] = lambda: TransformersEngine(
            name="parakeet_tdt",
            hf_model_id="nvidia/parakeet-tdt-1.1b",
            device=self.config.device,
        )
        self._engine_factories["whisper_base"] = lambda: OpenAIWhisperEngine(
            name="whisper_base",
            model_size="base",
            device=self.config.device if self.config.device != "auto" else "cpu",
        )
        self._engine_factories["whisper_tiny"] = lambda: OpenAIWhisperEngine(
            name="whisper_tiny",
            model_size="tiny",
            device=self.config.device if self.config.device != "auto" else "cpu",
        )
        self._engine_factories["mock"] = lambda: MockSTTEngine(name="mock")

    def register_engine(self, model_key: str, engine_factory: Any) -> None:
        """Registers a custom engine factory for a model key."""
        self._engine_factories[model_key] = engine_factory

    def get_engine(self, model_key: Optional[str] = None, auto_load: bool = True) -> STTEngine:
        """
        Retrieves or lazy-instantiates an STT engine for the given model key.
        Does not load heavy model weights until requested.
        """
        key = (model_key or self.config.default_model).lower().strip()

        # Check active cache
        if key in self._active_engines:
            engine = self._active_engines[key]
            if auto_load and not engine.is_loaded:
                engine.load()
            return engine

        # Check factory registry
        if key not in self._engine_factories:
            raise ModelNotFoundError(f"Unrecognized STT model key: '{key}'", model_name=key)

        # Lazy-instantiate
        logger.info(f"[STTManager] Lazy-instantiating engine for '{key}'...")
        engine = self._engine_factories[key]()
        self._active_engines[key] = engine

        if auto_load and not engine.is_loaded:
            engine.load()

        return engine

    def transcribe(
        self,
        audio: Union[bytes, str, Path, np.ndarray, Any],
        model_key: Optional[str] = None,
        allow_fallback: Optional[bool] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Executes speech transcription with explicit fallback tracking.
        If the primary engine fails, attempts configured fallback chain and logs the event.
        Never silently switches models.
        """
        target_model = (model_key or self.config.default_model).lower().strip()
        should_fallback = self.config.enable_fallback if allow_fallback is None else allow_fallback

        primary_error: Optional[Exception] = None

        # 1. Attempt primary engine
        try:
            primary_engine = self.get_engine(target_model, auto_load=True)
            return primary_engine.transcribe(audio, **kwargs)
        except Exception as err:
            primary_error = err
            logger.warning(f"[STTManager] Primary engine '{target_model}' failed: {err}")
            if not should_fallback:
                if isinstance(err, STTError):
                    raise err
                raise TranscriptionError(f"Transcription failed: {str(err)}", model_name=target_model, details=err)

        # 2. Execute explicit fallback chain
        fallback_candidates = [m for m in self.config.fallback_chain if m != target_model]
        for fb_key in fallback_candidates:
            if fb_key not in self._engine_factories:
                continue
            try:
                logger.info(f"[STTManager] Attempting fallback to '{fb_key}'...")
                fb_engine = self.get_engine(fb_key, auto_load=True)
                result = fb_engine.transcribe(audio, **kwargs)

                # Explicitly record the fallback audit metadata
                result.fallback_info = STTFallbackRecord(
                    primary_model=target_model,
                    fallback_model=fb_key,
                    reason=str(primary_error),
                )
                logger.warning(
                    f"[STTManager] Handled fallback: requested='{target_model}' -> used='{fb_key}'. Reason: {primary_error}"
                )
                return result
            except Exception as fb_err:
                logger.warning(f"[STTManager] Fallback candidate '{fb_key}' failed: {fb_err}")

        # If all candidates fail, raise standardized error
        raise TranscriptionError(
            f"Primary model '{target_model}' and all fallback engines failed. Original error: {str(primary_error)}",
            model_name=target_model,
            details=primary_error,
        )

    def unload_all(self) -> None:
        """Frees all active loaded engines from RAM and VRAM."""
        for name, engine in self._active_engines.items():
            try:
                engine.unload()
            except Exception:
                pass
        self._active_engines.clear()

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists all registered models with their current load and device status."""
        return [
            {
                "key": key,
                "is_loaded": self._active_engines[key].is_loaded if key in self._active_engines else False,
                "is_default": key == self.config.default_model,
            }
            for key in self._engine_factories
        ]


# Singleton global manager instance
_GLOBAL_STT_MANAGER: Optional[STTManager] = None


def get_stt_manager() -> STTManager:
    """Returns the singleton STTManager instance."""
    global _GLOBAL_STT_MANAGER
    if _GLOBAL_STT_MANAGER is None:
        _GLOBAL_STT_MANAGER = STTManager()
    return _GLOBAL_STT_MANAGER
