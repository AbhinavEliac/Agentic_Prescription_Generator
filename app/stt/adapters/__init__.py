"""
app.stt.adapters package
------------------------
Concrete adapters for Speech-to-Text inference engines.
"""

from app.stt.adapters.mock_engine import MockSTTEngine
from app.stt.adapters.ctranslate2_engine import CTranslate2Engine
from app.stt.adapters.whisper_engine import OpenAIWhisperEngine
from app.stt.adapters.transformers_engine import TransformersEngine
from app.stt.adapters.moonshine_engine import MoonshineEngine

__all__ = [
    "MockSTTEngine",
    "CTranslate2Engine",
    "OpenAIWhisperEngine",
    "TransformersEngine",
    "MoonshineEngine",
]
