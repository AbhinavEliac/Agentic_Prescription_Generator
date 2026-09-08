"""
app/stt/adapters/moonshine_engine.py
------------------------------------
Adapter for Useful Sensors Moonshine edge ASR models (moonshine-base, moonshine-tiny).
"""

from __future__ import annotations
from app.stt.adapters.transformers_engine import TransformersEngine


class MoonshineEngine(TransformersEngine):
    """
    Adapter for Useful Sensors Moonshine edge models.
    """

    def __init__(
        self,
        name: str = "moonshine_base",
        model_size: str = "base",
        device: str = "auto",
    ):
        hf_id = "usefulsensors/moonshine-base" if model_size == "base" else "usefulsensors/moonshine-tiny"
        super().__init__(
            name=name,
            hf_model_id=hf_id,
            device=device,
            trust_remote_code=True,
        )
