"""
app/prescription/agentic/adapter.py
-----------------------------------
LangGraph Agentic Extraction Adapter.

Executes the LangGraph multi-agent pipeline when LLM mode is active,
normalizes agent outputs into typed LLMExtractionCandidate models,
and strictly enforces that missing data remains None (never fabricated).
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.prescription.schema import LLMExtractionCandidate, TextSpan

logger = logging.getLogger(__name__)


class AgenticAdapter:
    """Adapts LangGraph multi-agent graph output to canonical LLMExtractionCandidate intermediate models."""

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def extract(self, text: str) -> List[LLMExtractionCandidate]:
        """
        Runs LangGraph multi-agent extraction pipeline and converts blocks to LLMExtractionCandidate.
        Returns an empty list if graph pipeline is unavailable or no valid candidates found.
        """
        # Ensure rx_extractor_app is importable
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        rx_app_path = repo_root / "rx_extractor_app"
        if str(rx_app_path) not in sys.path:
            sys.path.insert(0, str(rx_app_path))

        try:
            from rx_extractor_app.graph_pipeline import run_graph_extraction
            output, gen_time, agent_logs, blocks = run_graph_extraction(self.llm, text)
        except Exception as e:
            logger.warning(f"[AgenticAdapter] LangGraph multi-agent execution failed ({e}); executing fallback to empty candidate set")
            return []

        candidates: List[LLMExtractionCandidate] = []

        for idx, b in enumerate(blocks):
            raw_med = (b.get("Drug_name") or "").strip()
            if not raw_med or raw_med.upper() in ("NONE", "NULL", "N/A"):
                continue

            raw_strength = (b.get("strength") or "").strip()
            strength = None if raw_strength.upper() in ("NONE", "NULL", "", "N/A") else raw_strength

            raw_freq = (b.get("frequency") or "").strip()
            frequency = None if raw_freq.upper() in ("NONE", "NULL", "", "N/A") else raw_freq

            raw_dur = (b.get("duration") or "").strip()
            duration = None if raw_dur.upper() in ("NONE", "NULL", "", "N/A") else raw_dur

            raw_route = (b.get("route") or "").strip()
            route = None if raw_route.upper() in ("NONE", "NULL", "", "N/A") else raw_route.lower()

            raw_inst = (b.get("instruction") or "").strip()
            instruction = None if raw_inst.upper() in ("NONE", "NULL", "", "N/A") else raw_inst

            raw_add = (b.get("additional_instruction") or "").strip()
            additional = None if raw_add.upper() in ("NONE", "NULL", "", "N/A") else raw_add

            # Locate basic spans in source text if present
            evidence_spans: Dict[str, TextSpan] = {}
            if raw_med and raw_med.lower() in text.lower():
                idx_med = text.lower().find(raw_med.lower())
                evidence_spans["medicine"] = TextSpan(start=idx_med, end=idx_med + len(raw_med), text=text[idx_med:idx_med + len(raw_med)])
            if strength and strength.lower() in text.lower():
                idx_s = text.lower().find(strength.lower())
                evidence_spans["strength"] = TextSpan(start=idx_s, end=idx_s + len(strength), text=text[idx_s:idx_s + len(strength)])
            if frequency and frequency.lower() in text.lower():
                idx_f = text.lower().find(frequency.lower())
                evidence_spans["frequency"] = TextSpan(start=idx_f, end=idx_f + len(frequency), text=text[idx_f:idx_f + len(frequency)])
            if duration and duration.lower() in text.lower():
                idx_d = text.lower().find(duration.lower())
                evidence_spans["duration"] = TextSpan(start=idx_d, end=idx_d + len(duration), text=text[idx_d:idx_d + len(duration)])

            candidates.append(
                LLMExtractionCandidate(
                    medicine_name=raw_med,
                    strength=strength,
                    frequency=frequency,
                    duration=duration,
                    route=route,
                    instruction=instruction,
                    additional_instruction=additional,
                    evidence_spans=evidence_spans,
                    confidence_scores={
                        "medicine": 0.85,
                        "strength": 0.85,
                        "frequency": 0.85,
                        "duration": 0.85,
                        "route": 0.85,
                        "instruction": 0.85,
                    },
                    reasoning=f"Extracted by LangGraph Multi-Agent pipeline in {gen_time:.3f}s",
                )
            )

        return candidates
