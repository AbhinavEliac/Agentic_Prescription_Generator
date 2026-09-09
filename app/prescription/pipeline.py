"""
app/prescription/pipeline.py
----------------------------
Canonical Prescription Extraction Pipeline.

ONE SYSTEM that consolidates all extraction modes into a single coherent architecture:
- Fast Mode: Normalization -> Segmentation -> DeterministicExtraction -> Validation (<15ms)
- Standard Mode: Normalization -> Segmentation -> DeterministicExtraction -> LLMExtraction -> Reconciliation -> Validation
"""

from __future__ import annotations
import time
from enum import Enum
from typing import Optional, List, Dict, Any
from app.drugs.repository import DrugRepository
from app.prescription.schema import CanonicalPrescription
from app.prescription.normalizer import normalize_prescription_text
from app.prescription.segmenter import segment_prescription_clauses
from app.prescription.deterministic.engine import DeterministicEngine
from app.prescription.agentic.adapter import AgenticAdapter
from app.prescription.reconciler import ClinicalReconciler
from app.prescription.validator import ClinicalValidator
from app.prescription.formatter import CanonicalFormatter


class PipelineMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"


class PrescriptionPipeline:
    """Canonical Unified Pipeline for Prescription Extraction."""

    def __init__(
        self,
        drug_repo: Optional[DrugRepository] = None,
        llm: Optional[Any] = None,
    ):
        self.drug_repo = drug_repo or DrugRepository()
        self.deterministic_engine = DeterministicEngine(drug_repo=self.drug_repo)
        self.agentic_adapter = AgenticAdapter(llm=llm)
        self.reconciler = ClinicalReconciler()
        self.validator = ClinicalValidator(drug_repo=self.drug_repo)
        self.formatter = CanonicalFormatter()

    def extract(
        self,
        raw_text: str,
        mode: PipelineMode = PipelineMode.STANDARD,
    ) -> CanonicalPrescription:
        """
        Executes the canonical 7-stage extraction pipeline.
        Returns a strongly-typed CanonicalPrescription.
        """
        t0 = time.perf_counter()

        if not raw_text or not raw_text.strip():
            return self.formatter.format(
                raw_text="",
                normalized_text="",
                validated_items=[],
                execution_time_ms=0.0,
                pipeline_mode=mode.value,
            )

        # 1. Normalization
        normalized_text = normalize_prescription_text(raw_text)

        # 2. Segmentation
        clauses = segment_prescription_clauses(normalized_text)

        # 3. Deterministic Extraction
        det_candidates = self.deterministic_engine.extract_candidates(clauses, full_text=raw_text)

        # 4 & 5. LLM Extraction & Reconciliation
        if mode == PipelineMode.STANDARD:
            llm_candidates = self.agentic_adapter.extract(normalized_text)
            reconciled = self.reconciler.reconcile(
                deterministic_candidates=det_candidates,
                llm_candidates=llm_candidates,
                source_text=raw_text,
            )
        else:
            # FAST mode: deterministic pass-through
            reconciled = self.reconciler.reconcile(
                deterministic_candidates=det_candidates,
                llm_candidates=None,
                source_text=raw_text,
            )

        # 6. Clinical Validation
        validated_pairs = self.validator.validate_items(
            reconciled,
            raw_text=raw_text,
            normalized_text=normalized_text,
        )

        # 7. Formatter & Evidence Binding
        t1 = time.perf_counter()
        execution_time_ms = round((t1 - t0) * 1000, 2)

        return self.formatter.format(
            raw_text=raw_text,
            normalized_text=normalized_text,
            validated_items=validated_pairs,
            execution_time_ms=execution_time_ms,
            pipeline_mode=mode.value,
        )

    def extract_legacy_blocks(
        self,
        raw_text: str,
        mode: PipelineMode = PipelineMode.STANDARD,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method returning legacy dictionary blocks for backwards compatibility
        with existing UI tables, exporters, and client consumers.
        """
        canonical = self.extract(raw_text, mode=mode)
        blocks: List[Dict[str, Any]] = []
        for item in canonical.items:
            blocks.append({
                "Drug_name": item.medicine_name,
                "strength": item.strength or "NONE",
                "dose": item.dose,
                "dose_unit": item.dose_unit,
                "frequency": item.frequency or "NONE",
                "duration": item.duration or "NONE",
                "route": item.route or "oral",
                "instruction": item.instruction or "NONE",
                "additional_instruction": item.additional_instruction or "NONE",
                "available_drugs": item.available_drugs,
                "did_you_mean": item.did_you_mean,
                "did_you_mean_options": item.did_you_mean_options,
                "available_routes": item.available_routes,
            })
        return blocks
