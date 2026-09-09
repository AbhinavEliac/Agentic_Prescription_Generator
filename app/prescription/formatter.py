"""
app/prescription/formatter.py
-----------------------------
Canonical Prescription Formatter.

Converts validated reconciliation items into strongly-typed CanonicalPrescription models,
binding exact FieldEvidence spans and confidence metrics.
"""

from __future__ import annotations
import uuid
from typing import List, Optional, Dict, Any, Tuple
from app.prescription.schema import (
    CanonicalPrescription,
    PrescriptionItem,
    FieldEvidence,
    TextSpan,
    PipelineMetadata,
    ExtractionStatus,
    ReconciliationItem,
    ValidationResult,
)


def _find_span(text: str, sub: Optional[str]) -> Optional[TextSpan]:
    """Finds exact character span for a substring in raw text."""
    if not sub or not text:
        return None
    idx = text.lower().find(sub.lower())
    if idx >= 0:
        return TextSpan(start=idx, end=idx + len(sub), text=text[idx:idx + len(sub)])
    return None


class CanonicalFormatter:
    """Formats validated items into the canonical typed schema with evidence traceability."""

    def format(
        self,
        raw_text: str,
        normalized_text: str,
        validated_items: List[Tuple[ReconciliationItem, ValidationResult]],
        execution_time_ms: float,
        pipeline_mode: str = "standard",
    ) -> CanonicalPrescription:
        """Assembles a CanonicalPrescription from validated items."""
        items: List[PrescriptionItem] = []
        has_conflicts = False
        has_errors = False

        for idx, (rec_item, val_res) in enumerate(validated_items):
            if not val_res.is_valid:
                has_errors = True
                continue

            if rec_item.status == ExtractionStatus.CONFLICT:
                has_conflicts = True

            # Build field-level evidence
            evidence: Dict[str, FieldEvidence] = {}

            # Medicine evidence
            med_span = _find_span(raw_text, rec_item.medicine_name.split()[0])
            evidence["medicine"] = FieldEvidence(
                field_name="medicine_name",
                value=rec_item.medicine_name,
                span=med_span,
                confidence=rec_item.confidence,
                source_model="deterministic" if rec_item.deterministic_candidate else "llm",
            )

            # Strength evidence
            if rec_item.strength:
                str_span = _find_span(raw_text, rec_item.strength)
                evidence["strength"] = FieldEvidence(
                    field_name="strength",
                    value=rec_item.strength,
                    span=str_span,
                    confidence=0.90,
                    source_model="deterministic" if rec_item.deterministic_candidate else "llm",
                )

            # Frequency evidence
            if rec_item.frequency:
                freq_span = _find_span(raw_text, rec_item.frequency)
                evidence["frequency"] = FieldEvidence(
                    field_name="frequency",
                    value=rec_item.frequency,
                    span=freq_span,
                    confidence=0.90,
                    source_model="deterministic",
                )

            # Duration evidence
            if rec_item.duration:
                dur_span = _find_span(raw_text, rec_item.duration)
                evidence["duration"] = FieldEvidence(
                    field_name="duration",
                    value=rec_item.duration,
                    span=dur_span,
                    confidence=0.90,
                    source_model="deterministic",
                )

            # Route evidence
            if rec_item.route:
                route_span = _find_span(raw_text, rec_item.route)
                evidence["route"] = FieldEvidence(
                    field_name="route",
                    value=rec_item.route,
                    span=route_span,
                    confidence=0.85,
                    source_model="deterministic",
                )

            # Instruction evidence
            if rec_item.instruction:
                inst_span = _find_span(raw_text, rec_item.instruction.split("; ")[0])
                evidence["instruction"] = FieldEvidence(
                    field_name="instruction",
                    value=rec_item.instruction,
                    span=inst_span,
                    confidence=0.85,
                    source_model="deterministic",
                )

            # Additional Instruction evidence
            if rec_item.additional_instruction:
                add_span = _find_span(raw_text, rec_item.additional_instruction.split("; ")[0])
                evidence["additional_instruction"] = FieldEvidence(
                    field_name="additional_instruction",
                    value=rec_item.additional_instruction,
                    span=add_span,
                    confidence=0.85,
                    source_model="deterministic",
                )

            p_item = PrescriptionItem(
                item_id=idx + 1,
                medicine_name=rec_item.medicine_name,
                strength=rec_item.strength,
                dose=rec_item.dose,
                dose_unit=rec_item.dose_unit,
                frequency=rec_item.frequency,
                duration=rec_item.duration,
                route=rec_item.route,
                instruction=rec_item.instruction,
                additional_instruction=rec_item.additional_instruction,
                available_drugs=rec_item.available_drugs,
                did_you_mean=rec_item.did_you_mean,
                did_you_mean_options=rec_item.did_you_mean_options,
                available_routes=rec_item.available_routes,
                status=rec_item.status,
                confidence=rec_item.confidence,
                evidence=evidence,
                review_reasons=rec_item.review_reasons,
            )
            items.append(p_item)

        # Overall prescription status
        if has_errors and not items:
            overall_status = ExtractionStatus.ERROR
        elif has_conflicts or any(it.review_reasons for it in items):
            overall_status = ExtractionStatus.NEEDS_REVIEW
        else:
            overall_status = ExtractionStatus.VALIDATED

        metadata = PipelineMetadata(
            pipeline_mode=pipeline_mode,
            execution_time_ms=execution_time_ms,
            model_name="deterministic_engine" if pipeline_mode == "fast" else "hybrid_reconciled",
        )

        return CanonicalPrescription(
            prescription_id=str(uuid.uuid4()),
            raw_text=raw_text,
            normalized_text=normalized_text,
            items=items,
            overall_status=overall_status,
            metadata=metadata,
        )

    def format_plain_text(self, prescription: CanonicalPrescription) -> str:
        """Formats canonical prescription items as standardized key-value blocks."""
        blocks = []
        for it in prescription.items:
            b = [
                f"Drug_name: {it.medicine_name or 'NONE'}",
                f"strength: {it.strength or 'NONE'}",
                f"frequency: {it.frequency or 'NONE'}",
                f"duration: {it.duration or 'NONE'}",
                f"route: {it.route or 'NONE'}",
                f"instruction: {it.instruction or 'NONE'}",
                f"additional_instruction: {it.additional_instruction or 'NONE'}",
            ]
            blocks.append("\n".join(b))
        return "\n\n".join(blocks)
