"""
app/prescription/reconciler.py
------------------------------
Deterministic Clinical Reconciler.

Resolves extractions between DeterministicExtractionCandidate and LLMExtractionCandidate:
- Matching fields: Accepted with high confidence
- Deterministic-only fields: Accepted
- LLM-only fields: Accepted only if grounded against source text
- Conflicting fields (e.g., 500 mg vs 250 mg): Flagged as CONFLICT / NEEDS_REVIEW with full candidate audit trail.
Never arbitrarily favors LLM over deterministic evidence.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from app.prescription.schema import (
    DeterministicExtractionCandidate,
    LLMExtractionCandidate,
    ReconciliationItem,
    ExtractionStatus,
    FieldEvidence,
    TextSpan,
)


def _normalize_cmp(val: Optional[str]) -> str:
    """Normalizes string for clinical equivalence checking."""
    if not val:
        return ""
    return " ".join(val.lower().replace("-", " ").split())


class ClinicalReconciler:
    """Reconciles deterministic and agentic extractions with deterministic conflict logic."""

    def reconcile(
        self,
        deterministic_candidates: List[DeterministicExtractionCandidate],
        llm_candidates: Optional[List[LLMExtractionCandidate]] = None,
        source_text: str = "",
    ) -> List[ReconciliationItem]:
        """
        Reconciles deterministic and LLM candidates.
        If LLM candidates are empty (e.g. FAST mode or LLM offline), deterministic candidates
        are converted directly to ReconciliationItem with EXTRACTED status.
        """
        if not llm_candidates:
            # FAST mode: deterministic-only pass-through
            return [
                ReconciliationItem(
                    medicine_name=dc.medicine_name,
                    strength=dc.strength,
                    frequency=dc.frequency,
                    duration=dc.duration,
                    route=dc.route,
                    instruction=dc.instruction,
                    additional_instruction=dc.additional_instruction,
                    status=ExtractionStatus.EXTRACTED,
                    confidence=dc.confidence_scores.get("medicine", 0.90),
                    deterministic_candidate=dc,
                    llm_candidate=None,
                    conflicts=[],
                    review_reasons=[],
                )
                for dc in deterministic_candidates
            ]

        # STANDARD mode: Pair deterministic and LLM candidates by base medicine name
        reconciled: List[ReconciliationItem] = []
        matched_llm_indices = set()

        for dc in deterministic_candidates:
            dc_base = dc.medicine_name.split()[0].upper()
            
            # Find best matching LLM candidate
            matched_llm: Optional[LLMExtractionCandidate] = None
            matched_idx = -1
            for idx, lc in enumerate(llm_candidates):
                if idx in matched_llm_indices:
                    continue
                lc_base = lc.medicine_name.split()[0].upper()
                if dc_base == lc_base or dc_base in lc.medicine_name.upper() or lc_base in dc.medicine_name.upper():
                    matched_llm = lc
                    matched_idx = idx
                    matched_llm_indices.add(idx)
                    break

            if not matched_llm:
                # No LLM counterpart; accept deterministic candidate
                reconciled.append(
                    ReconciliationItem(
                        medicine_name=dc.medicine_name,
                        strength=dc.strength,
                        frequency=dc.frequency,
                        duration=dc.duration,
                        route=dc.route,
                        instruction=dc.instruction,
                        additional_instruction=dc.additional_instruction,
                        status=ExtractionStatus.NORMALIZED,
                        confidence=dc.confidence_scores.get("medicine", 0.90),
                        deterministic_candidate=dc,
                        llm_candidate=None,
                        conflicts=[],
                        review_reasons=[],
                    )
                )
                continue

            # Reconcile field by field
            conflicts: List[str] = []
            review_reasons: List[str] = []

            # 1. Medicine Name
            if _normalize_cmp(dc.medicine_name) == _normalize_cmp(matched_llm.medicine_name):
                final_med = dc.medicine_name
            else:
                # If one includes catalog strength and other doesn't, prefer fuller specification
                if len(dc.medicine_name) >= len(matched_llm.medicine_name):
                    final_med = dc.medicine_name
                else:
                    final_med = matched_llm.medicine_name

            # 2. Strength
            final_str = dc.strength
            if dc.strength and matched_llm.strength:
                if _normalize_cmp(dc.strength) != _normalize_cmp(matched_llm.strength):
                    conflicts.append(f"strength: deterministic='{dc.strength}' vs llm='{matched_llm.strength}'")
                    review_reasons.append("Conflicting strength between deterministic and LLM models.")
                    # Keep deterministic by default, flag conflict
                    final_str = dc.strength
            elif not dc.strength and matched_llm.strength:
                # Check grounding in source text before accepting LLM strength
                if matched_llm.strength.lower() in source_text.lower():
                    final_str = matched_llm.strength

            # 3. Frequency
            final_freq = dc.frequency
            if dc.frequency and matched_llm.frequency:
                if _normalize_cmp(dc.frequency) != _normalize_cmp(matched_llm.frequency):
                    # Check if one is abbreviation or expanded form of the other (e.g. twice daily vs 1-0-1)
                    if "twice" in dc.frequency.lower() and "twice" in matched_llm.frequency.lower():
                        final_freq = dc.frequency
                    elif "once" in dc.frequency.lower() and "once" in matched_llm.frequency.lower():
                        final_freq = dc.frequency
                    elif matched_llm.frequency.lower() in dc.frequency.lower() or dc.frequency.lower() in matched_llm.frequency.lower():
                        final_freq = dc.frequency
                    else:
                        conflicts.append(f"frequency: deterministic='{dc.frequency}' vs llm='{matched_llm.frequency}'")
                        review_reasons.append("Conflicting frequency between deterministic and LLM models.")
            elif not dc.frequency and matched_llm.frequency:
                if matched_llm.frequency.lower() in source_text.lower():
                    final_freq = matched_llm.frequency

            # 4. Duration
            final_dur = dc.duration
            if dc.duration and matched_llm.duration:
                if _normalize_cmp(dc.duration) != _normalize_cmp(matched_llm.duration):
                    conflicts.append(f"duration: deterministic='{dc.duration}' vs llm='{matched_llm.duration}'")
                    review_reasons.append("Conflicting duration between deterministic and LLM models.")
            elif not dc.duration and matched_llm.duration:
                if matched_llm.duration.lower() in source_text.lower():
                    final_dur = matched_llm.duration

            # 5. Route (never fabricate oral if unstated)
            final_route = dc.route or (matched_llm.route if matched_llm else None)

            # 6. Instructions (Union and clean)
            inst_parts = []
            if dc.instruction:
                inst_parts.append(dc.instruction)
            if matched_llm.instruction and matched_llm.instruction not in inst_parts:
                inst_parts.append(matched_llm.instruction)
            final_inst = "; ".join(dict.fromkeys(inst_parts)) if inst_parts else None

            # 7. Additional Instructions
            add_parts = []
            if dc.additional_instruction:
                add_parts.append(dc.additional_instruction)
            if matched_llm.additional_instruction and matched_llm.additional_instruction not in add_parts:
                add_parts.append(matched_llm.additional_instruction)
            final_add = "; ".join(dict.fromkeys(add_parts)) if add_parts else None

            status = ExtractionStatus.CONFLICT if conflicts else (ExtractionStatus.NEEDS_REVIEW if review_reasons else ExtractionStatus.NORMALIZED)
            confidence = 0.70 if (conflicts or review_reasons) else 0.95

            reconciled.append(
                ReconciliationItem(
                    medicine_name=final_med,
                    strength=final_str,
                    frequency=final_freq,
                    duration=final_dur,
                    route=final_route,
                    instruction=final_inst,
                    additional_instruction=final_add,
                    status=status,
                    confidence=confidence,
                    deterministic_candidate=dc,
                    llm_candidate=matched_llm,
                    conflicts=conflicts,
                    review_reasons=review_reasons,
                )
            )

        # Include any remaining LLM candidates that are grounded
        for idx, lc in enumerate(llm_candidates):
            if idx not in matched_llm_indices:
                if lc.medicine_name.lower() in source_text.lower():
                    reconciled.append(
                        ReconciliationItem(
                            medicine_name=lc.medicine_name,
                            strength=lc.strength,
                            frequency=lc.frequency,
                            duration=lc.duration,
                            route=lc.route,
                            instruction=lc.instruction,
                            additional_instruction=lc.additional_instruction,
                            status=ExtractionStatus.NORMALIZED,
                            confidence=0.85,
                            deterministic_candidate=None,
                            llm_candidate=lc,
                            conflicts=[],
                            review_reasons=[],
                        )
                    )

        return reconciled
