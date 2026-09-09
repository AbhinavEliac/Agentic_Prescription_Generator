"""
app/prescription/deterministic/engine.py
----------------------------------------
Deterministic Extraction Coordinator Engine.

Coordinates:
- medicine.py
- strength.py
- frequency.py
- duration.py
- route.py
- instruction.py

Produces strongly-typed DeterministicExtractionCandidate items with evidence spans.
"""

from __future__ import annotations
import re
from typing import List, Optional, Dict, Any
from app.drugs.repository import DrugRepository
from app.prescription.schema import DeterministicExtractionCandidate, TextSpan
from app.prescription.segmenter import PrescriptionClause
from app.prescription.deterministic.medicine import extract_medicine_candidate
from app.prescription.deterministic.strength import extract_strength_and_formulation
from app.prescription.deterministic.frequency import extract_frequency
from app.prescription.deterministic.duration import extract_duration
from app.prescription.deterministic.route import extract_route
from app.prescription.deterministic.instruction import extract_instructions


class DeterministicEngine:
    """Orchestrates modular deterministic clinical extraction across segmented clauses."""

    def __init__(self, drug_repo: Optional[DrugRepository] = None):
        self.drug_repo = drug_repo or DrugRepository()

    def extract_candidates(
        self,
        clauses: List[PrescriptionClause],
        full_text: str,
    ) -> List[DeterministicExtractionCandidate]:
        """
        Executes modular extractors across all clauses in a prescription.
        Returns a list of typed DeterministicExtractionCandidate objects.
        """
        candidates: List[DeterministicExtractionCandidate] = []

        for clause in clauses:
            # 1. Extract Base Medicine Name and Grounding
            base_name, med_span, med_conf, did_you_mean, matched_drug_id, did_you_mean_options = extract_medicine_candidate(
                clause.text,
                seed_drug_name=clause.seed_drug_name,
                drug_repo=self.drug_repo,
            )

            # If no grounded medicine found in this clause, check if it contains instructions/duration/frequency
            if not base_name:
                if candidates and (clause.text or clause.advice_clauses):
                    last_cand = candidates[-1]

                    # Trailing duration binding
                    trailing_dur, _, _ = extract_duration(clause.text, full_prescription_text=full_text)
                    if trailing_dur and not last_cand.duration:
                        last_cand.duration = trailing_dur

                    # Trailing frequency conflict detection
                    trailing_freq, _, _ = extract_frequency(clause.text, full_prescription_text=full_text)
                    if trailing_freq and last_cand.frequency and trailing_freq.lower() not in last_cand.frequency.lower():
                        last_cand.confidence = min(last_cand.confidence, 0.60)

                    prim_inst, add_inst, _ = extract_instructions(
                        clause.text,
                        advice_clauses=clause.advice_clauses,
                        full_prescription_text=full_text,
                    )
                    if prim_inst:
                        existing = last_cand.instruction or ""
                        combined = list(dict.fromkeys(filter(None, [existing, prim_inst])))
                        last_cand.instruction = "; ".join(combined)
                    if add_inst:
                        existing_add = last_cand.additional_instruction or ""
                        combined_add = list(dict.fromkeys(filter(None, [existing_add, add_inst])))
                        last_cand.additional_instruction = "; ".join(combined_add)
                continue

            # 2. Extract Strength, Formulation Dose, and Order Dose according to Rule 1 & Rule 2
            full_drug_name, strength, dose, dose_unit, str_span, str_conf, composite_options = extract_strength_and_formulation(
                clause.text,
                base_drug_name=base_name,
                drug_repo=self.drug_repo,
            )
            if composite_options:
                did_you_mean_options = composite_options
                did_you_mean = composite_options[0]["drug_name"]
                if not matched_drug_id:
                    matched_drug_id = composite_options[0]["drug_id"]

            # 3. Extract Frequency & Normalize Timing Code
            freq, freq_span, freq_conf = extract_frequency(
                clause.text,
                full_prescription_text=full_text,
            )

            # 4. Extract Duration
            dur, dur_span, dur_conf = extract_duration(
                clause.text,
                full_prescription_text=full_text,
            )

            # 5. Extract Route & Permissible Routes Catalog
            route, route_span, route_conf, available_routes = extract_route(
                clause.text,
                base_drug_name=base_name,
                drug_repo=self.drug_repo,
            )

            # 6. Extract Primary & Additional Instructions cleanly attributed to this medicine
            prim_inst, add_inst, inst_conf = extract_instructions(
                clause.text,
                advice_clauses=clause.advice_clauses,
                full_prescription_text=full_text,
            )

            # 7. Retrieve same-dose formulations from Drug_database for dropdown UI
            available_drugs: List[Dict[str, Any]] = []
            if self.drug_repo and base_name:
                target_dose = dose or strength or ""
                if not target_dose:
                    dm = re.search(r"\b(\d+(?:\.\d+)?)\b", full_drug_name or clause.text)
                    if dm:
                        target_dose = dm.group(1)
                available_drugs = self.drug_repo.find_same_dose_formulations(
                    drug_name_or_id=matched_drug_id or base_name,
                    dose=target_dose,
                )

            # Build evidence spans
            evidence_spans: Dict[str, TextSpan] = {}
            confidence_scores: Dict[str, float] = {
                "medicine": med_conf,
                "strength": str_conf,
                "frequency": freq_conf,
                "duration": dur_conf,
                "route": route_conf,
                "instruction": inst_conf,
            }

            clause_offset = clause.start_char

            if med_span:
                evidence_spans["medicine"] = TextSpan(
                    start=clause_offset + med_span[0],
                    end=clause_offset + med_span[1],
                    text=clause.text[med_span[0]:med_span[1]],
                )
            if str_span:
                evidence_spans["strength"] = TextSpan(
                    start=clause_offset + str_span[0],
                    end=clause_offset + str_span[1],
                    text=clause.text[str_span[0]:str_span[1]],
                )
            if freq_span:
                evidence_spans["frequency"] = TextSpan(
                    start=clause_offset + freq_span[0],
                    end=clause_offset + freq_span[1],
                    text=clause.text[freq_span[0]:freq_span[1]],
                )
            if dur_span:
                evidence_spans["duration"] = TextSpan(
                    start=clause_offset + dur_span[0],
                    end=clause_offset + dur_span[1],
                    text=clause.text[dur_span[0]:dur_span[1]],
                )

            cand = DeterministicExtractionCandidate(
                medicine_name=full_drug_name,
                matched_drug_id=matched_drug_id,
                strength=strength,
                dose=dose,
                dose_unit=dose_unit,
                frequency=freq,
                duration=dur,
                route=route,
                instruction=prim_inst,
                additional_instruction=add_inst,
                available_drugs=available_drugs,
                did_you_mean=did_you_mean,
                did_you_mean_options=did_you_mean_options,
                available_routes=available_routes,
                evidence_spans=evidence_spans,
                confidence_scores=confidence_scores,
            )
            candidates.append(cand)

        # Deduplicate identical repeats (same drug and same dose and frequency and duration)
        deduped: List[DeterministicExtractionCandidate] = []
        for c in candidates:
            c_base = c.medicine_name.split()[0].upper()
            existing = next(
                (
                    d for d in deduped
                    if d.medicine_name.split()[0].upper() == c_base
                    and d.strength == c.strength
                    and d.frequency == c.frequency
                    and d.duration == c.duration
                ),
                None,
            )
            if not existing:
                deduped.append(c)
            else:
                # Merge distinct instructions
                if c.instruction and existing.instruction != c.instruction:
                    existing.instruction = f"{existing.instruction or ''}; {c.instruction}".strip("; ")
                if c.additional_instruction and existing.additional_instruction != c.additional_instruction:
                    existing.additional_instruction = f"{existing.additional_instruction or ''}; {c.additional_instruction}".strip("; ")

        return deduped
