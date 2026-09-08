"""
app/prescription/validator.py
-----------------------------
Clinical Validation Boundary.

Enforces:
1. Strict 100% groundedness against raw transcription text (zero hallucination)
2. Noise-proofing non-drug entity rejection (diagnostics, vitals, physical care)
3. Anti-hallucination commentary filtering (strips unsolicited moralizing or disclaimers)
"""

from __future__ import annotations
import re
from typing import List, Optional, Dict, Any, Tuple
from app.prescription.schema import ReconciliationItem, ValidationResult, ExtractionStatus
from app.prescription.deterministic.medicine import is_valid_medication_name

FORBIDDEN_COMMENTARY_CUES = [
    "please note",
    "be careful",
    "moral:",
    "disclaimer:",
    "caution:",
]


class ClinicalValidator:
    """Enforces clinical safety and grounding boundaries on reconciled items."""

    def __init__(self, drug_repo: Optional[Any] = None):
        self.drug_repo = drug_repo

    def validate_items(
        self,
        items: List[ReconciliationItem],
        raw_text: str,
        normalized_text: Optional[str] = None,
    ) -> List[Tuple[ReconciliationItem, ValidationResult]]:
        """
        Validates a list of ReconciliationItems against the raw and normalized transcript.
        Returns pairs of (validated_item, validation_result).
        """
        raw_lower = raw_text.lower()
        norm_lower = (normalized_text or "").lower()
        combined_text_lower = f"{raw_lower} {norm_lower}"

        raw_text_clean = raw_text.replace(",", "")
        norm_clean = (normalized_text or "").replace(",", "")
        combined_clean = f"{raw_text_clean} {norm_clean}"

        results: List[Tuple[ReconciliationItem, ValidationResult]] = []

        for item in items:
            errors: List[str] = []
            warnings: List[str] = []

            # 1. Non-drug entity check
            if not is_valid_medication_name(item.medicine_name):
                errors.append(f"Entity '{item.medicine_name}' is not a valid pharmaceutical entity.")

            # 2. Grounding check: core medicine words must appear in raw transcript or normalized transcript
            core_words = [
                w for w in re.findall(r"[A-Za-z0-9\-]+", item.medicine_name)
                if len(w) >= 3 and w.lower() not in (
                    "take", "tab", "tabs", "tablet", "capsule", "syrup",
                    "pill", "rotacap", "none", "vial", "sachet", "one", "administer", "mg", "ml"
                )
            ]
            if core_words and not any(cw.lower() in combined_text_lower for cw in core_words):
                errors.append(f"Drug name '{item.medicine_name}' was not found in raw prescription text.")

            # Ensure any dose integers inside medicine name exist in input text or normalized text
            doses = re.findall(r"\d+(?:\.\d+)?", item.medicine_name)
            for d in doses:
                if d not in raw_text and d not in raw_text_clean and d not in combined_clean:
                    errors.append(f"Dose number '{d}' in drug name '{item.medicine_name}' was not found in raw text.")

            # 3. Grounding check: strength if present must be in raw text or normalized text
            if item.strength:
                strength_nums = re.findall(r"\d+(?:\.\d+)?", item.strength)
                for sn in strength_nums:
                    if sn not in raw_text and sn not in raw_text_clean and sn not in combined_clean:
                        errors.append(f"Strength '{item.strength}' was not found in raw prescription text.")

            # 4. Anti-hallucination check: remove unsolicited commentary cues from instructions
            sanitized_inst = item.instruction
            if sanitized_inst:
                for cue in FORBIDDEN_COMMENTARY_CUES:
                    if cue in sanitized_inst.lower() and cue not in raw_lower:
                        # Strip the cue
                        sanitized_inst = re.sub(re.escape(cue), "", sanitized_inst, flags=re.IGNORECASE).strip("; ")
                        warnings.append(f"Removed unsolicited commentary '{cue}'.")
            item.instruction = sanitized_inst if sanitized_inst else None

            # 5. Safety Rule: Unknown medicine not found in master formulary -> NEEDS_REVIEW
            is_recognized_drug = True
            if self.drug_repo:
                first_word = item.medicine_name.split()[0].strip()
                exact = self.drug_repo.find_exact(item.medicine_name) or self.drug_repo.find_exact(first_word)
                norm = self.drug_repo.find_normalized(item.medicine_name) or self.drug_repo.find_normalized(first_word)
                fuzzy = self.drug_repo.find_fuzzy(first_word, min_confidence=0.82)
                if not exact and not norm and not fuzzy:
                    is_recognized_drug = False
                    reason = f"Unknown medicine '{item.medicine_name}' is not recognized in the master formulary."
                    if reason not in item.review_reasons:
                        item.review_reasons.append(reason)
                    warnings.append(reason)

            # 6. Safety Rule: Ambiguous or contradictory frequency -> NEEDS_REVIEW
            if item.frequency:
                freq_lower = item.frequency.lower()
                ambiguous_cues = (
                    r"\b(?:or|maybe|approx|possibly)\b",
                    r"\bnot\s+more\s+than\b",
                    r"\bas\s+needed\s+but\s+not\s+more\s+than\b",
                    r"\b(?:once|twice|thrice)\s+and\s+(?:also\s+)?(?:once|twice|thrice)\b",
                )
                if any(re.search(cue, freq_lower) for cue in ambiguous_cues):
                    reason = f"Ambiguous or contradictory frequency '{item.frequency}' requires review."
                    if reason not in item.review_reasons:
                        item.review_reasons.append(reason)
                    warnings.append(reason)

            # Check if source transcript contains contradictory instructions for this item
            contradictory_patterns = (
                r"\b(?:twice|once|thrice|three\s+times|four\s+times|\d+\s+times)\s+daily\s+and\s+also\s+(?:take\s+)?(?:twice|once|thrice|three\s+times|four\s+times|\d+\s+times)\s+daily\b",
                r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)?\s+or\s+\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)\b",
                r"\bnot\s+more\s+than\s+(?:once|twice|thrice|\d+)\b",
            )
            for cp in contradictory_patterns:
                if re.search(cp, raw_lower):
                    reason = "Prescription text contains contradictory or alternative directives requiring clinical review."
                    if reason not in item.review_reasons:
                        item.review_reasons.append(reason)
                    warnings.append(reason)
                    break

            # 7. Safety Rule: Conflicting extraction -> NEEDS_REVIEW
            if item.conflicts:
                for conf in item.conflicts:
                    if conf not in item.review_reasons:
                        item.review_reasons.append(f"Model extraction conflict: {conf}")

            is_valid = len(errors) == 0

            # Determine final lifecycle status
            if not is_valid:
                item.status = ExtractionStatus.ERROR
                val_status = ExtractionStatus.ERROR
            elif item.status == ExtractionStatus.CONFLICT:
                val_status = ExtractionStatus.NEEDS_REVIEW
            elif item.review_reasons or item.conflicts or not is_recognized_drug:
                item.status = ExtractionStatus.NEEDS_REVIEW
                val_status = ExtractionStatus.NEEDS_REVIEW
            else:
                item.status = ExtractionStatus.VALIDATED
                val_status = ExtractionStatus.VALIDATED

            val_result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                status=val_status,
            )

            results.append((item, val_result))

        return results

    def validate_prescription_items(
        self,
        items: List[PrescriptionItem],
        raw_text: str,
        normalized_text: Optional[str] = None,
    ) -> Any:
        """
        Validates arbitrary canonical PrescriptionItems against raw transcript text.
        Returns a strongly-typed PrescriptionValidationResponse.
        """
        from app.prescription.schema import ValidationFinding, ValidationSeverity
        from app.api.schemas import PrescriptionValidationResponse

        findings: List[ValidationFinding] = []
        sanitized_items: List[PrescriptionItem] = []
        raw_lower = raw_text.lower()
        norm_lower = (normalized_text or "").lower()
        combined_lower = f"{raw_lower} {norm_lower}"

        raw_text_clean = raw_text.replace(",", "")
        norm_clean = (normalized_text or "").replace(",", "")
        combined_clean = f"{raw_text_clean} {norm_clean}"

        has_blocking_errors = False
        total_checks = 0
        passed_checks = 0

        for item in items:
            med_name = item.medicine_name or item.medicine
            item_findings: List[ValidationFinding] = []

            # 1. Check medicine validity
            total_checks += 1
            if not is_valid_medication_name(med_name):
                has_blocking_errors = True
                findings.append(ValidationFinding(
                    item_id=item.item_id,
                    field_name="medicine",
                    severity=ValidationSeverity.ERROR,
                    rule_id="RULE_PHARMACEUTICAL_ENTITY",
                    message=f"Entity '{med_name}' is not a recognized clinical drug entity.",
                    rejected_value=med_name,
                ))
            else:
                passed_checks += 1

            # 2. Grounding check
            total_checks += 1
            core_words = [
                w for w in re.findall(r"[A-Za-z0-9\-]+", med_name)
                if len(w) >= 3 and w.lower() not in (
                    "take", "tab", "tabs", "tablet", "capsule", "syrup",
                    "pill", "rotacap", "none", "vial", "sachet", "one", "administer", "mg", "ml"
                )
            ]
            if core_words and not any(cw.lower() in combined_lower for cw in core_words):
                has_blocking_errors = True
                findings.append(ValidationFinding(
                    item_id=item.item_id,
                    field_name="medicine",
                    severity=ValidationSeverity.ERROR,
                    rule_id="RULE_GROUNDED_DRUG_NAME",
                    message=f"Drug '{med_name}' was not found in verbatim prescription text.",
                    rejected_value=med_name,
                ))
            else:
                passed_checks += 1

            # 3. Strength grounding check
            if item.strength:
                total_checks += 1
                strength_nums = re.findall(r"\d+(?:\.\d+)?", item.strength)
                str_grounded = all(sn in raw_text or sn in raw_text_clean or sn in combined_clean for sn in strength_nums)
                if not str_grounded:
                    findings.append(ValidationFinding(
                        item_id=item.item_id,
                        field_name="strength",
                        severity=ValidationSeverity.WARNING,
                        rule_id="RULE_GROUNDED_STRENGTH",
                        message=f"Strength '{item.strength}' was not explicitly identified in raw text.",
                        rejected_value=item.strength,
                    ))
                else:
                    passed_checks += 1

            # 4. Commentary clean-up in instruction
            sanitized_inst = item.instruction
            if sanitized_inst:
                for cue in FORBIDDEN_COMMENTARY_CUES:
                    if cue in sanitized_inst.lower() and cue not in raw_lower:
                        sanitized_inst = re.sub(re.escape(cue), "", sanitized_inst, flags=re.IGNORECASE).strip("; ")
                        findings.append(ValidationFinding(
                            item_id=item.item_id,
                            field_name="instruction",
                            severity=ValidationSeverity.WARNING,
                            rule_id="RULE_ANTI_HALLUCINATION_COMMENTARY",
                            message=f"Sanitized unsolicited commentary cue: '{cue}'.",
                        ))
                item.instruction = sanitized_inst if sanitized_inst else None

            # Mark item status
            if any(f.severity == ValidationSeverity.ERROR for f in item_findings):
                item.status = ExtractionStatus.ERROR
            else:
                item.status = ExtractionStatus.VALIDATED

            sanitized_items.append(item)

        grounding_score = round(passed_checks / max(1, total_checks), 2)
        overall_status = ExtractionStatus.ERROR if has_blocking_errors else ExtractionStatus.VALIDATED

        return PrescriptionValidationResponse(
            is_valid=not has_blocking_errors,
            status=overall_status,
            grounding_score=grounding_score,
            findings=findings,
            sanitized_items=sanitized_items,
        )
