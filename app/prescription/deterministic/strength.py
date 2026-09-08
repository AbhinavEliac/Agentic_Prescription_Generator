"""
app/prescription/deterministic/strength.py
------------------------------------------
Deterministic Strength & Formulation Dose Extractor.

Implements:
1. Clinical Dosage Rule 1 (Dual-dose splitting: medicine + formulation dose + secondary order strength)
2. Clinical Dosage Rule 2 (Catalog binding: single dose matching catalog binds to drug name, order dose is None)
3. Titration Shielding: Prevents trailing conditional changes ("increase dose by 100 mg") from hijacking initial strength.
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, List, Dict, Set, Any
from app.drugs.repository import DrugRepository

DOSAGE_UNIT_REGEX = (
    r"\b(\d+(?:\.\d+)?)\s*(mg(?:\/ml|\/g)?|grams?|gm|g|mcg|µg|ml|l|iu|units?|%|meq)(?!\w)"
)

TITRATION_PATTERN = (
    r"(?i)\b(?:increase|decrease|reduce|double|taper)\s+(?:the\s+)?(?:dose|dosage)\s+(?:by\s+)?(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)?)"
)


def extract_strength_and_formulation(
    clause_text: str,
    base_drug_name: str,
    drug_repo: Optional[DrugRepository] = None,
) -> Tuple[str, Optional[str], Optional[Tuple[int, int]], float]:
    """
    Extracts prescription strength and resolves formulation dose binding to drug name.
    Returns:
      (full_drug_name, order_strength, strength_span, confidence)
    """
    if not base_drug_name:
        return "", None, None, 0.0

    # 1. Mask titration dose directives to prevent titration hijacking
    titration_matches = list(re.finditer(TITRATION_PATTERN, clause_text))
    masked_clause = clause_text
    titration_spans = set()
    for tm in titration_matches:
        titration_spans.add((tm.start(), tm.end()))

    # 2. Extract all dose occurrences that are NOT inside titration clauses
    all_doses = []
    for m in re.finditer(DOSAGE_UNIT_REGEX, masked_clause, re.IGNORECASE):
        start, end = m.start(), m.end()
        # Skip if within a titration span
        is_titration = any(t_start <= start and end <= t_end for t_start, t_end in titration_spans)
        if not is_titration:
            all_doses.append(m)

    # Fetch formulary known strengths for this medicine
    db_strengths: Set[str] = set()
    if drug_repo:
        base_clean = base_drug_name.upper().strip()
        for k, v in drug_repo.drug_strengths_map.items():
            if base_clean in k or k in base_clean:
                db_strengths.update(v)
        if not db_strengths:
            from app.drugs.repository import soundex
            first_word = base_clean.split()[0]
            sx_candidates = drug_repo.soundex_buckets.get(soundex(first_word), [])
            for c in sx_candidates:
                db_strengths.update(drug_repo.get_strengths_for_base(c))

    full_drug_name = base_drug_name
    order_strength: Optional[str] = None
    strength_span: Optional[Tuple[int, int]] = None
    confidence = 0.90

    if len(all_doses) >= 2:
        # Check if doses are separated by 'or' indicating alternative / contradictory clinical order
        is_alternative = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)?\s+or\s+\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)?\b", clause_text, re.IGNORECASE))
        if is_alternative:
            d1 = all_doses[0]
            order_strength = d1.group(0).strip()
            strength_span = (d1.start(), d1.end())
        else:
            # Rule 1: Dual dose (medicine + formulation strength + order strength)
            # First dose is checked against catalog strengths
            d1 = all_doses[0]
            d_final = all_doses[1]
            
            num1 = d1.group(1)
            if num1 in db_strengths or len(all_doses) == 2:
                formulation_str = d1.group(0).strip()
                # If base name doesn't already have this formulation dose, append it
                if formulation_str.lower() not in full_drug_name.lower():
                    full_drug_name = f"{base_drug_name} {formulation_str}".strip()
                order_strength = d_final.group(0).strip()
                strength_span = (d_final.start(), d_final.end())
            else:
                order_strength = d_final.group(0).strip()
                strength_span = (d_final.start(), d_final.end())

    elif len(all_doses) == 1:
        # Rule 2: Single dose
        d = all_doses[0]
        d_txt = d.group(0).strip()
        num = d.group(1)

        if num in db_strengths:
            # Matches formulary specification -> also binds to drug name
            if d_txt.lower() not in full_drug_name.lower():
                full_drug_name = f"{base_drug_name} {d_txt}".strip()
        
        # In canonical clinical schema, strength must be captured as the prescribed formulation strength
        order_strength = d_txt
        strength_span = (d.start(), d.end())

    else:
        # 1. Check if base_drug_name itself contains a numeric formulation dose (e.g. 'Crocin 650', 'Dolo 650', 'Amoxyclav 625', 'Foracort 200', 'Pantocid 40')
        med_num_match = re.search(r"\b(\d+(?:\.\d+)?)\b", base_drug_name)
        if med_num_match:
            num = med_num_match.group(1)
            if (db_strengths and num in db_strengths) or (float(num) in (650, 500, 625, 40, 20, 10, 5, 25, 50, 100, 200, 250, 850, 1000) or (0.1 <= float(num) <= 2000 and num not in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "14", "30"))):
                order_strength = num
                strength_span = (med_num_match.start(1), med_num_match.end(1))
            else:
                order_strength = None
                strength_span = None
        else:
            # 2. Check for unit-less numeric dose following medicine name in clause_text
            first_word = base_drug_name.split()[0]
            pattern = rf"(?i)\b(?:{re.escape(base_drug_name)}|{re.escape(first_word)})\s+(\d+(?:\.\d+)?)\b(?!\s*(?:days?|weeks?|months?|hours?|hrs?|times))"
            unitless_match = re.search(pattern, clause_text)
            if unitless_match:
                num = unitless_match.group(1)
                # Verify if num matches known formulary strength or standard oral formulation integer
                if (db_strengths and num in db_strengths) or (float(num) in (650, 500, 625, 40, 20, 10, 5, 25, 50, 100, 200, 250, 850, 1000) or (0.1 <= float(num) <= 2000 and num not in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "14", "30"))):
                    order_strength = num
                    strength_span = (unitless_match.start(1), unitless_match.end(1))
                    if num not in full_drug_name:
                        full_drug_name = f"{base_drug_name} {num}".strip()
                else:
                    order_strength = None
                    strength_span = None
            else:
                order_strength = None
                strength_span = None

    # Clean up drug name
    full_drug_name = re.sub(
        r"(?i)^(?:take\s+|administer\s+|give\s+|prescribe\s+|start\s+)?(?:\d+\s+|one\s+|two\s+|three\s+)?(?:of\s+|a\s+|an\s+|the\s+)?",
        "",
        full_drug_name,
    ).strip()
    full_drug_name = re.sub(r"(?i)\s+(?:orally|topically|by\s+mouth|inhale|apply|combination)$", "", full_drug_name).strip()
    full_drug_name = re.sub(r"[\s,;\-]+(?=\s+\d)", "", full_drug_name).strip()
    full_drug_name = re.sub(r"\s+", " ", full_drug_name).strip()

    return full_drug_name, order_strength, strength_span, confidence
