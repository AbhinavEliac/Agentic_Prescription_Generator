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
    r"\b(\d+(?:\.\d+)?)\s*(mg(?:\/ml|\/g)?|grams?|gm|g|mcg|µg|ml|l|iu|units?|%|meq|"
    r"tablets?|tabs?|capsules?|caps?|pills?|sachets?|vials?|drops?|sprays?|puffs?|rotacaps?|respules?)(?!\w)"
)

TITRATION_PATTERN = (
    r"(?i)\b(?:increase|decrease|reduce|double|taper)\s+(?:the\s+)?(?:dose|dosage)\s+(?:by\s+)?(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)?)"
)

WORD_NUMBERS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "half": "0.5",
    "single": "1",
}


def normalize_clause_word_numbers(clause_text: str) -> str:
    """Normalizes spoken number words preceding dosage forms (e.g. 'one tablet' -> '1 tablet')."""
    pattern = r"(?i)\b(one|two|three|four|five|half|single)\s+(tablets?|tabs?|capsules?|caps?|pills?|sachets?|vials?|drops?|sprays?|puffs?|rotacaps?|respules?)\b"
    return re.sub(pattern, lambda m: f"{WORD_NUMBERS.get(m.group(1).lower(), m.group(1))} {m.group(2)}", clause_text)


FORM_COUNT_UNITS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "pill", "pills", "sachet", "sachets", "vial", "vials", "drop", "drops",
    "spray", "sprays", "puff", "puffs", "rotacap", "rotacaps", "respule", "respules"
}


def _has_int_in_drug_name(drug_name_str: str, num_str: str) -> bool:
    """Checks if integer num exists in registered drug name outside parenthetical generic strength."""
    if not drug_name_str or not num_str:
        return False
    name_without_parens = re.sub(r"\(.*?\)", "", drug_name_str)
    return bool(re.search(rf"(?<!\d){re.escape(num_str)}(?!\d)", name_without_parens, re.IGNORECASE))


def extract_strength_and_formulation(
    clause_text: str,
    base_drug_name: str,
    drug_repo: Optional[DrugRepository] = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[Tuple[int, int]], float, List[Dict[str, Any]]]:
    """
    Extracts prescription strength and resolves formulation dose binding vs order dose according to:
    Rule 1: medicine + (int1 + units1) + (int2 + units2) -> 2nd/final is dose & dose_unit; 1st is formulation strength in name.
    Rule 2: medicine + (int + units) -> int is dose only when Drug_database does NOT have those int+units in their names,
            else no dose, just medicine with name and int+units within the name.

    Returns:
      (full_drug_name, strength, dose, dose_unit, strength_span, confidence, composite_options)
    """
    if not base_drug_name:
        return "", None, None, None, None, 0.0, []

    normalized_clause = normalize_clause_word_numbers(clause_text)

    # 1. Mask titration dose directives to prevent titration hijacking
    titration_matches = list(re.finditer(TITRATION_PATTERN, normalized_clause))
    titration_spans = set()
    for tm in titration_matches:
        titration_spans.add((tm.start(), tm.end()))

    # 2. Extract all dose occurrences that are NOT inside titration clauses
    all_doses = []
    for m in re.finditer(DOSAGE_UNIT_REGEX, normalized_clause, re.IGNORECASE):
        start, end = m.start(), m.end()
        is_titration = any(t_start <= start and end <= t_end for t_start, t_end in titration_spans)
        if not is_titration:
            all_doses.append(m)

    # Fetch formulary entries and check if database drug names contain integer + units
    db_strengths: Set[str] = set()
    db_entries_for_drug: List[Any] = []
    if drug_repo:
        base_clean = base_drug_name.upper().strip()
        db_entries_for_drug = drug_repo.drugs_by_clean_name.get(base_clean, [])
        if not db_entries_for_drug and base_clean in drug_repo.brand_index:
            db_entries_for_drug = drug_repo.brand_index[base_clean]
        if not db_entries_for_drug:
            first_w = base_clean.split()[0]
            db_entries_for_drug = drug_repo.drugs_by_clean_name.get(first_w, [])
        if not db_entries_for_drug:
            fuzzy = drug_repo.find_fuzzy(base_clean, min_confidence=0.68, limit=3)
            if fuzzy:
                db_entries_for_drug = [f.drug for f in fuzzy]

        for d in db_entries_for_drug:
            db_strengths.update(d.strength_values)
        if not db_strengths:
            for k, v in drug_repo.drug_strengths_map.items():
                if base_clean in k or k in base_clean:
                    db_strengths.update(v)

    full_drug_name = base_drug_name
    order_strength: Optional[str] = None
    dose: Optional[str] = None
    dose_unit: Optional[str] = None
    strength_span: Optional[Tuple[int, int]] = None
    confidence = 0.90
    composite_options: List[Dict[str, Any]] = []

    # 0. Check for combination dose pattern: (int-1 units) + (int-2 units) e.g. "500 mg + 125 mg", "500 + 125 mg"
    combo_regex = r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml)?\s*(?:\+|\band\b|\/)\s*(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml)?"
    combo_match = re.search(combo_regex, normalized_clause, re.IGNORECASE)
    if combo_match and drug_repo:
        try:
            int1 = float(combo_match.group(1))
            int2 = float(combo_match.group(3))
            unit = combo_match.group(4) or combo_match.group(2) or "mg"
            int0 = int1 + int2

            composite_entry = drug_repo.find_composite_formulation(base_drug_name, int1, int2, unit)
            if composite_entry:
                int1_s = str(int(int1)) if int1.is_integer() else str(int1)
                int2_s = str(int(int2)) if int2.is_integer() else str(int2)
                int0_s = str(int(int0)) if int0.is_integer() else str(int0)

                # DO NOT make (int-1 units) as dose automatically and drop int-2 units!
                # Keep dose as None (pending doctor confirmation), set order_strength as combination string
                order_strength = f"{int1_s} + {int2_s} {unit}"
                dose = None
                dose_unit = None
                strength_span = (combo_match.start(), combo_match.end())

                same_dose = drug_repo.find_same_dose_formulations(composite_entry.drug_id, int0_s)
                composite_options = [{
                    "drug_name": composite_entry.drug_name,
                    "base_name": composite_entry.base_name.title(),
                    "drug_id": composite_entry.drug_id,
                    "confidence": 0.95,
                    "dose": int0_s,
                    "dose_unit": unit,
                    "available_routes": composite_entry.routes,
                    "available_drugs": same_dose,
                }]

                if db_entries_for_drug:
                    for d in db_entries_for_drug:
                        if d.drug_id != composite_entry.drug_id and len(composite_options) < 3:
                            t_dose = sorted(d.strength_values)[0] if d.strength_values else None
                            composite_options.append({
                                "drug_name": d.drug_name,
                                "base_name": d.base_name.title(),
                                "drug_id": d.drug_id,
                                "confidence": 0.85,
                                "dose": t_dose,
                                "dose_unit": unit if t_dose else "",
                                "available_routes": d.routes,
                                "available_drugs": drug_repo.find_same_dose_formulations(d.drug_id, t_dose or "")
                            })

                return full_drug_name, order_strength, dose, dose_unit, strength_span, confidence, composite_options
        except Exception:
            pass

    # Partition matches into active chemical substance doses vs administration count units
    substance_doses = [m for m in all_doses if m.group(2).lower() not in FORM_COUNT_UNITS]
    count_doses = [m for m in all_doses if m.group(2).lower() in FORM_COUNT_UNITS]

    if len(substance_doses) >= 2:
        # Check if doses are separated by 'or' indicating alternative
        is_alternative = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)?\s+or\s+\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml)?\b", normalized_clause, re.IGNORECASE))
        if is_alternative:
            d1 = substance_doses[0]
            order_strength = d1.group(0).strip()
            strength_span = (d1.start(), d1.end())
        else:
            # Rule 1: medicine + (int1 + units1) + (int2 + units2)
            # Second / final is dose and dose units! First is formulation strength!
            d1 = substance_doses[0]
            d_final = substance_doses[-1]

            num1 = d1.group(1)
            num_final = d_final.group(1)
            unit_final = d_final.group(2)

            formulation_str = d1.group(0).strip()
            if num1 not in full_drug_name:
                full_drug_name = f"{base_drug_name} {formulation_str}".strip()

            order_strength = d_final.group(0).strip()
            dose = num_final
            dose_unit = unit_final
            strength_span = (d_final.start(), d_final.end())

    elif len(substance_doses) == 1:
        # Rule 2: medicine + (integer + units)
        d = substance_doses[0]
        d_txt = d.group(0).strip()
        num = d.group(1)
        unit = d.group(2)

        has_int_in_db_names = False
        if db_entries_for_drug:
            for entry in db_entries_for_drug:
                if num in entry.strength_values:
                    has_int_in_db_names = True
                    break
                d_name = getattr(entry, "drug_name", "") or ""
                b_name = getattr(entry, "base_name", "") or ""
                br_name = getattr(entry, "brand_name", "") or ""
                if (
                    _has_int_in_drug_name(d_name, num)
                    or _has_int_in_drug_name(b_name, num)
                    or _has_int_in_drug_name(br_name, num)
                ):
                    has_int_in_db_names = True
                    break
        elif num in db_strengths:
            has_int_in_db_names = True
        elif _has_int_in_drug_name(base_drug_name, num):
            has_int_in_db_names = True

        if has_int_in_db_names:
            # Database HAS this integer in its name -> No dose! Just medicine with name and integer+units
            dose = None
            dose_unit = None
            order_strength = d_txt
            if num not in full_drug_name:
                full_drug_name = f"{base_drug_name} {d_txt}".strip()
            strength_span = (d.start(), d.end())
        else:
            # Database does NOT have integer in its name -> integer is dose!
            dose = num
            dose_unit = unit
            order_strength = d_txt
            full_drug_name = base_drug_name
            strength_span = (d.start(), d.end())

    elif len(count_doses) >= 1:
        cd = count_doses[0]
        dose = cd.group(1)
        dose_unit = cd.group(2)
        strength_span = (cd.start(), cd.end())

    else:
        # Check if base_drug_name itself contains a numeric formulation dose (e.g. 'Crocin 650', 'Dolo 650')
        med_num_match = re.search(r"\b(\d+(?:\.\d+)?)\b", base_drug_name)
        if med_num_match:
            num = med_num_match.group(1)
            order_strength = num
            dose = None
            dose_unit = None
            strength_span = (med_num_match.start(1), med_num_match.end(1))
        else:
            # Check for unit-less numeric dose in clause_text
            unitless_match = None
            first_word = base_drug_name.split()[0]
            pattern = rf"(?i)\b(?:{re.escape(base_drug_name)}|{re.escape(first_word)})\s+(\d+(?:\.\d+)?)\b(?!\s*(?:days?|weeks?|months?|hours?|hrs?|times))"
            verbatim_m = re.search(pattern, normalized_clause)
            if verbatim_m:
                unitless_match = verbatim_m
            else:
                # If medicine was matched phonetically from spoken tokens, look for numeric token in clause
                num_cands = list(re.finditer(r"\b(\d+(?:\.\d+)?)\b(?!\s*(?:days?|weeks?|months?|hours?|hrs?|times))", normalized_clause))
                for nc in num_cands:
                    val = nc.group(1)
                    # Skip doctor timing shorthand codes (e.g. 101, 100, 111, 110, 010, 001)
                    if re.match(r"^[01]{3,4}$", val) and any(w in normalized_clause.lower() for w in ("daily", "day", "twice", "once", "thrice")):
                        continue
                    unitless_match = nc
                    break

            if unitless_match:
                num = unitless_match.group(1)
                has_int_in_db_names = False
                if db_entries_for_drug:
                    for entry in db_entries_for_drug:
                        d_name = getattr(entry, "drug_name", "") or ""
                        b_name = getattr(entry, "base_name", "") or ""
                        br_name = getattr(entry, "brand_name", "") or ""
                        if (
                            _has_int_in_drug_name(d_name, num)
                            or _has_int_in_drug_name(b_name, num)
                            or _has_int_in_drug_name(br_name, num)
                        ):
                            has_int_in_db_names = True
                            break
                elif _has_int_in_drug_name(base_drug_name, num):
                    has_int_in_db_names = True

                if has_int_in_db_names:
                    dose = None
                    dose_unit = None
                    order_strength = num
                    if num not in full_drug_name:
                        full_drug_name = f"{base_drug_name} {num}".strip()
                else:
                    dose = num
                    dose_unit = None
                    order_strength = num
                    full_drug_name = base_drug_name
                strength_span = (unitless_match.start(1), unitless_match.end(1))
            else:
                order_strength = None
                dose = None
                dose_unit = None
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

    return full_drug_name, order_strength, dose, dose_unit, strength_span, confidence, composite_options
