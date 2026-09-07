"""
agents/medicine_strength_agent.py
---------------------------------
Medicine & Strength Agent:
Drift-proof extractor for drug names and strengths across any clinical prescription format.
"""
import re
from typing import Dict, Any, List
from graph_state import AgenticRxState, MedicineItem
from agents.utils import (
    FORM_PATTERN,
    ACTION_VERBS_PATTERN,
    is_placeholder,
    safe_parse_json,
    segment_prescription,
)
import prompt
import os
import json

DOSAGE_REGEX = r"\d+(?:\.\d+)?\s*(?:mg(?:\/ml|\/g)?|g|mcg|µg|ml|l|iu|units?|%|meq|puffs?|drops?|tablets?|capsules?|sachets?|vials?)"

_DRUG_STRENGTHS_MAP = None

def _get_drug_strengths_map():
    global _DRUG_STRENGTHS_MAP
    if _DRUG_STRENGTHS_MAP is None:
        _DRUG_STRENGTHS_MAP = {}
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "Drug_databse", "drugList.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    data = json.load(f).get("drugData", [])
                    for item in data:
                        raw_name = item.get("drug_name", "").upper()
                        base = re.sub(r"\b(TAB|TABS|TABLET|TABLETS|CAP|CAPS|CAPSULE|CAPSULES|SYP|SYRUP|INJ|INJECTION|DROPS)\b", "", raw_name)
                        base = re.sub(r"\b\d+(?:\.\d+)?\s*(MG|G|MCG|ML|L|IU|%)\b", "", base)
                        base = re.sub(r"[^\w\s]", "", base).strip()
                        strengths = re.findall(r"(\d+(?:\.\d+)?)\s*(?:MG|G|MCG|ML|L|IU|%)\b", raw_name)
                        if base:
                            if base not in _DRUG_STRENGTHS_MAP:
                                _DRUG_STRENGTHS_MAP[base] = set()
                            for s in strengths:
                                _DRUG_STRENGTHS_MAP[base].add(s)
            except Exception:
                pass
    return _DRUG_STRENGTHS_MAP


def medicine_strength_agent(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Medicine & Strength Agent.
    """
    input_text = state.get("input_text", "")
    feedback = state.get("validation_feedback", {}).get("medicine_agent", "")

    extracted_meds: List[MedicineItem] = []

    if llm is not None:
        p = prompt.MEDICINE_STRENGTH_PROMPT.replace("{{VOICE_INPUT}}", input_text)
        p = p.replace("{{FEEDBACK}}", feedback if feedback else "None")
        try:
            raw_out = llm.invoke(p)
            if isinstance(raw_out, str):
                if "Prescription Input:" in raw_out:
                    raw_out = raw_out.split("Prescription Input:")[-1]
                parsed = safe_parse_json(raw_out)
                if isinstance(parsed, list):
                    for idx, item in enumerate(parsed):
                        d_name = item.get("drug_name") or item.get("Drug_name", "")
                        strength = item.get("strength", "NONE")
                        if d_name and d_name != "NONE" and not is_placeholder(d_name):
                            core_words = [
                                w for w in re.findall(r"[A-Za-z0-9\-]+", d_name)
                                if len(w) >= 3 and w.lower() not in (
                                    "take", "tab", "tabs", "tablet", "capsule", "syrup",
                                    "pill", "rotacap", "none", "vial", "sachet", "one", "administer"
                                )
                            ]
                            if core_words and any(cw.lower() in input_text.lower() for cw in core_words):
                                d_name = re.sub(FORM_PATTERN, "", d_name).strip()
                                d_name = re.sub(ACTION_VERBS_PATTERN, "", d_name).strip()
                                d_name = re.sub(r"\s+", " ", d_name).strip()
                                extracted_meds.append({
                                    "medicine_id": idx + 1,
                                    "drug_name": d_name,
                                    "strength": "NONE" if is_placeholder(strength) else (strength if strength else "NONE"),
                                })
        except Exception:
            pass

    if not extracted_meds:
        segments = segment_prescription(input_text)
        for s in segments:
            m_id = s["medicine_id"]
            clause = s["clause"]
            seed = s.get("seed_name", "")

            if seed:
                cleaned_name = re.sub(FORM_PATTERN, "", seed).strip()
                cleaned_name = re.sub(ACTION_VERBS_PATTERN, "", cleaned_name).strip()
                cleaned_name = re.sub(r"(?i)^(?:take\s+|administer\s+|give\s+|prescribe\s+|start\s+)?(?:\d+\s+|one\s+|two\s+|three\s+)?", "", cleaned_name).strip()
                cleaned_name = re.sub(r"\b(?:of|one|vial|sachet|combination|ear|eye|nasal|oral|topical)\b", "", cleaned_name, flags=re.IGNORECASE).strip()
                extracted_meds.append({
                    "medicine_id": m_id,
                    "drug_name": re.sub(r"\s+", " ", cleaned_name).strip(),
                    "strength": "NONE",
                })
            else:
                doses = list(re.finditer(DOSAGE_REGEX, clause, re.IGNORECASE))
                
                if doses:
                    raw_lead = clause[:doses[0].start()].strip()
                    cleaned_name = re.sub(FORM_PATTERN, "", raw_lead).strip()
                    cleaned_name = re.sub(ACTION_VERBS_PATTERN, "", cleaned_name).strip()
                    cleaned_name = re.sub(r"(?i)^(?:take|administer|give|start|prescribe|consume|dissolve|inhale|apply|put|instill|inject|infuse)\s+", "", cleaned_name).strip()
                    cleaned_name = re.sub(r"(?i)^(?:of\s+|a\s+|an\s+|the\s+)", "", cleaned_name).strip()
                    cleaned_name = re.sub(r"[\s,;\-]+$", "", cleaned_name).strip()
                    cleaned_name = re.sub(r"\s*,\s*", " ", cleaned_name).strip()

                    # Check which integers exist with this medicine in the database
                    db_map = _get_drug_strengths_map()
                    base_upper = cleaned_name.upper()
                    db_strengths = set()
                    for k, v in db_map.items():
                        if base_upper in k or k in base_upper:
                            db_strengths.update(v)

                    # Partition spoken doses
                    matching_db_doses = []
                    non_matching_doses = []
                    for d in doses:
                        d_txt = d.group(0).strip()
                        num_m = re.search(r"\d+(?:\.\d+)?", d_txt)
                        num = num_m.group(0) if num_m else ""
                        if num and num in db_strengths:
                            matching_db_doses.append(d_txt)
                        else:
                            non_matching_doses.append(d_txt)

                    if len(doses) >= 2:
                        # Rule 1: medicine + (integer + units) + (integer + units) == second or final integer + units is dose and dose units
                        order_strength = doses[-1].group(0).strip()
                        # The earlier integer binds to formulation strength if in database
                        formulation_strength = doses[0].group(0).strip()
                        full_drug_name = f"{cleaned_name} {formulation_strength}".strip()
                    elif len(doses) == 1:
                        # Rule 2: medicine + (integer + units) == integer is dose only when the database of medicines/drugs do not have those integer + units with their names, else no dose just medicine with name and integer + units within the name
                        d_txt = doses[0].group(0).strip()
                        num_m = re.search(r"\d+(?:\.\d+)?", d_txt)
                        num = num_m.group(0) if num_m else ""
                        if num and num in db_strengths:
                            # DB has this integer with medicine name -> NO DOSE!
                            order_strength = "NONE"
                            full_drug_name = f"{cleaned_name} {d_txt}".strip()
                        else:
                            # DB does not have this integer with medicine name -> integer is dose!
                            order_strength = d_txt
                            full_drug_name = cleaned_name
                    else:
                        order_strength = "NONE"
                        full_drug_name = cleaned_name

                    full_drug_name = re.sub(r"(?i)^(?:take\s+|administer\s+|give\s+|prescribe\s+|start\s+)?(?:\d+\s+|one\s+|two\s+|three\s+)?(?:of\s+|a\s+|an\s+|the\s+)?", "", full_drug_name).strip()
                    full_drug_name = re.sub(r"(?i)\s+(?:orally|topically|by\s+mouth|inhale|apply|combination)$", "", full_drug_name).strip()
                    full_drug_name = re.sub(r"[\s,;\-]+(?=\s+\d)", "", full_drug_name).strip()
                    full_drug_name = re.sub(r"\s+", " ", full_drug_name).strip()

                    extracted_meds.append({
                        "medicine_id": m_id,
                        "drug_name": full_drug_name,
                        "strength": order_strength,
                    })
                else:
                    match_nodose = re.search(
                        r"(?:take|administer|give|consume|dissolve|inhale|apply|put|instill|gently\s+massage|massage|cleanse)?\s*(?:one|two|three)?\s*(?:tablet|tab|capsule|cap|rotacap|pill|vial|sachet|puff)?\s*(?:of\s+)?([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,3}?)\s*(?:tablet|capsule|oral\s+suspension|suspension|sachet|vial|cream|ointment|gel|drops|rotacap|puff|paste|wash|orally|topically|by\s+mouth|before|after|twice|once|three|up\s+to|in\s+one\s+liter|every|onto|along|to\s+the|over|for|till|upto|until|daily|SOS|HS|STAT)\b",
                        clause,
                        re.IGNORECASE,
                    )
                    if match_nodose:
                        raw_name = match_nodose.group(1).strip()
                        cleaned_name = re.sub(FORM_PATTERN, "", raw_name).strip()
                        cleaned_name = re.sub(ACTION_VERBS_PATTERN, "", cleaned_name).strip()
                        cleaned_name = re.sub(r"\b(?:of|one|vial|sachet|combination)\b", "", cleaned_name, flags=re.IGNORECASE).strip()
                        if cleaned_name and len(cleaned_name) >= 3 and cleaned_name.lower() not in ("stick", "avoid", "visit", "seek", "please", "keep", "fomentation", "all the medicines", "both of them"):
                            extracted_meds.append({
                                "medicine_id": m_id,
                                "drug_name": re.sub(r"\s+", " ", cleaned_name).strip(),
                                "strength": "NONE",
                            })

    if not extracted_meds:
        extracted_meds.append({"medicine_id": 1, "drug_name": "NONE", "strength": "NONE"})

    return {"medicines": extracted_meds}
