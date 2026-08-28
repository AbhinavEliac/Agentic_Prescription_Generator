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

DOSAGE_REGEX = r"\d+(?:\.\d+)?\s*(?:mg(?:\/ml|\/g)?|g|mcg|µg|ml|l|iu|units?|%|meq|puffs?|drops?|tablets?|capsules?|sachets?|vials?)"


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
                    first_dose_match = doses[0]
                    first_dose_end = first_dose_match.end()
                    first_dose_txt = first_dose_match.group(0).strip()
                    
                    raw_lead = clause[:first_dose_match.start()].strip()
                    cleaned_name = re.sub(FORM_PATTERN, "", raw_lead).strip()
                    cleaned_name = re.sub(ACTION_VERBS_PATTERN, "", cleaned_name).strip()
                    cleaned_name = re.sub(r"(?i)^(?:take|administer|give|start|prescribe|consume|dissolve|inhale|apply|put|instill|inject|infuse)\s+", "", cleaned_name).strip()
                    cleaned_name = re.sub(r"(?i)^(?:of\s+|a\s+|an\s+|the\s+)", "", cleaned_name).strip()
                    cleaned_name = re.sub(r"[\s,;\-]+$", "", cleaned_name).strip()
                    cleaned_name = re.sub(r"\s*,\s*", " ", cleaned_name).strip()
                    
                    second_dose_txt = "NONE"
                    if len(doses) >= 2:
                        second_dose_cand = doses[1]
                        gap_text = clause[first_dose_end:second_dose_cand.start()].strip()
                        if len(gap_text) <= 5 or gap_text.lower() in ("+", "/", "and", "with"):
                            second_dose_txt = second_dose_cand.group(0).strip()

                    full_drug_name = f"{cleaned_name} {first_dose_txt}".strip()
                    full_drug_name = re.sub(r"(?i)^(?:take\s+|administer\s+|give\s+|prescribe\s+|start\s+)?(?:\d+\s+|one\s+|two\s+|three\s+)?(?:of\s+|a\s+|an\s+|the\s+)?", "", full_drug_name).strip()
                    full_drug_name = re.sub(r"(?i)\s+(?:orally|topically|by\s+mouth|inhale|apply|combination)$", "", full_drug_name).strip()
                    full_drug_name = re.sub(r"[\s,;\-]+(?=\s+\d)", "", full_drug_name).strip()
                    full_drug_name = re.sub(r"\s+", " ", full_drug_name).strip()

                    extracted_meds.append({
                        "medicine_id": m_id,
                        "drug_name": full_drug_name,
                        "strength": second_dose_txt,
                    })
                else:
                    match_nodose = re.search(
                        r"(?:take|administer|give|consume|dissolve|inhale|apply|put|instill|gently\s+massage|massage|cleanse)?\s*(?:one|two|three)?\s*(?:tablet|tab|capsule|cap|rotacap|pill|vial|sachet|puff)?\s*(?:of\s+)?([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,3}?)\s*(?:tablet|capsule|oral\s+suspension|suspension|sachet|vial|cream|ointment|gel|drops|rotacap|puff|paste|wash|orally|topically|by\s+mouth|before|after|twice|once|three|up\s+to|in\s+one\s+liter|every|onto|along|to\s+the|over)\b",
                        clause,
                        re.IGNORECASE,
                    )
                    if match_nodose:
                        raw_name = match_nodose.group(1).strip()
                        cleaned_name = re.sub(FORM_PATTERN, "", raw_name).strip()
                        cleaned_name = re.sub(ACTION_VERBS_PATTERN, "", cleaned_name).strip()
                        cleaned_name = re.sub(r"\b(?:of|one|vial|sachet|combination)\b", "", cleaned_name, flags=re.IGNORECASE).strip()
                        if cleaned_name and len(cleaned_name) >= 3 and cleaned_name.lower() not in ("stick", "avoid", "visit", "seek", "please", "keep", "fomentation"):
                            extracted_meds.append({
                                "medicine_id": m_id,
                                "drug_name": re.sub(r"\s+", " ", cleaned_name).strip(),
                                "strength": "NONE",
                            })

    if not extracted_meds:
        extracted_meds.append({"medicine_id": 1, "drug_name": "NONE", "strength": "NONE"})

    return {"medicines": extracted_meds}
