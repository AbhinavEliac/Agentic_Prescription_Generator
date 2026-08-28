"""
agents/route_agent.py
---------------------
Route Agent:
Clinical route of administration extractor with complete multi-specialty coverage.
"""
import re
from typing import Dict, Any, List
from graph_state import AgenticRxState, RouteItem
from agents.utils import (
    is_placeholder,
    safe_parse_json,
    segment_prescription,
)
import prompt


def route_agent(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Route Agent.
    """
    input_text = state.get("input_text", "")
    feedback = state.get("validation_feedback", {}).get("route_agent", "")

    extracted_routes: List[RouteItem] = []

    if llm is not None:
        p = prompt.ROUTE_PROMPT.replace("{{VOICE_INPUT}}", input_text)
        p = p.replace("{{FEEDBACK}}", feedback if feedback else "None")
        try:
            raw_out = llm.invoke(p)
            if isinstance(raw_out, str):
                if "Prescription Input:" in raw_out:
                    raw_out = raw_out.split("Prescription Input:")[-1]
                parsed = safe_parse_json(raw_out)
                if isinstance(parsed, list):
                    for idx, item in enumerate(parsed):
                        r_val = item.get("route", "NONE").lower().strip()
                        if not is_placeholder(r_val):
                            extracted_routes.append({
                                "medicine_id": item.get("medicine_id", idx + 1),
                                "drug_name": item.get("drug_name", ""),
                                "route": r_val if r_val else "NONE",
                            })
        except Exception:
            pass

    if not extracted_routes:
        segments = segment_prescription(input_text)
        input_lower = input_text.lower()

        for s in segments:
            m_id = s["medicine_id"]
            clause = s["clause"]
            # Extract first 60 chars of clause (the core medicine administration directive)
            core_directive = clause[:80].lower()
            clause_lower = clause.lower()

            route = "NONE"

            # 1. Check immediate form and administration verb in the core directive first
            if any(w in core_directive for w in ("ear drops", "affected ear", "eardrum", "into each ear")):
                route = "otic"
            elif any(w in core_directive for w in ("eye drops", "into both eyes", "into left eye", "into right eye")):
                route = "ophthalmic"
            elif any(w in core_directive for w in ("nasal spray", "nasal drops", "into each nostril", "both nostrils")):
                route = "nasal"
            elif any(w in core_directive for w in ("inhale", "rotacap", "puff", "revolizer", "spacer", "turbuhaler", "respule", "inhaler")):
                route = "inhalation"
            elif any(w in core_directive for w in ("apply", "gel", "ointment", "cream", "lotion", "massage", "sunscreen", "cleanse", "wash")):
                route = "topical"
            elif any(w in core_directive for w in ("tablet", "tab", "capsule", "cap", "syrup", "sachet", "pill", "suspension", "vial", "consume", "dissolve", "drinking water", "take", "administer", "give", "gargle", "mouthwash", "lozenge", "paste", "mixed in milk", "orally", "overly", "by mouth")):
                route = "oral"

            # 2. Check full clause if not resolved by core directive
            if route == "NONE":
                if any(w in clause_lower for w in ("ear drops", "affected ear", "eardrum")):
                    route = "otic"
                elif any(w in clause_lower for w in ("eye drops", "ophthalmic")):
                    route = "ophthalmic"
                elif any(w in clause_lower for w in ("nasal spray", "nasal drops", "nostril")):
                    route = "nasal"
                elif any(w in clause_lower for w in ("rotacap", "revolizer", "puff", "spacer", "turbuhaler")):
                    route = "inhalation"
                elif any(w in clause_lower for w in ("onto your lower back", "along the clean suture line", "to the affected patches", "over dry skin", "sunscreen")):
                    route = "topical"
                elif any(w in clause_lower for w in ("tablet", "tab", "capsule", "cap", "syrup", "sachet", "pill", "vial", "orally", "by mouth")):
                    route = "oral"
                elif "take" in input_lower or "administer" in input_lower or "consume" in input_lower:
                    route = "oral"

            extracted_routes.append({
                "medicine_id": m_id,
                "drug_name": "",
                "route": route,
            })

    return {"routes": extracted_routes}
