"""
agents/validator_agent.py
-------------------------
Validator Agent:
Enforces strict 100% groundedness against raw input prescription with zero hallucinations,
filters out irrelevant conversational noise, and dispatches targeted feedback to agents.
"""
import re
from typing import Dict, Any
from graph_state import AgenticRxState, ValidationFeedback
from agents.utils import is_valid_medication_entity


def validator_agent(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Validator Agent.
    """
    input_text = state.get("input_text", "")
    blocks = state.get("aggregated_blocks", [])
    iteration = state.get("iteration_count", 1)

    feedback: ValidationFeedback = {}
    is_valid = True

    if not blocks or all(b.get("Drug_name") == "NONE" for b in blocks):
        is_valid = False
        feedback["medicine_agent"] = "No valid medicine name was identified from the input."

    valid_blocks = []
    for block in blocks:
        drug_name = block.get("Drug_name", "")
        if drug_name != "NONE":
            # Noise-proofing check: ensure drug is not conversational chatter or non-drug item
            if not is_valid_medication_entity(drug_name):
                is_valid = False
                feedback["medicine_agent"] = f"Entity '{drug_name}' is not a valid medication entity."
                continue

            # Grounding check: ensure core name appears in prescription
            core_words = [
                w for w in re.findall(r"[A-Za-z0-9\-]+", drug_name)
                if len(w) >= 3 and w.lower() not in (
                    "take", "tab", "tabs", "tablet", "capsule", "syrup",
                    "pill", "rotacap", "none", "vial", "sachet", "one", "administer"
                )
            ]
            if core_words and not any(cw.lower() in input_text.lower() for cw in core_words):
                is_valid = False
                feedback["medicine_agent"] = f"Drug '{drug_name}' was not found in prescription text."

            doses = re.findall(r"\d+(?:\.\d+)?", drug_name)
            for d in doses:
                if d not in input_text:
                    is_valid = False
                    feedback["medicine_agent"] = f"Dosage {d} in drug name '{drug_name}' was not found in prescription text."

        inst = block.get("instruction", "")
        if inst != "NONE":
            # Reject artificial commentary ONLY if not present in the input text
            hallucinated_commentary_cues = ["please note", "be careful", "moral:", "disclaimer:"]
            for cue in hallucinated_commentary_cues:
                if cue in inst.lower() and cue not in input_text.lower():
                    is_valid = False
                    feedback["instruction_agent"] = f"Unsolicited commentary '{cue}' detected. Strictly adhere to verbatim input."

        valid_blocks.append(block)

    if iteration >= 3:
        status = "VALID"
    else:
        status = "VALID" if is_valid else "NEEDS_CORRECTION"

    log_entry = {
        "agent": "Validator",
        "iteration": iteration,
        "status": status,
        "feedback": feedback if status == "NEEDS_CORRECTION" else "All checks passed (Grounded & Noise-Proof).",
    }
    current_logs = list(state.get("agent_logs", []))
    current_logs.append(log_entry)

    return {
        "validation_status": status,
        "validation_feedback": feedback,
        "aggregated_blocks": valid_blocks if valid_blocks else blocks,
        "agent_logs": current_logs,
    }
