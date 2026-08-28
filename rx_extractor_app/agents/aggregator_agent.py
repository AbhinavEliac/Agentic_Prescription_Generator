"""
agents/aggregator_agent.py
--------------------------
Aggregator Agent:
Unifies extracted outputs from all parallel agents into complete clinical records (including additional_instruction),
and filters out non-medication noise entities.
"""
from typing import Dict, Any, List
from graph_state import AgenticRxState, PrescriptionBlock
from agents.utils import is_valid_medication_entity, clean_instruction_noise


def aggregator_agent(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Aggregator Agent with noise-proofing entity validation.
    """
    meds = state.get("medicines", [])
    routes = {r.get("medicine_id", idx + 1): r.get("route", "NONE") for idx, r in enumerate(state.get("routes", []))}
    dur_freq = {df.get("medicine_id", idx + 1): df for idx, df in enumerate(state.get("durations_frequencies", []))}
    
    inst_map = {}
    add_inst_map = {}
    for idx, ins in enumerate(state.get("instructions", [])):
        m_id = ins.get("medicine_id", idx + 1)
        inst_map[m_id] = clean_instruction_noise(ins.get("instruction", "NONE"))
        add_inst_map[m_id] = clean_instruction_noise(ins.get("additional_instruction", "NONE"))

    aggregated: List[PrescriptionBlock] = []

    for idx, med in enumerate(meds):
        m_id = med.get("medicine_id", idx + 1)
        df_entry = dur_freq.get(m_id, {})
        d_name = med.get("drug_name", "NONE").strip()

        # Noise filter: ensure entity is a valid pharmaceutical drug name
        if not is_valid_medication_entity(d_name) and len(meds) > 1:
            continue

        block: PrescriptionBlock = {
            "Drug_name": d_name,
            "strength": med.get("strength", "NONE"),
            "frequency": df_entry.get("frequency", "NONE"),
            "duration": df_entry.get("duration", "NONE"),
            "route": routes.get(m_id, "NONE"),
            "instruction": inst_map.get(m_id, "NONE"),
            "additional_instruction": add_inst_map.get(m_id, "NONE"),
        }
        aggregated.append(block)

    log_entry = {
        "agent": "Aggregator",
        "action": f"Aggregated {len(aggregated)} noise-proof medicine block(s)",
    }
    current_logs = list(state.get("agent_logs", []))
    current_logs.append(log_entry)

    return {
        "aggregated_blocks": aggregated,
        "agent_logs": current_logs,
    }
