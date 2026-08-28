"""
agents/formatter_agent.py
-------------------------
Formatter Node:
Generates the exact structured block representation (including additional_instruction) with zero extraneous commentary.
"""
from typing import Dict, Any
from graph_state import AgenticRxState


def formatter_node(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Formatter Node:
    Produces the strict structured blocks format with no explanations, notes, or moralizing.
    """
    blocks = state.get("aggregated_blocks", [])
    output_lines = []

    for b in blocks:
        output_lines.append(f"Drug_name: {b.get('Drug_name', 'NONE')}")
        output_lines.append(f"strength: {b.get('strength', 'NONE')}")
        output_lines.append(f"frequency: {b.get('frequency', 'NONE')}")
        output_lines.append(f"duration: {b.get('duration', 'NONE')}")
        output_lines.append(f"route: {b.get('route', 'NONE')}")
        output_lines.append(f"instruction: {b.get('instruction', 'NONE')}")
        output_lines.append(f"additional_instruction: {b.get('additional_instruction', 'NONE')}")

    final_text = "\n".join(output_lines)

    log_entry = {
        "agent": "Formatter",
        "action": "Generated final structured clinical blocks",
    }
    current_logs = list(state.get("agent_logs", []))
    current_logs.append(log_entry)

    return {
        "final_output": final_text,
        "agent_logs": current_logs,
    }
