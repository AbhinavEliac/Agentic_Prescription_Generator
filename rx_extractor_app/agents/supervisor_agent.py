"""
agents/supervisor_agent.py
--------------------------
Supervisor Agent: Coordinates multi-agent graph execution, cleans conversational noise/transcription artifacts,
tracks iterations (capped at max 3), and routes targeted feedback across iterations.
"""
from typing import Dict, Any
from graph_state import AgenticRxState
from agents.utils import clean_noise_and_chatter


def supervisor_node(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Supervisor Agent:
    Cleans transcript noise and coordinates parallel extraction tasks.
    """
    raw_input = state.get("input_text", "").strip()
    cleaned_input = clean_noise_and_chatter(raw_input)
    current_iteration = state.get("iteration_count", 0) + 1
    feedback = state.get("validation_feedback", {})

    log_entry = {
        "agent": "Supervisor",
        "iteration": current_iteration,
        "action": "Noise filtering & dispatching parallel extraction agents",
        "has_feedback": bool(feedback and any(feedback.values())),
    }

    current_logs = list(state.get("agent_logs", []))
    current_logs.append(log_entry)

    return {
        "input_text": cleaned_input if cleaned_input else raw_input,
        "iteration_count": current_iteration,
        "agent_logs": current_logs,
    }
