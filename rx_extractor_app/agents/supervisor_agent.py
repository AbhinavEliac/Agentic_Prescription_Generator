"""
agents/supervisor_agent.py
--------------------------
Supervisor Agent: Coordinates multi-agent graph execution, tracks iterations (capped at max 3),
and routes targeted feedback across iterations.
"""
from typing import Dict, Any
from graph_state import AgenticRxState


def supervisor_node(state: AgenticRxState, llm: Any = None) -> Dict[str, Any]:
    """
    Supervisor Agent:
    Coordinates extraction tasks and manages validation-correction iterations.
    """
    input_text = state.get("input_text", "").strip()
    current_iteration = state.get("iteration_count", 0) + 1
    feedback = state.get("validation_feedback", {})

    log_entry = {
        "agent": "Supervisor",
        "iteration": current_iteration,
        "action": "Dispatching parallel extraction agents",
        "has_feedback": bool(feedback and any(feedback.values())),
    }

    current_logs = list(state.get("agent_logs", []))
    current_logs.append(log_entry)

    return {
        "iteration_count": current_iteration,
        "agent_logs": current_logs,
    }
