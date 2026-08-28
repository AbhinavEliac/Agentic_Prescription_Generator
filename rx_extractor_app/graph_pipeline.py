"""
graph_pipeline.py
-----------------
Compiles and executes the LangGraph Multi-Agent Prescription Extractor.
Features:
- Supervisor node coordinating parallel extraction agents
- 4 Parallel specialized extraction agents (Medicine/Strength, Route, Duration/Frequency, Instructions)
- Aggregator node combining parallel outputs into unified 6-field records
- Validator node enforcing 100% groundedness, anti-hallucination, and feedback loops (max 3 reps)
- Strict 6-field formatter for clinical team use.
"""
import time
from typing import Tuple, Dict, Any, List

from graph_state import AgenticRxState
from agents import (
    supervisor_node,
    medicine_strength_agent,
    route_agent,
    duration_frequency_agent,
    instruction_agent,
    aggregator_agent,
    validator_agent,
    formatter_node,
)


def _build_langgraph(llm: Any = None):
    """Builds and compiles the StateGraph using LangGraph."""
    try:
        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(AgenticRxState)

        # 1. Register Nodes
        builder.add_node("supervisor", lambda state: supervisor_node(state, llm))
        builder.add_node("medicine_agent", lambda state: medicine_strength_agent(state, llm))
        builder.add_node("route_agent", lambda state: route_agent(state, llm))
        builder.add_node("duration_frequency_agent", lambda state: duration_frequency_agent(state, llm))
        builder.add_node("instruction_agent", lambda state: instruction_agent(state, llm))
        builder.add_node("aggregator", lambda state: aggregator_agent(state, llm))
        builder.add_node("validator", lambda state: validator_agent(state, llm))
        builder.add_node("formatter", lambda state: formatter_node(state, llm))

        # 2. Add Flow Edges
        builder.add_edge(START, "supervisor")

        # Fan-out to parallel extraction agents
        builder.add_edge("supervisor", "medicine_agent")
        builder.add_edge("supervisor", "route_agent")
        builder.add_edge("supervisor", "duration_frequency_agent")
        builder.add_edge("supervisor", "instruction_agent")

        # Fan-in to aggregator
        builder.add_edge("medicine_agent", "aggregator")
        builder.add_edge("route_agent", "aggregator")
        builder.add_edge("duration_frequency_agent", "aggregator")
        builder.add_edge("instruction_agent", "aggregator")

        # Aggregator to Validator
        builder.add_edge("aggregator", "validator")

        # 3. Conditional Feedback Loop (max 3 reps)
        def should_continue_or_correct(state: AgenticRxState) -> str:
            status = state.get("validation_status", "VALID")
            iteration = state.get("iteration_count", 1)
            if status == "NEEDS_CORRECTION" and iteration < 3:
                return "supervisor"
            return "formatter"

        builder.add_conditional_edges(
            "validator",
            should_continue_or_correct,
            {
                "supervisor": "supervisor",
                "formatter": "formatter",
            }
        )

        builder.add_edge("formatter", END)

        return builder.compile()

    except ImportError:
        # Graceful pure-python runner fallback if langgraph is being installed
        return None


def run_graph_extraction(llm: Any, input_text: str) -> Tuple[str, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes the multi-agent graph extraction pipeline on the input prescription.
    Returns:
        (final_output_text, generation_time_sec, agent_logs, aggregated_blocks)
    """
    clean_input = input_text.strip().strip('"').strip("'").strip()
    initial_state: AgenticRxState = {
        "input_text": clean_input,
        "iteration_count": 0,
        "agent_logs": [],
        "validation_feedback": {},
    }

    t_start = time.perf_counter()

    compiled_graph = _build_langgraph(llm)

    if compiled_graph is not None:
        try:
            final_state = compiled_graph.invoke(initial_state)
        except Exception:
            # If graph invocation encounters issues, execute the graph flow directly
            final_state = _execute_flow_manually(llm, initial_state)
    else:
        final_state = _execute_flow_manually(llm, initial_state)

    t_end = time.perf_counter()
    generation_time = round(t_end - t_start, 3)

    output_text = final_state.get("final_output", "")
    agent_logs = final_state.get("agent_logs", [])
    aggregated_blocks = final_state.get("aggregated_blocks", [])

    return output_text, generation_time, agent_logs, aggregated_blocks


def _execute_flow_manually(llm: Any, initial_state: AgenticRxState) -> AgenticRxState:
    """Sequential direct execution of the multi-agent graph flow with loop cap."""
    state = dict(initial_state)

    while True:
        # 1. Supervisor
        sup_out = supervisor_node(state, llm)
        state.update(sup_out)

        # 2. Parallel extractors
        med_out = medicine_strength_agent(state, llm)
        state.update(med_out)

        route_out = route_agent(state, llm)
        state.update(route_out)

        df_out = duration_frequency_agent(state, llm)
        state.update(df_out)

        inst_out = instruction_agent(state, llm)
        state.update(inst_out)

        # 3. Aggregator
        agg_out = aggregator_agent(state, llm)
        state.update(agg_out)

        # 4. Validator
        val_out = validator_agent(state, llm)
        state.update(val_out)

        # Check loop condition (max 3 reps)
        if state.get("validation_status") == "NEEDS_CORRECTION" and state.get("iteration_count", 1) < 3:
            continue
        else:
            break

    # 5. Formatter
    form_out = formatter_node(state, llm)
    state.update(form_out)

    return state
