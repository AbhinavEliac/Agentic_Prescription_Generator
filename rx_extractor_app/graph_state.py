"""
graph_state.py
--------------
State models and TypedDict schemas for the LangGraph Multi-Agent
Prescription Extraction Graph.
"""
from typing import TypedDict, List, Dict, Any, Optional


class MedicineItem(TypedDict, total=False):
    medicine_id: int
    drug_name: str
    strength: str


class RouteItem(TypedDict, total=False):
    medicine_id: int
    drug_name: str
    route: str


class DurationFrequencyItem(TypedDict, total=False):
    medicine_id: int
    drug_name: str
    frequency: str
    duration: str


class InstructionItem(TypedDict, total=False):
    medicine_id: int
    drug_name: str
    instruction: str
    additional_instruction: str


class PrescriptionBlock(TypedDict):
    Drug_name: str
    strength: str
    frequency: str
    duration: str
    route: str
    instruction: str
    additional_instruction: str


class ValidationFeedback(TypedDict, total=False):
    medicine_agent: str
    route_agent: str
    duration_frequency_agent: str
    instruction_agent: str


class AgenticRxState(TypedDict, total=False):
    # Prescription input
    input_text: str
    system_prompt: Optional[str]

    # Iteration tracker (capped at max 3)
    iteration_count: int

    # Agent outputs from parallel extraction nodes
    medicines: List[MedicineItem]
    routes: List[RouteItem]
    durations_frequencies: List[DurationFrequencyItem]
    instructions: List[InstructionItem]

    # Aggregator output
    aggregated_blocks: List[PrescriptionBlock]

    # Validator evaluation
    validation_status: str  # "VALID" | "NEEDS_CORRECTION"
    validation_feedback: ValidationFeedback

    # Final formatted block text
    final_output: str

    # Execution telemetry & audit logs
    agent_logs: List[Dict[str, Any]]
    generation_time: float
