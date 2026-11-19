
from __future__ import annotations

from datetime import UTC, datetime
from operator import add
from typing import Annotated, Literal, TypedDict

AgentName = Literal[
    "traffic",
    "predictive",
    "geospatial",
    "network",
]

InvestigationStatus = Literal[
    "planning",
    "running",
    "completed",
    "failed",
]


class AgentFinding(TypedDict, total=False):
    """Structured result produced by a specialist agent."""

    agent: AgentName
    summary: str
    observations: list[str]
    evidence: list[dict]
    conclusions: list[str]
    limitations: list[str]
    confidence: float
    created_at: str


class ToolExecution(TypedDict, total=False):
    """Traceable record of an analytical tool execution."""

    tool_name: str
    agent: str
    arguments: dict
    result_summary: str
    success: bool
    error: str
    executed_at: str


class InvestigationState(TypedDict, total=False):
    # Request context
    investigation_id: str
    user_question: str
    objectives: list[str]
    investigation_scope: dict

    # Execution plan
    selected_agents: list[AgentName]
    execution_queue: list[AgentName]
    current_agent: AgentName | None

    # Specialist findings
    traffic_findings: list[AgentFinding]
    predictive_findings: list[AgentFinding]
    geospatial_findings: list[AgentFinding]
    network_findings: list[AgentFinding]

    # Execution history
    tool_history: Annotated[list[ToolExecution], add]
    agent_history: Annotated[list[AgentName], add]
    tool_calls_count: int
    max_tool_calls: int

    # Investigation control
    iteration_count: int
    max_iterations: int
    status: InvestigationStatus
    next_action: str
    supervisor_notes: Annotated[list[str], add]

    # Final output
    final_report: dict | None
    errors: Annotated[list[str], add]


def create_initial_state(
    question: str,
    *,
    investigation_scope: dict | None = None,
    max_iterations: int = 4,
    max_tool_calls: int = 12,
) -> InvestigationState:
    if not question.strip():
        raise ValueError("The investigation question cannot be empty.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1.")

    return {
        "investigation_id": datetime.now(UTC).isoformat(),
        "user_question": question,
        "objectives": [],
        "investigation_scope": investigation_scope or {},
        "selected_agents": [],
        "current_agent": None,
        "agent_history": [],
        "traffic_findings": [],
        "predictive_findings": [],
        "geospatial_findings": [],
        "network_findings": [],
        "tool_history": [],
        "tool_calls_count": 0,
        "max_tool_calls": max_tool_calls,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "supervisor_notes": [],
        "errors": [],
        "status": "planning",
        "final_report": None,
    }
