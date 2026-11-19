
from __future__ import annotations

from datetime import datetime, timezone
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
    max_iterations: int = 8,
) -> InvestigationState:
    """Create a fresh investigation state."""

    return InvestigationState(
        investigation_id=datetime.now(timezone.utc).isoformat(),
        user_question=question,
        objectives=[],
        investigation_scope={},
        selected_agents=[],
        execution_queue=[],
        current_agent=None,
        traffic_findings=[],
        predictive_findings=[],
        geospatial_findings=[],
        network_findings=[],
        tool_history=[],
        agent_history=[],
        iteration_count=0,
        max_iterations=max_iterations,
        status="planning",
        next_action="plan",
        supervisor_notes=[],
        final_report=None,
        errors=[],
    )
