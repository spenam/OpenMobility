from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable, Mapping

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from openmobility.intelligence.schemas import (
    AgentName,
    IntelligenceReport,
    SpecialistFinding,
    SupervisorDecision,
)
from openmobility.intelligence.state import (
    InvestigationState,
    ToolExecution,
)


AGENTS: tuple[AgentName, ...] = (
    "traffic",
    "predictive",
    "geospatial",
    "network",
)

FINDING_KEYS = {
    "traffic": "traffic_findings",
    "predictive": "predictive_findings",
    "geospatial": "geospatial_findings",
    "network": "network_findings",
}


class ToolBudgetExceeded(RuntimeError):
    """Raised when an investigation reaches its tool-call limit."""


class ToolBudget:
    """Per-specialist access to the investigation's remaining tool budget."""

    def __init__(
        self,
        *,
        agent: AgentName,
        used: int,
        maximum: int,
    ) -> None:
        self.agent = agent
        self.used = used
        self.maximum = maximum
        self.history: list[ToolExecution] = []

    def call(
        self,
        tool_name: str,
        function: Callable,
        /,
        *args,
        arguments: dict | None = None,
        result_summary: Callable[[object], str] | None = None,
        **kwargs,
    ):
        if self.used >= self.maximum:
            raise ToolBudgetExceeded(
                f"Investigation tool limit reached ({self.maximum})."
            )

        self.used += 1
        record: ToolExecution = {
            "tool_name": tool_name,
            "agent": self.agent,
            "arguments": arguments or {},
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = function(*args, **kwargs)
            record["success"] = True

            if result_summary is not None:
                record["result_summary"] = result_summary(result)
            else:
                record["result_summary"] = (
                    f"Completed {tool_name}."
                )

            self.history.append(record)
            return result

        except Exception as exc:
            record["success"] = False
            record["error"] = str(exc)
            self.history.append(record)
            raise


# Each runner receives the current investigation state and its tool budget.
AgentRunner = Callable[
    [InvestigationState, ToolBudget],
    SpecialistFinding,
]


def _findings_as_dict(state: InvestigationState) -> dict:
    result = {}

    for key in FINDING_KEYS.values():
        result[key] = [
            finding.model_dump()
            if hasattr(finding, "model_dump")
            else finding
            for finding in state.get(key, [])
        ]

    return result


def _supervisor_context(state: InvestigationState) -> str:
    context = {
        "question": state["user_question"],
        "scope": state.get("investigation_scope", {}),
        "objectives": state.get("objectives", []),
        "selected_agents": state.get("selected_agents", []),
        "agents_already_run": state.get("agent_history", []),
        "findings": _findings_as_dict(state),
        "iterations_used": state.get("iteration_count", 0),
        "iteration_limit": state.get("max_iterations", 4),
        "tool_calls_used": state.get("tool_calls_count", 0),
        "tool_call_limit": state.get("max_tool_calls", 12),
    }

    return json.dumps(context, default=str, indent=2)


def _report_context(state: InvestigationState) -> str:
    return json.dumps(
        {
            "question": state["user_question"],
            "scope": state.get("investigation_scope", {}),
            "findings": _findings_as_dict(state),
            "tool_history": state.get("tool_history", []),
            "errors": state.get("errors", []),
        },
        default=str,
        indent=2,
    )


def build_intelligence_graph(
    agent_runners: Mapping[AgentName, AgentRunner],
    *,
    llm=None,
    model: str = "gpt-4.1-mini",
):
    """
    Build and compile the sequential multi-agent investigation graph.

    The caller supplies the four specialist implementations. This keeps
    orchestration independent from the project's data-processing functions.
    """
    if llm is None:
        llm = ChatOpenAI(model=model, temperature=0)

    supervisor_model = llm.with_structured_output(SupervisorDecision)
    report_model = llm.with_structured_output(IntelligenceReport)

    def supervisor_node(
        state: InvestigationState,
    ) -> dict:
        iterations = state.get("iteration_count", 0)
        tool_calls = state.get("tool_calls_count", 0)

        if (
            iterations >= state.get("max_iterations", 4)
            or tool_calls >= state.get("max_tool_calls", 12)
        ):
            return {
                "current_agent": None,
                "status": "reporting",
                "supervisor_notes": [
                    "Stopped because an investigation limit was reached."
                ],
            }

        prompt = f"""
You are the autonomous supervisor of an urban mobility intelligence system.

You coordinate four specialist agents:
- traffic: current traffic conditions, patterns, and anomalies
- predictive: forecasts, model outputs, and prediction performance
- geospatial: sensor locations and geographic relationships
- network: road topology, connected segments, intersections, and
  connectivity patterns

Choose at most ONE next specialist per decision.

Rules:
1. Use only the internal OpenMobility datasets and tools.
2. Choose specialists based on the user's question and findings so far.
3. Do not repeat a specialist that has already run.
4. Do not claim a calculation or data lookup happened unless it appears
   in the findings or tool history.
5. Stop when the question has been adequately investigated, or when
   the iteration/tool budget is exhausted.
6. If evidence is insufficient, say so rather than inventing results.
7. Do not treat correlation or adjacent congestion as proof of causation.

Return:
- objectives: the investigation objectives
- selected_agents: the specialists relevant to this investigation
- next_agent: one eligible specialist, or null
- investigation_complete: true when no more specialist work is needed
- reasoning: a concise explanation of your decision

Investigation state:
{_supervisor_context(state)}
"""

        try:
            decision: SupervisorDecision = supervisor_model.invoke(prompt)
        except Exception as exc:
            return {
                "current_agent": None,
                "status": "failed",
                "errors": [
                    *state.get("errors", []),
                    f"Supervisor decision failed: {exc}",
                ],
            }

        selected = list(state.get("selected_agents", []))

        for agent in decision.selected_agents:
            if agent in AGENTS and agent not in selected:
                selected.append(agent)

        eligible = [
            agent
            for agent in AGENTS
            if agent not in state.get("agent_history", [])
        ]

        next_agent = decision.next_agent

        if (
            decision.investigation_complete
            or next_agent is None
            or next_agent not in eligible
        ):
            return {
                "objectives": decision.objectives,
                "selected_agents": selected,
                "current_agent": None,
                "status": "reporting",
                "supervisor_notes": [
                    *state.get("supervisor_notes", []),
                    decision.reasoning,
                ],
            }

        if next_agent not in selected:
            selected.append(next_agent)

        return {
            "objectives": decision.objectives,
            "selected_agents": selected,
            "current_agent": next_agent,
            "status": "running",
            "supervisor_notes": [
                *state.get("supervisor_notes", []),
                decision.reasoning,
            ],
        }

    def make_specialist_node(agent: AgentName):
        finding_key = FINDING_KEYS[agent]

        def specialist_node(state: InvestigationState) -> dict:
            budget = ToolBudget(
                agent=agent,
                used=state.get("tool_calls_count", 0),
                maximum=state.get("max_tool_calls", 12),
            )

            try:
                runner = agent_runners.get(agent)

                if runner is None:
                    raise RuntimeError(
                        f"No specialist runner registered for '{agent}'."
                    )

                finding = runner(state, budget)

                if finding.agent != agent:
                    raise ValueError(
                        f"{agent} runner returned a "
                        f"{finding.agent} finding."
                    )

            except Exception as exc:
                finding = SpecialistFinding(
                    agent=agent,
                    summary=f"{agent.title()} investigation failed.",
                    limitations=[
                        "This specialist did not complete its analysis."
                    ],
                    confidence=0.0,
                    status="failed",
                    error=str(exc),
                )

            previous_findings = list(state.get(finding_key, []))
            previous_findings.append(finding)

            return {
                finding_key: previous_findings,
                "current_agent": None,
                "agent_history": [
                    *state.get("agent_history", []),
                    agent,
                ],
                "iteration_count": state.get("iteration_count", 0) + 1,
                "tool_calls_count": budget.used,
                "tool_history": [
                    *state.get("tool_history", []),
                    *budget.history,
                ],
                "errors": (
                    [
                        *state.get("errors", []),
                        finding.error,
                    ]
                    if finding.error
                    else state.get("errors", [])
                ),
                "status": "running",
            }

        return specialist_node

    def report_node(state: InvestigationState) -> dict:
        try:
            report: IntelligenceReport = report_model.invoke(
                f"""
Prepare a structured intelligence report using only the supplied
investigation evidence.

Do not invent metrics, sensor IDs, dates, geographic relationships,
forecasts, or causal explanations. Distinguish observed facts from
interpretations. Include limitations and uncertainty.

Map findings into:
- executive_summary
- current_conditions
- anomalies
- historical_comparisons
- forecasts
- geographic_observations
- network_observations
- evidence
- limitations
- confidence

Investigation:
{_report_context(state)}
"""
            )

            report_data = report.model_dump()

        except Exception as exc:
            # A deterministic fallback ensures the investigation can still
            # return its collected evidence if report generation fails.
            all_findings = [
                finding
                for key in FINDING_KEYS.values()
                for finding in state.get(key, [])
            ]

            report_data = IntelligenceReport(
                executive_summary=(
                    "The investigation collected specialist findings, "
                    "but automated report synthesis was unavailable."
                ),
                current_conditions=[
                    item
                    for finding in state.get("traffic_findings", [])
                    for item in finding.observations
                ],
                forecasts=[
                    item
                    for finding in state.get("predictive_findings", [])
                    for item in finding.observations
                ],
                geographic_observations=[
                    item
                    for finding in state.get("geospatial_findings", [])
                    for item in finding.observations
                ],
                network_observations=[
                    item
                    for finding in state.get("network_findings", [])
                    for item in finding.observations
                ],
                evidence=[
                    evidence
                    for finding in all_findings
                    for evidence in finding.evidence
                ],
                limitations=[
                    limitation
                    for finding in all_findings
                    for limitation in finding.limitations
                ] + [f"Report synthesis failed: {exc}"],
                confidence=0.0,
            ).model_dump()

        return {
            "final_report": report_data,
            "current_agent": None,
            "status": "completed",
        }

    graph = StateGraph(InvestigationState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("traffic", make_specialist_node("traffic"))
    graph.add_node("predictive", make_specialist_node("predictive"))
    graph.add_node("geospatial", make_specialist_node("geospatial"))
    graph.add_node("network", make_specialist_node("network"))
    graph.add_node("report", report_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda state: (
            "failed"
            if state.get("status") == "failed"
            else state.get("current_agent") or "report"
        ),
        {
            "traffic": "traffic",
            "predictive": "predictive",
            "geospatial": "geospatial",
            "network": "network",
            "report": "report",
            "failed": END,
        },
    )

    for agent in AGENTS:
        graph.add_edge(agent, "supervisor")

    graph.add_edge("report", END)

    return graph.compile()