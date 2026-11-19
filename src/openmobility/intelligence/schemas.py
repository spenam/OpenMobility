
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from openmobility.intelligence.state import AgentName


class EvidenceMetric(BaseModel):
    name: str
    value: str


class Evidence(BaseModel):
    source: str
    description: str
    dataset: str | None = None
    sensor_ids: list[str] = Field(default_factory=list)
    metrics: list[EvidenceMetric] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_metrics(cls, data: Any) -> Any:
        """Accept the metric dictionaries produced by existing runners."""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        metrics = data.get("metrics", {})

        if isinstance(metrics, dict):
            data["metrics"] = [
                {
                    "name": str(name),
                    "value": (
                        value if isinstance(value, str)
                        else json.dumps(value, default=str)
                    ),
                }
                for name, value in metrics.items()
            ]

        return data


class SpecialistFinding(BaseModel):
    agent: AgentName
    summary: str
    observations: list[str] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["completed", "failed"] = "completed"
    error: str | None = None


class SupervisorDecision(BaseModel):
    objectives: list[str] = Field(default_factory=list)
    selected_agents: list[AgentName] = Field(default_factory=list)
    next_agent: AgentName | None = None
    investigation_complete: bool
    reasoning: str


class InvestigationPlan(BaseModel):
    objectives: list[str]
    selected_agents: list[AgentName]
    execution_order: list[AgentName]
    reasoning: str


class IntelligenceReport(BaseModel):
    executive_summary: str
    current_conditions: list[str] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    historical_comparisons: list[str] = Field(default_factory=list)
    forecasts: list[str] = Field(default_factory=list)
    geographic_observations: list[str] = Field(default_factory=list)
    network_observations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
