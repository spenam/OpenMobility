
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AgentName = Literal[
    "traffic",
    "predictive",
    "geospatial",
    "network",
]


class Evidence(BaseModel):
    source: str
    description: str
    dataset: str | None = None
    sensor_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | None] = Field(
        default_factory=dict
    )


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