"""Public contracts for the Auto Fin workflow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AutoFinModel(BaseModel):
    """Strict program-owned Auto Fin data."""

    model_config = ConfigDict(extra="forbid")


class AutoFinAgentModel(AutoFinModel):
    """Agent output tolerant of harmless extra fields."""

    model_config = ConfigDict(extra="ignore")


class AutoFinReportOutput(AutoFinAgentModel):
    """Final Markdown returned by the agentic news-research Agent."""

    title: str
    description: str
    body: str
