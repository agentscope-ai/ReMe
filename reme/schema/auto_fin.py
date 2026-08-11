"""Public contracts for the Auto Fin workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AutoFinModel(BaseModel):
    """Strict program-owned Auto Fin data."""

    model_config = ConfigDict(extra="forbid")


class AutoFinAgentModel(AutoFinModel):
    """Agent output tolerant of harmless extra fields."""

    model_config = ConfigDict(extra="ignore")


class AutoFinEventReference(AutoFinAgentModel):
    """One current news item related to an ETF."""

    news_id: str
    reason: str


class AutoFinEtfSelection(AutoFinAgentModel):
    """One ETF selected from the configured codes."""

    etf_code: str
    etf_name: str = ""
    events: list[AutoFinEventReference] = Field(default_factory=list)


class AutoFinEtfsOutput(AutoFinAgentModel):
    """Selections returned by the first Agent."""

    etfs: list[AutoFinEtfSelection] = Field(default_factory=list)


class AutoFinHistoricalReference(AutoFinAgentModel):
    """One historical event selected by the second Agent."""

    news_id: str
    reason: str
    direction: Literal["same", "opposite"]


class AutoFinHistoricalOutput(AutoFinAgentModel):
    """Historical matches for one current news item."""

    historical_events: list[AutoFinHistoricalReference] = Field(default_factory=list)


class AutoFinReturns(AutoFinModel):
    """Adjusted cumulative ETF returns after one historical event."""

    d1: float | None = None
    d2: float | None = None
    d3: float | None = None
    d5: float | None = None


class AutoFinHistoricalEvent(AutoFinModel):
    """A resolved historical event and its observed ETF performance."""

    news_id: str
    event_time: datetime
    title: str
    content: str
    reason: str
    direction: Literal["same", "opposite"]
    returns: AutoFinReturns


class AutoFinCurrentEvent(AutoFinModel):
    """One current event with comparable historical evidence."""

    news_id: str
    event_time: datetime
    title: str
    content: str
    reason: str
    historical_events: list[AutoFinHistoricalEvent] = Field(default_factory=list)


class AutoFinEtfAnalysis(AutoFinModel):
    """All evidence prepared for the final Agent for one ETF."""

    etf_code: str
    etf_name: str
    events: list[AutoFinCurrentEvent] = Field(default_factory=list)


class AutoFinReportOutput(AutoFinAgentModel):
    """Final Markdown returned by the third Agent."""

    title: str = ""
    description: str = ""
    body: str = ""
