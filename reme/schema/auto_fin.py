"""Public contracts for the Auto Fin workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutoFinModel(BaseModel):
    """Strict base for Agent output."""

    model_config = ConfigDict(extra="forbid")


class AutoFinEvent(AutoFinModel):
    """One material development in a topic."""

    event_time: datetime
    event_content: str


class AutoFinTopicsOutput(AutoFinModel):
    """Current news compressed into topic timelines."""

    topics: dict[str, list[AutoFinEvent]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def non_empty_topics(self) -> "AutoFinTopicsOutput":
        """Reject blank topic names and empty timelines."""
        if any(not topic.strip() or not events for topic, events in self.topics.items()):
            raise ValueError("topic names and event timelines must be non-empty")
        return self


class AutoFinReturnCurve(AutoFinModel):
    """Cumulative return from the pre-event baseline to each close."""

    d1_return: float | None = None
    d1_d2_return: float | None = None
    d1_d3_return: float | None = None
    d1_d4_return: float | None = None
    d1_d5_return: float | None = None


class AutoFinHistoricalEvent(AutoFinModel):
    """One historical event found in ReMe memory."""

    event_time: datetime
    event_content: str
    source_path: str


class AutoFinHistoricalResearch(AutoFinModel):
    """Historical events found for one current topic."""

    topic: str
    historical_events: list[AutoFinHistoricalEvent] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AutoFinMarketSample(AutoFinModel):
    """Actual ETF reaction following one historical event."""

    event_time: datetime
    baseline_time: datetime
    baseline_price: float = Field(gt=0)
    intraday_returns: list["AutoFinIntradayPoint"] = Field(default_factory=list)
    reaction_summary: str
    returns: AutoFinReturnCurve


class AutoFinIntradayPoint(AutoFinModel):
    """Return at one completed 15-minute checkpoint."""

    bar_time: datetime
    return_from_baseline: float


class AutoFinForecast(AutoFinModel):
    """Current-event forecast and holding-time suggestion."""

    anchor_event_time: datetime
    baseline_time: datetime
    baseline_price: float | None = Field(default=None, gt=0)
    returns: AutoFinReturnCurve
    suggested_holding_period: Literal["D1", "D1-D2", "D1-D3", "D1-D4", "D1-D5", "不建议持有"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    exit_condition: str
    invalidation_condition: str


class AutoFinEtfAnalysis(AutoFinModel):
    """Historical evidence and forecast for one directly related ETF."""

    etf_code: str
    etf_name: str
    asset_type: str
    market: str
    relationship: str
    current_intraday_returns: list[AutoFinIntradayPoint] = Field(default_factory=list)
    historical_samples: list[AutoFinMarketSample] = Field(default_factory=list)
    forecast: AutoFinForecast


class AutoFinTopicAnalysis(AutoFinModel):
    """Independent Agent analysis for one current topic."""

    topic: str
    historical_events: list[AutoFinHistoricalEvent] = Field(default_factory=list)
    etfs: list[AutoFinEtfAnalysis] = Field(min_length=1, max_length=5)
    summary: str
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_etfs(self) -> "AutoFinTopicAnalysis":
        """Keep one result per ETF."""
        codes = [etf.etf_code.upper() for etf in self.etfs]
        if len(codes) != len(set(codes)):
            raise ValueError("ETF codes must be unique within a topic")
        return self


class AutoFinReportOutput(AutoFinModel):
    """Final Markdown content generated from all topic analyses."""

    title: str
    description: str
    body: str
    limitations: list[str] = Field(default_factory=list)
