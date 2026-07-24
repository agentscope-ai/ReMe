"""Public contracts for the Auto Fin news-case workflow."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutoFinModel(BaseModel):
    """Strict base for generated and persisted Auto Fin data."""

    model_config = ConfigDict(extra="forbid")


class AutoFinHistoricalCase(AutoFinModel):
    """A prior case returned by ReMe memory search."""

    trade_date: date
    source_path: str
    summary: str


class AutoFinThemePlan(AutoFinModel):
    """One current-news theme and its representative domestic ETF."""

    theme: str
    direction: Literal["POSITIVE", "NEGATIVE", "MIXED"]
    news_ids: list[str]
    etf_code: str
    etf_name: str
    memory_query: str
    historical_cases: list[AutoFinHistoricalCase] = Field(default_factory=list)


class AutoFinResearchPlan(AutoFinModel):
    """Themes selected after current-news analysis and memory retrieval."""

    themes: list[AutoFinThemePlan] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_themes_and_etfs(self) -> "AutoFinResearchPlan":
        """Reject duplicate themes and representative ETFs."""
        for values in (
            [item.theme for item in self.themes],
            [item.etf_code for item in self.themes],
        ):
            if len(values) != len(set(values)):
                raise ValueError("themes and representative ETF codes must be unique")
        return self


class AutoFinRecommendation(AutoFinModel):
    """A reference suggestion independent of portfolio state."""

    theme: str
    etf_code: str
    etf_name: str
    action: Literal["BUY", "SELL", "HOLD"]
    price_in: Literal["YES", "NO", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    historical_evidence: str
    invalidation_condition: str


class AutoFinDecisionOutput(AutoFinModel):
    """Final Markdown analysis and structured case decisions."""

    description: str
    body: str
    recommendations: list[AutoFinRecommendation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_recommendations(self) -> "AutoFinDecisionOutput":
        """Keep one decision for each planned theme and ETF."""
        keys = [(item.theme, item.etf_code) for item in self.recommendations]
        if len(keys) != len(set(keys)):
            raise ValueError("recommendations must be unique by theme and ETF")
        return self
