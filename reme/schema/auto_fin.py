"""Public contracts for the Auto Fin workflow."""

from __future__ import annotations

from datetime import date, datetime
from math import isclose
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _shanghai_local_time(value):
    """Normalize aware input to naive Shanghai wall-clock time."""
    if not isinstance(value, (str, datetime)):
        return value
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        parsed = parsed.astimezone(_SHANGHAI).replace(tzinfo=None)
    return parsed


ShanghaiDateTime = Annotated[datetime, BeforeValidator(_shanghai_local_time)]


class AutoFinModel(BaseModel):
    """Strict base for Agent output."""

    model_config = ConfigDict(extra="forbid")


class AutoFinEtfEventReference(AutoFinModel):
    """One selected news item and why it is relevant to an ETF."""

    reason: str
    news_id: str

    @model_validator(mode="after")
    def non_empty_values(self) -> "AutoFinEtfEventReference":
        """Reject blank event references."""
        self.reason = self.reason.strip()
        self.news_id = self.news_id.strip()
        if not self.reason or not self.news_id:
            raise ValueError("ETF event reason and news ID must be non-empty")
        return self


class AutoFinSelectedEvent(AutoFinModel):
    """A selected current event with its source news reference."""

    event_time: ShanghaiDateTime
    event_content: str
    reason: str
    news_id: str
    event_title: str = ""


class AutoFinEtfSelection(AutoFinModel):
    """One liquid ETF selected for current news."""

    etf_code: str
    etf_name: str
    events: list[AutoFinEtfEventReference] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_news_ids(self) -> "AutoFinEtfSelection":
        """Reject blank or duplicate news references."""
        news_ids = [event.news_id for event in self.events]
        if len(news_ids) != len(set(news_ids)):
            raise ValueError("ETF event news IDs must be unique")
        return self


class AutoFinEtfsOutput(AutoFinModel):
    """Liquid ETFs related to current news, deduplicated by name and theme."""

    etfs: list[AutoFinEtfSelection] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_etfs(self) -> "AutoFinEtfsOutput":
        """Reject duplicate ETF codes or names."""
        codes = [item.etf_code.strip().upper() for item in self.etfs]
        names = [item.etf_name.strip().casefold() for item in self.etfs]
        if any(not code for code in codes) or any(not name for name in names):
            raise ValueError("ETF codes and names must be non-empty")
        if len(codes) != len(set(codes)) or len(names) != len(set(names)):
            raise ValueError("ETF codes and names must be unique")
        return self


class AutoFinHistoricalEvent(AutoFinModel):
    """One historical event found in ReMe memory."""

    event_time: ShanghaiDateTime
    event_content: str
    source_path: str


class AutoFinEtfHistoricalEvents(AutoFinModel):
    """Historical events returned by the search Agent before market calculation."""

    etf_code: str
    etf_name: str
    historical_events: list[AutoFinHistoricalEvent] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AutoFinDailyEntry(AutoFinModel):
    """First daily open or close that can be traded after an event."""

    entry_time: ShanghaiDateTime
    trade_date: date
    price_type: Literal["open", "close"]
    raw_price: float = Field(gt=0)
    adj_factor: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_entry_timestamp(self) -> "AutoFinDailyEntry":
        """Require a Shanghai-local timestamp matching the daily price."""
        if self.entry_time.date() != self.trade_date:
            raise ValueError("entry time and trade date must match")
        expected_clock = (9, 30) if self.price_type == "open" else (15, 0)
        if (
            (self.entry_time.hour, self.entry_time.minute) != expected_clock
            or self.entry_time.second
            or self.entry_time.microsecond
        ):
            raise ValueError(f"{self.price_type} entry time must use the official daily price timestamp")
        return self


class AutoFinFutureReturnPoint(AutoFinModel):
    """Cumulative adjusted return at one future valid close."""

    horizon: int = Field(ge=1, le=10)
    trade_date: date
    raw_close: float = Field(gt=0)
    adj_factor: float = Field(gt=0)
    cumulative_return: float


class AutoFinMarketSample(AutoFinModel):
    """Daily adjusted ETF returns following one historical event."""

    event_time: ShanghaiDateTime
    entry: AutoFinDailyEntry | None = None
    future_returns: list[AutoFinFutureReturnPoint] = Field(default_factory=list, max_length=10)
    reaction_summary: str

    @model_validator(mode="after")
    def valid_daily_return_path(self) -> "AutoFinMarketSample":
        """Reject look-ahead entries and inconsistent adjusted returns."""
        if self.entry is None:
            if self.future_returns:
                raise ValueError("future returns require an entry")
            return self
        if self.entry.entry_time <= self.event_time:
            raise ValueError("entry must be strictly after the event")

        expected_horizons = list(range(1, len(self.future_returns) + 1))
        if [point.horizon for point in self.future_returns] != expected_horizons:
            raise ValueError("future return horizons must be contiguous and start at 1")
        trade_dates = [point.trade_date for point in self.future_returns]
        if trade_dates != sorted(set(trade_dates)):
            raise ValueError("future return trade dates must be unique and ascending")
        if trade_dates:
            first_trade_date = trade_dates[0]
            if self.entry.price_type == "open" and first_trade_date < self.entry.trade_date:
                raise ValueError("an open entry cannot use an earlier close")
            if self.entry.price_type == "close" and first_trade_date <= self.entry.trade_date:
                raise ValueError("a close entry requires a later close")

        adjusted_entry = self.entry.raw_price * self.entry.adj_factor
        for point in self.future_returns:
            expected_return = point.raw_close * point.adj_factor / adjusted_entry - 1
            if not isclose(point.cumulative_return, expected_return, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"incorrect adjusted return at horizon {point.horizon}")
        return self


class AutoFinEtfHistoricalResearch(AutoFinModel):
    """Historical events and their calculated ETF return paths."""

    etf_code: str
    etf_name: str
    historical_events: list[AutoFinHistoricalEvent] = Field(default_factory=list)
    historical_samples: list[AutoFinMarketSample] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def samples_match_events(self) -> "AutoFinEtfHistoricalResearch":
        """Require one ordered calculated sample for every historical event."""
        if [event.event_time for event in self.historical_events] != [
            sample.event_time for sample in self.historical_samples
        ]:
            raise ValueError("historical samples must correspond one-to-one with historical events")
        return self


class AutoFinHistoricalMatch(AutoFinModel):
    """One historical event selected for the weighted forecast."""

    event_time: ShanghaiDateTime
    similarity: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    reason: str


class AutoFinForecastReturnPoint(AutoFinModel):
    """Weighted expected cumulative return for one holding horizon."""

    horizon: int = Field(ge=1, le=10)
    expected_return: float | None = None


class AutoFinWeightedForecast(AutoFinModel):
    """Agent-calculated forecast derived from similar historical events."""

    returns: list[AutoFinForecastReturnPoint] = Field(min_length=10, max_length=10)
    suggested_holding_days: int | None = Field(default=None, ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    @model_validator(mode="after")
    def complete_horizons(self) -> "AutoFinWeightedForecast":
        """Require one ordered forecast point for every D1-D10 horizon."""
        if [point.horizon for point in self.returns] != list(range(1, 11)):
            raise ValueError("forecast horizons must be ordered D1-D10")
        return self


class AutoFinSelectedEtfAnalysis(AutoFinModel):
    """Weighted forecast for one selected ETF."""

    etf_code: str
    etf_name: str
    matched_historical_events: list[AutoFinHistoricalMatch] = Field(default_factory=list)
    forecast: AutoFinWeightedForecast
    calculation_code: str
    summary: str
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_historical_weights(self) -> "AutoFinSelectedEtfAnalysis":
        """Reject duplicate matches and invalid normalized weights."""
        event_times = [event.event_time for event in self.matched_historical_events]
        if len(event_times) != len(set(event_times)):
            raise ValueError("matched historical events must be unique")
        if event_times and not isclose(
            sum(event.weight for event in self.matched_historical_events),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("matched historical event weights must sum to 1")
        if not self.calculation_code.strip():
            raise ValueError("calculation code must not be empty")
        return self


class AutoFinEtfHistoryDetail(AutoFinModel):
    """Complete historical research and market result for one selected ETF."""

    etf: AutoFinEtfSelection
    current_events: list[AutoFinSelectedEvent] = Field(min_length=1)
    historical_research: AutoFinEtfHistoricalResearch
    market_analysis: AutoFinSelectedEtfAnalysis

    @model_validator(mode="after")
    def consistent_etf_and_events(self) -> "AutoFinEtfHistoryDetail":
        """Reject stale or cross-ETF outputs from dispatched steps."""
        identity = (self.etf.etf_code, self.etf.etf_name)
        if (self.historical_research.etf_code, self.historical_research.etf_name) != identity:
            raise ValueError("historical research ETF must match the selected ETF")
        if (self.market_analysis.etf_code, self.market_analysis.etf_name) != identity:
            raise ValueError("market analysis ETF must match the selected ETF")
        return self


class AutoFinReportOutput(AutoFinModel):
    """Final report, recommendation, and delivery summary for all selected ETFs."""

    title: str
    description: str
    body: str
    final_recommendation: str
    concise_summary: str
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def non_empty_report(self) -> "AutoFinReportOutput":
        """Require every human-readable report representation."""
        for field in ("title", "description", "body", "final_recommendation", "concise_summary"):
            value = getattr(self, field).strip()
            if not value:
                raise ValueError(f"{field} must not be empty")
            setattr(self, field, value)
        return self
