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
    """Strict base for program-owned Auto Fin data."""

    model_config = ConfigDict(extra="forbid")


class AutoFinAgentModel(AutoFinModel):
    """Tolerant base for raw Agent output."""

    model_config = ConfigDict(extra="ignore")


class AutoFinEtfEventReference(AutoFinAgentModel):
    """One selected news item and why it is relevant to an ETF."""

    reason: str
    news_id: str


class AutoFinSelectedEvent(AutoFinModel):
    """A selected current event with its source news reference."""

    event_time: ShanghaiDateTime
    event_content: str
    reason: str
    news_id: str
    event_title: str = ""


class AutoFinEtfSelection(AutoFinAgentModel):
    """One ETF selection returned by the Topic Agent."""

    etf_code: str
    etf_name: str = ""
    events: list[AutoFinEtfEventReference] = Field(default_factory=list)


class AutoFinEtfsOutput(AutoFinAgentModel):
    """ETF selections returned by the Topic Agent before normalization."""

    etfs: list[AutoFinEtfSelection] = Field(default_factory=list)


class AutoFinHistoricalEventReference(AutoFinAgentModel):
    """One historical news item selected by the search Agent."""

    reason: str
    news_id: str
    source_path: str = ""


class AutoFinEtfHistoricalEvents(AutoFinAgentModel):
    """Historical news references returned by the search Agent."""

    etf_code: str = ""
    etf_name: str = ""
    historical_events: list[AutoFinHistoricalEventReference] = Field(default_factory=list)


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


class AutoFinHistoricalEvent(AutoFinModel):
    """One resolved historical news item and its calculated ETF return path."""

    reason: str
    news_id: str
    source_path: str
    event_time: ShanghaiDateTime
    event_title: str
    event_content: str
    market_entry: AutoFinDailyEntry | None = None
    future_returns: list[AutoFinFutureReturnPoint] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def valid_historical_event(self) -> "AutoFinHistoricalEvent":
        """Require source identity and validate the embedded market reaction."""
        for field in ("reason", "news_id", "source_path", "event_title", "event_content"):
            value = getattr(self, field).strip()
            if not value:
                raise ValueError(f"historical event {field} must not be empty")
            setattr(self, field, value)
        AutoFinMarketSample(
            event_time=self.event_time,
            entry=self.market_entry,
            future_returns=self.future_returns,
            reaction_summary="",
        )
        return self


class AutoFinEtfHistoricalResearch(AutoFinModel):
    """Resolved historical events with embedded calculated ETF return paths."""

    etf_code: str
    etf_name: str
    historical_events: list[AutoFinHistoricalEvent] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_historical_news(self) -> "AutoFinEtfHistoricalResearch":
        """Reject duplicate resolved source records."""
        news_ids = [event.news_id for event in self.historical_events]
        if len(news_ids) != len(set(news_ids)):
            raise ValueError("historical event news IDs must be unique")
        return self


class AutoFinHistoricalDirectionReference(AutoFinAgentModel):
    """One direction-classified historical event returned by the Market Agent."""

    reason: str
    news_id: str


class AutoFinMarketSelection(AutoFinAgentModel):
    """Same- and opposite-direction historical events returned by the Market Agent."""

    same_direction_events: list[AutoFinHistoricalDirectionReference] = Field(default_factory=list)
    opposite_direction_events: list[AutoFinHistoricalDirectionReference] = Field(default_factory=list)


class AutoFinHistoricalMatch(AutoFinModel):
    """One direction-classified historical event used by the equal-weight forecast."""

    reason: str
    news_id: str
    event_time: ShanghaiDateTime
    direction: Literal["same", "opposite"]
    weight: float = Field(ge=0.0, le=1.0)


class AutoFinForecastReturnPoint(AutoFinModel):
    """Weighted expected cumulative return for one holding horizon."""

    horizon: int = Field(ge=1, le=10)
    expected_return: float | None = None


class AutoFinWeightedForecast(AutoFinModel):
    """Program-calculated forecast derived from similar historical events."""

    returns: list[AutoFinForecastReturnPoint] = Field(min_length=10, max_length=10)
    suggested_holding_days: int | None = Field(default=None, ge=1, le=10)

    @model_validator(mode="after")
    def complete_horizons(self) -> "AutoFinWeightedForecast":
        """Require one ordered forecast point for every D1-D10 horizon."""
        if [point.horizon for point in self.returns] != list(range(1, 11)):
            raise ValueError("forecast horizons must be ordered D1-D10")
        return self


class AutoFinSelectedEtfAnalysis(AutoFinModel):
    """Program-calculated weighted forecast for one selected ETF."""

    etf_code: str
    etf_name: str
    matched_historical_events: list[AutoFinHistoricalMatch] = Field(default_factory=list)
    forecast: AutoFinWeightedForecast
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_historical_weights(self) -> "AutoFinSelectedEtfAnalysis":
        """Reject duplicate matches and invalid normalized weights."""
        news_ids = [event.news_id for event in self.matched_historical_events]
        if len(news_ids) != len(set(news_ids)):
            raise ValueError("matched historical events must be unique")
        if news_ids and not isclose(
            sum(event.weight for event in self.matched_historical_events),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("matched historical event weights must sum to 1")
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


class AutoFinReportOutput(AutoFinAgentModel):
    """Final Markdown title and body for all selected ETFs."""

    title: str = ""
    body: str = ""
