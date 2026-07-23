"""Public contracts for the Auto Fin simulated-portfolio workflow."""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutoFinModel(BaseModel):
    """Strict base model used by persisted Auto Fin documents."""

    model_config = ConfigDict(extra="forbid")


class Checkpoint(StrEnum):
    """Supported A-share decision checkpoints."""

    OPEN = "0900"
    MIDDAY = "1145"
    CLOSE = "1445"


class RunStatus(StrEnum):
    """Overall status of one checkpoint run."""

    COMPLETE = "COMPLETE"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ActionType(StrEnum):
    """Discrete simulated-portfolio action."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ActionStatus(StrEnum):
    """Lifecycle status for one simulated action."""

    PROPOSED = "PROPOSED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    PENDING_DATA = "PENDING_DATA"
    MISSED_CUTOFF = "MISSED_CUTOFF"


class InstrumentType(StrEnum):
    """Phase-one instruments, all handled as T+1."""

    STOCK = "stock"
    DOMESTIC_EQUITY_ETF = "domestic_equity_etf"


class DataSource(AutoFinModel):
    """Traceable source invocation used by an analysis."""

    tool: str
    endpoint: str
    fetched_at: datetime
    query_hash: str
    request_started_at: datetime | None = None
    coverage_start: datetime | date | None = None
    coverage_end: datetime | date | None = None
    row_count: int = Field(default=0, ge=0)
    max_timestamp: datetime | date | None = None
    snapshot_path: str | None = None
    content_hash: str | None = None
    used_cache: bool = False
    warnings: list[str] = Field(default_factory=list)


class EventWindow(AutoFinModel):
    """Continuous event cursor window."""

    start_exclusive: datetime
    end_inclusive: datetime


class EventSignal(AutoFinModel):
    """One de-duplicated event and its bounded interpretation."""

    event_id: str
    published_at: datetime
    fetched_at: datetime
    title: str
    industries: list[str] = Field(default_factory=list)
    codes: list[str] = Field(default_factory=list)
    dedupe_key: str
    known_before_cutoff: bool
    direction: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: str
    summary: str
    source_ref: str


class EventCursor(AutoFinModel):
    """Last processed event cursor."""

    last_event_time: datetime | None = None
    last_event_id: str = ""


class EtfScore(AutoFinModel):
    """One ETF score produced by a deterministic research dimension."""

    code: str
    name: str = ""
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0, le=100.0)
    expected_return: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    price_in: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class RankingMetrics(AutoFinModel):
    """Out-of-sample ranking diagnostics for one machine-learning model."""

    rank_ic: float | None = None
    rank_ic_ir: float | None = None
    ndcg_at_20: float | None = Field(default=None, ge=0.0, le=1.0)
    train_sample_count: int = Field(default=0, ge=0)
    validation_sample_count: int = Field(default=0, ge=0)
    validation_date_count: int = Field(default=0, ge=0)


class ExtremeAnalysis(AutoFinModel):
    """Tail-regime diagnostic attached to a ranking."""

    code: str
    direction: Literal["UP", "DOWN"]
    observed_return: float
    threshold: float = Field(gt=0.0)
    historical_sample_count: int = Field(default=0, ge=0)
    next_return_mean: float | None = None
    next_return_hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    conclusion: str


class DimensionRanking(AutoFinModel):
    """Rebuildable Top-N output for one research dimension."""

    dimension: Literal["event", "backtest", "us_correlation"]
    as_of: datetime
    status: Literal["COMPLETE", "INSUFFICIENT_DATA", "FAILED"]
    methodology: str
    model_name: str = ""
    manifest_path: str = ""
    candidates: list[EtfScore] = Field(default_factory=list)
    metrics: RankingMetrics | None = None
    extremes: list[ExtremeAnalysis] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranking(self) -> "DimensionRanking":
        """Require stable rank order and unique candidates."""
        if len(self.candidates) > 20:
            raise ValueError("dimension ranking cannot contain more than 20 candidates")
        codes = [candidate.code for candidate in self.candidates]
        if len(codes) != len(set(codes)):
            raise ValueError("dimension ranking candidates must have unique codes")
        if [candidate.rank for candidate in self.candidates] != list(
            range(1, len(self.candidates) + 1),
        ):
            raise ValueError("dimension ranking candidates must use contiguous ranks")
        return self


class FusionRanking(AutoFinModel):
    """Weighted fusion of all available research dimensions."""

    as_of: datetime
    weights: dict[Literal["event", "backtest", "us_correlation"], float]
    candidates: list[EtfScore] = Field(default_factory=list)
    methodology: str
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fusion(self) -> "FusionRanking":
        """Validate normalized weights and stable Top20 ranks."""
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("fusion ranking weights must be finite and non-negative")
        if self.weights and not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("fusion ranking weights must sum to one")
        if len(self.candidates) > 20:
            raise ValueError("fusion ranking cannot contain more than 20 candidates")
        if [candidate.rank for candidate in self.candidates] != list(
            range(1, len(self.candidates) + 1),
        ):
            raise ValueError("fusion ranking candidates must use contiguous ranks")
        return self


class EventAnalysisOutput(AutoFinModel):
    """Structured output required from the event-analysis Agent."""

    description: str
    body: str
    window: EventWindow
    sources: list[DataSource] = Field(default_factory=list)
    events: list[EventSignal] = Field(default_factory=list)
    cursor: EventCursor
    ranking: DimensionRanking | None = None
    limitations: list[str] = Field(default_factory=list)


class PositionMark(AutoFinModel):
    """One reproducible return interval used by the deterministic ledger."""

    code: str
    interval_id: str
    interval_start: datetime
    interval_end: datetime
    interval_return: float
    source_manifest: str

    @model_validator(mode="after")
    def validate_interval(self) -> "PositionMark":
        """Reject impossible intervals and non-finite returns."""
        if self.interval_end <= self.interval_start:
            raise ValueError("position mark interval_end must be after interval_start")
        if not math.isfinite(self.interval_return) or self.interval_return <= -1.0:
            raise ValueError("position mark interval_return must be finite and greater than -1")
        if not self.source_manifest:
            raise ValueError("position mark source_manifest is required")
        return self


class BacktestExperiment(AutoFinModel):
    """One reproducible backtest result."""

    experiment_id: str
    sample_start: date
    sample_end: date
    sample_count: int = Field(ge=0)
    status: Literal["COMPLETE", "INSUFFICIENT_DATA", "FAILED"]
    summary: str


class BacktestSignal(AutoFinModel):
    """One evidence-backed market signal."""

    scope: Literal["market", "industry", "instrument"]
    code: str
    direction: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: str


class BacktestAnalysisOutput(AutoFinModel):
    """Structured output required from the backtest-analysis Agent."""

    description: str
    body: str
    market_cutoff: datetime
    data_manifest: str
    code_version: str
    parameter_hash: str
    adjustment: str
    market_data_complete: bool
    settlement_marks: list[PositionMark] = Field(default_factory=list)
    position_marks: list[PositionMark] = Field(default_factory=list)
    experiments: list[BacktestExperiment] = Field(default_factory=list)
    signals: list[BacktestSignal] = Field(default_factory=list)
    ranking: DimensionRanking | None = None
    limitations: list[str] = Field(default_factory=list)


class CorrelationMapping(AutoFinModel):
    """One aligned US-to-A-share relationship."""

    us_code: str
    a_share_industries: list[str] = Field(default_factory=list)
    a_share_codes: list[str] = Field(default_factory=list)
    correlation_method: str
    sample_count: int = Field(ge=0)
    conclusion: str


class UsCorrelationAnalysisOutput(AutoFinModel):
    """Structured output required from the US-correlation Agent."""

    description: str
    body: str
    as_of: datetime
    us_session_date: date
    a_share_trade_date: date
    universe_method: Literal["top50_by_recent_average_amount"]
    lookbacks: list[Literal["1D", "5D", "30D"]]
    mappings: list[CorrelationMapping] = Field(default_factory=list)
    ranking: DimensionRanking | None = None
    limitations: list[str] = Field(default_factory=list)


class ProposedAction(AutoFinModel):
    """Agent proposal enriched and validated by the deterministic executor."""

    action_id: str = ""
    action: ActionType
    code: str
    name: str = ""
    instrument_type: InstrumentType
    slot_count: Literal[1] = 1
    reason: str
    counterexample: str
    invalidation_condition: str
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_at: datetime | None = None
    scheduled_fill_at: datetime | None = None
    status: ActionStatus = ActionStatus.PROPOSED
    rejection_reason: str = ""


class PortfolioProposalOutput(AutoFinModel):
    """Structured output required from the portfolio-analysis Agent."""

    description: str
    body: str
    actions: list[ProposedAction] = Field(default_factory=list)
    fusion_ranking: FusionRanking | None = None
    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AnalysisState(AutoFinModel):
    """Compact persisted state for an analysis whose prose lives in Markdown."""

    # These fields accept legacy auto-fin/v1 documents, but are deliberately
    # omitted when new frontmatter is serialized because the report body
    # already contains them.
    description: str = Field(default="", exclude=True)
    body: str = Field(default="", exclude=True)
    ranking: DimensionRanking | None = Field(default=None, exclude=True)
    limitations: list[str] = Field(default_factory=list, exclude=True)
    ranking_manifest: str = ""

    @model_validator(mode="before")
    @classmethod
    def capture_ranking_manifest(cls, value):
        """Replace an embedded ranking with its rebuildable manifest reference."""
        if not isinstance(value, dict) or value.get("ranking_manifest"):
            return value
        ranking = value.get("ranking")
        if isinstance(ranking, DimensionRanking):
            manifest_path = ranking.manifest_path
        elif isinstance(ranking, dict):
            manifest_path = str(ranking.get("manifest_path") or "")
        else:
            manifest_path = ""
        return {**value, "ranking_manifest": manifest_path}


class EventAnalysisState(AnalysisState):
    """Non-rendered event-analysis state retained in frontmatter."""

    window: EventWindow
    sources: list[DataSource] = Field(default_factory=list)
    events: list[EventSignal] = Field(default_factory=list)
    cursor: EventCursor


class BacktestAnalysisState(AnalysisState):
    """Non-rendered backtest state retained in frontmatter."""

    market_cutoff: datetime
    data_manifest: str
    code_version: str
    parameter_hash: str
    adjustment: str
    market_data_complete: bool
    settlement_marks: list[PositionMark] = Field(default_factory=list)
    position_marks: list[PositionMark] = Field(default_factory=list)
    experiments: list[BacktestExperiment] = Field(default_factory=list)
    signals: list[BacktestSignal] = Field(default_factory=list)


class UsCorrelationAnalysisState(AnalysisState):
    """Non-rendered US-correlation state retained in frontmatter."""

    as_of: datetime
    us_session_date: date
    a_share_trade_date: date
    universe_method: Literal["top50_by_recent_average_amount"]
    lookbacks: list[Literal["1D", "5D", "30D"]]
    mappings: list[CorrelationMapping] = Field(default_factory=list)


class Position(AutoFinModel):
    """One normalized position; no real price, quantity, or lot fields exist."""

    code: str
    name: str
    instrument_type: InstrumentType
    buy_trade_date: date
    eligible_sell_date: date
    entry_notional: float = Field(gt=0.0)
    cumulative_return_factor: float = Field(default=1.0, gt=0.0)
    interval_return: float = 0.0
    cumulative_return: float = 0.0
    normalized_value: float = Field(gt=0.0)
    portfolio_contribution: float = 0.0
    marked_at: datetime | None = None
    applied_interval_ids: list[str] = Field(default_factory=list)


class PortfolioSnapshot(AutoFinModel):
    """Complete rebuildable state after one checkpoint mark."""

    nav: float = Field(default=1.0, gt=0.0)
    cash_nav: float = Field(default=1.0, ge=0.0)
    realized_return: float = 0.0
    positions: list[Position] = Field(default_factory=list)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    settled_action_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_invariants(self) -> "PortfolioSnapshot":
        """Validate the ten-slot normalized ledger invariants."""
        if len(self.positions) > 10:
            raise ValueError("portfolio cannot hold more than 10 positions")
        codes = [position.code for position in self.positions]
        if len(codes) != len(set(codes)):
            raise ValueError("portfolio positions must have unique codes")
        expected_nav = self.cash_nav + sum(position.normalized_value for position in self.positions)
        if not math.isclose(self.nav, expected_nav, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"portfolio nav mismatch: nav={self.nav}, expected={expected_nav}")
        return self


class Settlement(AutoFinModel):
    """Result of attempting to settle one prior proposal."""

    action_id: str
    action: ActionType
    code: str
    status: ActionStatus
    fill_basis: str
    filled_at: datetime | None = None
    reason: str = ""


class PortfolioMetrics(AutoFinModel):
    """Summary metrics rendered into portfolio frontmatter and Markdown."""

    nav: float
    cash_nav: float
    position_count: int = Field(ge=0, le=10)
    interval_return: float = 0.0


class UpstreamAnalysis(AutoFinModel):
    """Freshness metadata passed to the portfolio Agent."""

    run_id: str
    status: RunStatus
    data_cutoff: datetime
    generated_at: datetime
    stale: bool = False


class AnalysisRun(AutoFinModel):
    """Common persisted metadata for one analysis run."""

    run_id: str
    checkpoint: Checkpoint
    status: RunStatus
    decision_at: datetime
    data_cutoff: datetime
    generated_at: datetime
    stale: bool = False
    error: str = ""


class EventAnalysisRun(AnalysisRun):
    """One persisted event-analysis run."""

    analysis: EventAnalysisState | None = None


class BacktestAnalysisRun(AnalysisRun):
    """One persisted backtest-analysis run."""

    analysis: BacktestAnalysisState | None = None


class UsCorrelationAnalysisRun(AnalysisRun):
    """One persisted US-correlation run."""

    analysis: UsCorrelationAnalysisState


class PortfolioRun(AnalysisRun):
    """One complete, rebuildable portfolio checkpoint run."""

    # Accepted for auto-fin/v1 compatibility, but omitted from new documents:
    # the prior metrics are already rendered in the checkpoint section.
    portfolio_before: PortfolioMetrics | None = Field(default=None, exclude=True)
    settlements: list[Settlement] = Field(default_factory=list)
    # Accepted for auto-fin/v1 compatibility, but omitted from new documents:
    # both values are already represented by snapshot and the report body.
    positions: list[Position] = Field(default_factory=list, exclude=True)
    portfolio_after_mark: PortfolioMetrics | None = Field(default=None, exclude=True)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    rejected_actions: list[ProposedAction] = Field(default_factory=list)
    upstream: dict[str, UpstreamAnalysis]
    snapshot: PortfolioSnapshot = Field(default_factory=PortfolioSnapshot)


_RunT = TypeVar("_RunT", bound=AnalysisRun)


class AutoFinDocument(AutoFinModel, Generic[_RunT]):
    """Frontmatter contract shared by all Auto Fin Markdown documents."""

    schema_version: Literal["auto-fin/v1"] = "auto-fin/v1"
    document_type: str
    trade_date: date
    timezone: str
    updated_at: datetime
    runs: list[_RunT] = Field(default_factory=list)


class EventAnalysisDocument(AutoFinDocument[EventAnalysisRun]):
    """Public frontmatter contract for event_analysis.md."""

    document_type: Literal["event_analysis"] = "event_analysis"


class BacktestAnalysisDocument(AutoFinDocument[BacktestAnalysisRun]):
    """Public frontmatter contract for backtest_analysis.md."""

    document_type: Literal["backtest_analysis"] = "backtest_analysis"


class UsCorrelationAnalysisDocument(AutoFinDocument[UsCorrelationAnalysisRun]):
    """Public frontmatter contract for us_correlation_analysis.md."""

    document_type: Literal["us_correlation_analysis"] = "us_correlation_analysis"


class PortfolioDocument(AutoFinDocument[PortfolioRun]):
    """Public frontmatter contract for portfolio.md."""

    document_type: Literal["portfolio"] = "portfolio"
