"""Focused tests for the deterministic Auto Fin contracts and wiring."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
import numpy as np
import polars as pl

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.outbound_proxy import FixedHttpOutboundProxy
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.enumeration import ComponentEnum
from reme.schema import (
    ActionStatus,
    ActionType,
    BacktestAnalysisDocument,
    BacktestAnalysisOutput,
    Checkpoint,
    EventAnalysisDocument,
    EventAnalysisOutput,
    InstrumentType,
    PortfolioDocument,
    PortfolioProposalOutput,
    PortfolioSnapshot,
    Position,
    PositionMark,
    ProposedAction,
    RunStatus,
    UsCorrelationAnalysisDocument,
    UsCorrelationAnalysisOutput,
)
from reme.steps.cookbook.auto_fin._common import latest_portfolio_snapshot, load_document, upsert_report
from reme.steps.cookbook.auto_fin.analysis import (
    AutoFinBacktestStep,
    AutoFinEventStep,
    AutoFinPortfolioStep,
    AutoFinQuantStep,
    AutoFinQuantResearch,
    AutoFinUsCorrelationStep,
)
from reme.steps.cookbook.auto_fin.analysis.quant import TushareResearchClient
from reme.steps.cookbook.auto_fin.ledger import AutoFinLedger, next_trade_date
from reme.steps.cookbook.auto_fin.notification import AutoFinNotificationStep
from reme.steps.cookbook.auto_fin.pipeline import AutoFinPipelineStep
from reme.steps.cookbook.dingtalk.send import DingTalkMarkdownSendStep
from reme.utils.tushare import create_tushare_api

TZ = ZoneInfo("Asia/Shanghai")


class _AutoFinAnalysisWrapper(BaseAgentWrapper):
    """Return minimal valid outputs for an end-to-end pipeline test."""

    async def reply(self, _inputs, **kwargs):
        output_schema = kwargs["output_schema"]
        if output_schema is EventAnalysisOutput:
            value = {
                "description": "无新增事件",
                "body": "本窗口没有可验证的新增事件。",
                "window": {
                    "start_exclusive": "2026-07-22T15:00:00+08:00",
                    "end_inclusive": "2026-07-23T09:00:00+08:00",
                },
                "sources": [],
                "events": [],
                "cursor": {},
                "limitations": [],
            }
        elif output_schema is BacktestAnalysisOutput:
            value = {
                "description": "行情数据完整",
                "body": "当前空仓，无需应用收益区间。",
                "market_cutoff": "2026-07-22T15:00:00+08:00",
                "data_manifest": "resource/auto-fin/2026-07-23/manifests/backtest-0900.json",
                "code_version": "test",
                "parameter_hash": "test",
                "adjustment": "ETF OHLC×fund_adj; A-share close×adj_factor",
                "market_data_complete": True,
                "settlement_marks": [],
                "position_marks": [],
                "experiments": [],
                "signals": [],
                "limitations": [],
            }
        elif output_schema is UsCorrelationAnalysisOutput:
            value = {
                "description": "美股关联样本",
                "body": "测试关联输出。",
                "as_of": "2026-07-23T09:00:00+08:00",
                "us_session_date": "2026-07-22",
                "a_share_trade_date": "2026-07-23",
                "universe_method": "top50_by_recent_average_amount",
                "lookbacks": ["1D", "5D", "30D"],
                "mappings": [],
                "limitations": [],
            }
        elif output_schema is PortfolioProposalOutput:
            value = {
                "description": "保持空仓",
                "body": "证据不足，保持空仓。纯模拟盘，不会执行真实交易。",
                "actions": [
                    {
                        "action": "HOLD",
                        "code": "CASH",
                        "instrument_type": "domestic_equity_etf",
                        "reason": "证据不足",
                        "counterexample": "后续证据改善",
                        "invalidation_condition": "出现可靠信号",
                        "confidence": 0.8,
                    },
                ],
                "risks": [],
                "limitations": [],
            }
        else:  # pragma: no cover - a new analysis stage must update this fixture.
            raise AssertionError(output_schema)
        return {"structured_output": output_schema.model_validate(value)}


def _action(action_id: str, action: ActionType, code: str) -> ProposedAction:
    return ProposedAction(
        action_id=action_id,
        action=action,
        code=code,
        name=code,
        instrument_type=InstrumentType.DOMESTIC_EQUITY_ETF,
        reason="test evidence",
        counterexample="test counterexample",
        invalidation_condition="test invalidation",
        confidence=0.7,
    )


def _mark(
    code: str,
    interval_id: str,
    start_hour: int,
    end_hour: int,
    value: float,
) -> PositionMark:
    return PositionMark(
        code=code,
        interval_id=interval_id,
        interval_start=datetime(2026, 7, 23, start_hour, tzinfo=TZ),
        interval_end=datetime(2026, 7, 23, end_hour, tzinfo=TZ),
        interval_return=value,
        source_manifest="resource/auto-fin/2026-07-23/manifests/test.json",
    )


def test_ledger_buys_marks_compound_and_sells_without_prices():
    ledger = AutoFinLedger()
    assert ledger.snapshot.nav == ledger.snapshot.cash_nav == 1.0

    settlements = ledger.settle(
        [_action("buy-1", ActionType.BUY, "510300.SH")],
        trade_date=date(2026, 7, 23),
        eligible_sell_date=date(2026, 7, 24),
        fill_at=datetime(2026, 7, 23, 9, 30, tzinfo=TZ),
        fill_basis="0930_OPEN",
        market_data_complete=True,
    )
    assert settlements[0].status is ActionStatus.FILLED
    assert ledger.snapshot.cash_nav == pytest.approx(0.9)
    assert ledger.snapshot.nav == pytest.approx(1.0)
    assert ledger.snapshot.positions[0].entry_notional == pytest.approx(0.1)

    ledger.apply_marks(
        [
            _mark("510300.SH", "first", 9, 11, 0.02),
            _mark("510300.SH", "second", 11, 14, -0.01),
        ],
    )
    position = ledger.snapshot.positions[0]
    assert position.cumulative_return_factor == pytest.approx(1.02 * 0.99)
    assert position.normalized_value == pytest.approx(0.1 * 1.02 * 0.99)
    assert ledger.snapshot.nav == pytest.approx(0.9 + position.normalized_value)

    # Applying an already-recorded interval is idempotent.
    ledger.apply_marks([_mark("510300.SH", "second", 11, 14, -0.01)])
    assert ledger.snapshot.positions[0].cumulative_return_factor == pytest.approx(
        1.02 * 0.99,
    )

    same_day = ledger.settle(
        [_action("sell-early", ActionType.SELL, "510300.SH")],
        trade_date=date(2026, 7, 23),
        eligible_sell_date=date(2026, 7, 24),
        fill_at=datetime(2026, 7, 23, 15, 0, tzinfo=TZ),
        fill_basis="1500_CLOSE",
        market_data_complete=True,
    )
    assert same_day[0].status is ActionStatus.REJECTED
    assert "T+1" in same_day[0].reason

    next_day = ledger.settle(
        [_action("sell-next", ActionType.SELL, "510300.SH")],
        trade_date=date(2026, 7, 24),
        eligible_sell_date=date(2026, 7, 27),
        fill_at=datetime(2026, 7, 24, 9, 30, tzinfo=TZ),
        fill_basis="0930_OPEN",
        market_data_complete=True,
    )
    assert next_day[0].status is ActionStatus.FILLED
    assert not ledger.snapshot.positions
    assert ledger.snapshot.nav == ledger.snapshot.cash_nav

    assert "price" not in Position.model_fields
    assert "quantity" not in Position.model_fields
    assert "shares" not in Position.model_fields


def test_ledger_enforces_ten_unique_slots_and_cash():
    ledger = AutoFinLedger()
    actions = [_action(f"buy-{index}", ActionType.BUY, f"{index:06d}.SH") for index in range(10)]
    results = ledger.settle(
        actions,
        trade_date=date(2026, 7, 23),
        eligible_sell_date=date(2026, 7, 24),
        fill_at=datetime(2026, 7, 23, 9, 30, tzinfo=TZ),
        fill_basis="0930_OPEN",
        market_data_complete=True,
    )
    assert all(result.status is ActionStatus.FILLED for result in results)
    assert len(ledger.snapshot.positions) == 10
    assert ledger.snapshot.cash_nav == pytest.approx(0.0)

    extra = ledger.settle(
        [_action("buy-extra", ActionType.BUY, "999999.SH")],
        trade_date=date(2026, 7, 24),
        eligible_sell_date=date(2026, 7, 27),
        fill_at=datetime(2026, 7, 24, 9, 30, tzinfo=TZ),
        fill_basis="0930_OPEN",
        market_data_complete=True,
    )
    assert extra[0].status is ActionStatus.REJECTED
    assert "limit" in extra[0].reason


def test_degraded_run_rejects_buy_and_trade_calendar_handles_weekend():
    ledger = AutoFinLedger()
    accepted, rejected = ledger.validate_proposals(
        [_action("", ActionType.BUY, "510300.SH")],
        run_id="2026-07-23T1145+08:00",
        trade_date=date(2026, 7, 23),
        proposed_at=datetime(2026, 7, 23, 11, 46, tzinfo=TZ),
        scheduled_fill_at=datetime(2026, 7, 23, 13, 0, tzinfo=TZ),
        run_status=RunStatus.DEGRADED,
        research_only=False,
    )
    assert not accepted
    assert rejected[0].status is ActionStatus.REJECTED
    assert "cannot add risk" in rejected[0].rejection_reason

    assert next_trade_date(
        date(2026, 7, 24),
        [date(2026, 7, 24), date(2026, 7, 27)],
    ) == date(2026, 7, 27)


def test_checkpoint_schedule_uses_distinct_fill_and_mark_times():
    schedule = AutoFinPipelineStep._schedule(
        date(2026, 7, 23),
        date(2026, 7, 22),
        Checkpoint.MIDDAY,
        TZ,
    )
    assert schedule.settlement_fill_at.hour == 9
    assert schedule.settlement_fill_at.minute == 30
    assert schedule.market_cutoff.hour == 11
    assert schedule.market_cutoff.minute == 30
    assert schedule.scheduled_fill_at.hour == 13


def test_checkpoint_rejects_execution_before_declared_cutoff():
    schedule = AutoFinPipelineStep._schedule(
        date(2026, 7, 23),
        date(2026, 7, 22),
        Checkpoint.OPEN,
        TZ,
    )
    with pytest.raises(ValueError, match="checkpoint has not been reached"):
        AutoFinPipelineStep._require_checkpoint_reached(
            schedule,
            datetime(2026, 7, 23, 8, 59, tzinfo=TZ),
        )
    AutoFinPipelineStep._require_checkpoint_reached(
        schedule,
        datetime(2026, 7, 23, 9, 0, tzinfo=TZ),
    )


def test_quant_features_use_adjusted_prices_and_next_open_to_open_label():
    trade_dates = [date(2026, 7, 20) + timedelta(days=index) for index in range(4)]
    raw = pl.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_date": trade_day.strftime("%Y%m%d"),
                "open": 10.0 + index,
                "high": 11.0 + index,
                "low": 9.0 + index,
                "close": 10.5 + index,
                "pre_close": 9.5 + index,
                "amount": 100.0,
            }
            for index, trade_day in enumerate(trade_dates)
        ],
    )
    factors = pl.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_date": trade_day.strftime("%Y%m%d"),
                "adj_factor": factor,
            }
            for trade_day, factor in zip(trade_dates, (1.0, 1.0, 2.0, 2.0))
        ],
    )
    daily, _ = AutoFinQuantResearch._prepare_universe(
        {
            "universe": pl.DataFrame([{"ts_code": "510300.SH", "name": "沪深300ETF"}]),
            "etf_daily": raw,
            "fund_adj": factors,
        },
        1,
    )
    features = AutoFinQuantResearch._features(daily)
    first = features.sort("trade_date").row(0, named=True)

    assert first["prediction_date"] == trade_dates[1]
    assert first["future_open_return"] == pytest.approx((12.0 * 2.0) / (11.0 * 1.0) - 1.0)
    assert daily.sort("trade_date")["open"].to_list() == pytest.approx([10.0, 11.0, 24.0, 26.0])

    with pytest.raises(ValueError, match="adjustment factors are incomplete"):
        AutoFinQuantResearch._prepare_universe(
            {
                "universe": pl.DataFrame([{"ts_code": "510300.SH", "name": "沪深300ETF"}]),
                "etf_daily": raw,
                "fund_adj": factors.head(3),
            },
            1,
        )


def test_tushare_api_uses_one_explicit_proxy_without_environment_fallback(monkeypatch):
    import requests
    import tushare

    calls: list[dict] = []

    class FakeApi:
        """Expose the transport settings used by TuShare DataApi."""

        _DataApi__http_url = "http://api.example.test/dataapi"
        _DataApi__timeout = 17

    class FakeResponse:
        """Minimal successful requests response."""

        text = """
        {
          "code": 0,
          "msg": "",
          "data": {
            "fields": ["exchange", "cal_date", "is_open"],
            "items": [["SSE", "20260723", 1]]
          }
        }
        """

        def __bool__(self):
            return True

    class FakeSession:
        """Capture the explicit proxy settings used for one request."""

        def __init__(self):
            self.trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            calls.append({"url": url, "trust_env": self.trust_env, **kwargs})
            return FakeResponse()

    monkeypatch.setattr(tushare, "pro_api", lambda _token: FakeApi())
    monkeypatch.setattr(requests, "Session", FakeSession)
    monkeypatch.setenv("HTTP_PROXY", "http://environment-proxy.example:8080")

    client = create_tushare_api("test-token", proxy_url="http://managed-proxy.example:18080")
    frame = client.trade_cal(exchange="SSE", fields="exchange,cal_date,is_open")

    assert frame.to_dict(orient="records") == [{"exchange": "SSE", "cal_date": "20260723", "is_open": 1}]
    assert calls == [
        {
            "url": "http://api.example.test/dataapi/trade_cal",
            "trust_env": False,
            "json": {
                "api_name": "trade_cal",
                "token": "test-token",
                "params": {
                    "exchange": "SSE",
                    "ts_type_name": "http://api.example.test/dataapi",
                },
                "fields": "exchange,cal_date,is_open",
            },
            "timeout": 17,
            "proxies": {
                "http": "http://managed-proxy.example:18080",
                "https": "http://managed-proxy.example:18080",
            },
        },
    ]


def test_tushare_research_client_forwards_managed_proxy(monkeypatch):
    created: list[tuple[str, str | None]] = []
    api = MagicMock()

    def fake_create(token: str, *, proxy_url: str | None = None):
        created.append((token, proxy_url))
        return api

    monkeypatch.setattr("reme.steps.cookbook.auto_fin.analysis.quant.create_tushare_api", fake_create)

    client = TushareResearchClient(
        "test-token",
        concurrency=3,
        proxy_url="http://managed-proxy.example:18080",
    )

    assert client._pro is api
    assert created == [("test-token", "http://managed-proxy.example:18080")]


def test_trade_calendar_forwards_managed_proxy(monkeypatch):
    received: list[tuple[str, str | None]] = []
    frame = MagicMock()
    frame.to_dict.return_value = [
        {"exchange": "SSE", "cal_date": "20260723", "is_open": 1, "pretrade_date": "20260722"},
    ]
    api = MagicMock()
    api.trade_cal.return_value = frame

    def fake_create(token: str, *, proxy_url: str | None = None):
        received.append((token, proxy_url))
        return api

    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setattr("reme.steps.cookbook.auto_fin.pipeline.create_tushare_api", fake_create)

    open_dates, records = AutoFinPipelineStep._fetch_trade_calendar_sync(
        date(2026, 7, 20),
        date(2026, 7, 24),
        "http://managed-proxy.example:18080",
    )

    assert open_dates == [date(2026, 7, 23)]
    assert records == frame.to_dict.return_value
    assert received == [("test-token", "http://managed-proxy.example:18080")]


def test_auto_fin_network_steps_resolve_default_outbound_proxy(tmp_path: Path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    proxy = FixedHttpOutboundProxy(url="http://127.0.0.1:18080")
    app_context.components = {ComponentEnum.OUTBOUND_PROXY: {"default": proxy}}

    assert AutoFinPipelineStep(app_context=app_context).outbound_proxy is proxy
    assert AutoFinQuantStep(app_context=app_context).outbound_proxy is proxy


def test_complete_backtest_requires_exact_marks_for_held_positions():
    schedule = AutoFinPipelineStep._schedule(
        date(2026, 7, 23),
        date(2026, 7, 22),
        Checkpoint.MIDDAY,
        TZ,
    )
    snapshot = PortfolioSnapshot(
        nav=1.0,
        cash_nav=0.9,
        positions=[
            Position(
                code="510300.SH",
                name="沪深300ETF",
                instrument_type=InstrumentType.DOMESTIC_EQUITY_ETF,
                buy_trade_date=date(2026, 7, 22),
                eligible_sell_date=date(2026, 7, 23),
                entry_notional=0.1,
                normalized_value=0.1,
                marked_at=datetime(2026, 7, 22, 15, 0, tzinfo=TZ),
            ),
        ],
    )
    output = BacktestAnalysisOutput(
        description="missing marks",
        body="",
        market_cutoff=schedule.market_cutoff,
        data_manifest="resource/auto-fin/test.json",
        code_version="test",
        parameter_hash="test",
        adjustment="ETF OHLC×fund_adj; A-share close×adj_factor",
        market_data_complete=True,
    )
    AutoFinBacktestStep.validate_output(output, schedule.market_cutoff)
    with pytest.raises(ValueError, match="fund_adj-adjusted"):
        AutoFinBacktestStep.validate_output(
            output.model_copy(update={"adjustment": "raw prices"}),
            schedule.market_cutoff,
        )
    with pytest.raises(ValueError, match="settlement mark"):
        AutoFinBacktestStep.validate_required_marks(
            snapshot,
            output,
            settlement_fill_at=schedule.settlement_fill_at,
            market_cutoff=schedule.market_cutoff,
        )


def test_auto_fin_analysis_steps_load_independent_prompts():
    prompts = (
        (AutoFinEventStep, "event_user"),
        (AutoFinBacktestStep, "backtest_user"),
        (AutoFinUsCorrelationStep, "us_user"),
        (AutoFinPortfolioStep, "portfolio_user"),
    )
    for step_type, prompt_name in prompts:
        step = step_type()
        assert step.get_prompt(prompt_name)


def test_quant_research_builds_three_top20_rankings_and_fusion():
    random = np.random.default_rng(7)
    trade_dates: list[date] = []
    current = date(2025, 1, 2)
    while len(trade_dates) < 90:
        if current.weekday() < 5:
            trade_dates.append(current)
        current += timedelta(days=1)

    etf_rows: list[dict] = []
    basic_rows: list[dict] = []
    for code_index in range(20):
        code = f"{510000 + code_index:06d}.SH"
        basic_rows.append({"ts_code": code, "name": f"行业{code_index}ETF"})
        close = 1.0 + code_index / 100.0
        for trade_day in trade_dates:
            daily_return = code_index / 100000.0 + random.normal(0.0, 0.01)
            opening = close * (1.0 + random.normal(0.0, 0.002))
            previous_close = close
            close *= 1.0 + daily_return
            etf_rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_day.strftime("%Y%m%d"),
                    "open": opening,
                    "high": max(opening, close) * 1.002,
                    "low": min(opening, close) * 0.998,
                    "close": close,
                    "pre_close": previous_close,
                    "vol": 1_000_000.0,
                    "amount": 100_000_000.0 + code_index * 1_000_000.0,
                },
            )
    us_rows: list[dict] = []
    for us_code in ("NVDA", "SOXX", "SOXL"):
        close = 100.0
        for trade_day in trade_dates:
            daily_return = random.normal(0.0, 0.015)
            close *= 1.0 + daily_return
            us_rows.append(
                {
                    "ts_code": us_code,
                    "trade_date": (trade_day - timedelta(days=1)).strftime("%Y%m%d"),
                    "close": close,
                    "pct_change": daily_return * 100.0,
                    "adj_factor": 1.0,
                },
            )
    event = EventAnalysisOutput.model_validate(
        {
            "description": "行业事件",
            "body": "行业1出现可验证事件。",
            "window": {
                "start_exclusive": "2025-05-07T15:00:00+08:00",
                "end_inclusive": "2025-05-08T09:00:00+08:00",
            },
            "events": [
                {
                    "event_id": "event-1",
                    "published_at": "2025-05-08T08:00:00+08:00",
                    "fetched_at": "2025-05-08T08:05:00+08:00",
                    "title": "行业1事件",
                    "industries": ["行业1"],
                    "codes": [],
                    "dedupe_key": "event-1",
                    "known_before_cutoff": True,
                    "direction": "POSITIVE",
                    "confidence": 0.8,
                    "horizon": "5D",
                    "summary": "测试",
                    "source_ref": "fixture",
                },
            ],
            "cursor": {},
        },
    )
    quant_bundle = {
        "universe": pl.DataFrame(basic_rows),
        "etf_daily": pl.DataFrame(etf_rows),
        "fund_adj": pl.DataFrame(
            [
                {
                    "ts_code": row["ts_code"],
                    "trade_date": row["trade_date"],
                    "adj_factor": 1.0,
                }
                for row in etf_rows
            ],
        ),
        "us_daily": pl.DataFrame(us_rows),
        "holdings": pl.DataFrame(
            [
                {
                    "ts_code": "510019.SH",
                    "symbol": f"00000{index}.SZ",
                    "end_date": "20250331",
                    "stk_mkv_ratio": 10.0 - index,
                }
                for index in range(1, 4)
            ],
        ),
        "stock_daily": pl.DataFrame(
            [
                {
                    "ts_code": f"00000{index}.SZ",
                    "trade_date": trade_day.strftime("%Y%m%d"),
                    "close": 10.0 + day_index + index,
                }
                for index in range(1, 4)
                for day_index, trade_day in enumerate(trade_dates[-3:])
            ],
        ),
        "stock_adj": pl.DataFrame(
            [
                {
                    "ts_code": f"00000{index}.SZ",
                    "trade_date": trade_day.strftime("%Y%m%d"),
                    "adj_factor": 1.0,
                }
                for index in range(1, 4)
                for trade_day in trade_dates[-3:]
            ],
        ),
        "stock_valuation": pl.DataFrame(
            [
                {
                    "ts_code": f"00000{index}.SZ",
                    "pe_ttm": 15.0 + index,
                    "pb": 2.0 + index / 10.0,
                }
                for index in range(1, 4)
            ],
        ),
    }
    quant_engine = AutoFinQuantResearch(tree_count=5)
    rankings, fusion = quant_engine.run(
        quant_bundle,
        event_output=event,
        as_of=datetime(2025, 5, 8, 9, tzinfo=TZ),
        universe_size=20,
        weights={"event": 0.30, "backtest": 0.45, "us_correlation": 0.25},
    )

    assert rankings["event"].candidates
    assert rankings["event"].candidates[0].price_in is not None
    constituent_candidate = next(
        candidate for candidate in rankings["event"].candidates if candidate.code == "510019.SH"
    )
    assert any("PE(TTM)" in reason for reason in constituent_candidate.reasons)
    for dimension in ("backtest", "us_correlation"):
        assert rankings[dimension].status == "COMPLETE", rankings[dimension].limitations
        assert len(rankings[dimension].candidates) == 20
        assert rankings[dimension].metrics.validation_sample_count > 0
        assert rankings[dimension].metrics.ndcg_at_20 is not None
    assert len(fusion.candidates) == 20
    assert sum(fusion.weights.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_report_upsert_replaces_same_checkpoint(tmp_path: Path):
    path = tmp_path / "daily" / "2026-07-23" / "portfolio.md"
    base_run = {
        "run_id": "2026-07-23T0900+08:00",
        "decision_at": "2026-07-23T09:00:00+08:00",
        "generated_at": "2026-07-23T09:01:00+08:00",
    }
    kwargs = {
        "document_type": "portfolio",
        "trade_date": "2026-07-23",
        "timezone": "Asia/Shanghai",
        "checkpoint": Checkpoint.OPEN,
        "title": "2026-07-23 Auto Fin Portfolio",
    }
    await upsert_report(path, run={**base_run, "value": 1}, section="first", **kwargs)
    await upsert_report(path, run={**base_run, "value": 2}, section="second", **kwargs)

    document = load_document(path)
    assert len(document.metadata["runs"]) == 1
    assert document.metadata["runs"][0]["value"] == 2
    assert "second" in document.content
    assert "first" not in document.content


@pytest.mark.asyncio
async def test_latest_portfolio_snapshot_rebuilds_pending_actions(tmp_path: Path):
    path = tmp_path / "daily" / "2026-07-23" / "portfolio.md"
    pending = _action("buy-pending", ActionType.BUY, "510300.SH").model_copy(
        update={
            "proposed_at": datetime(2026, 7, 23, 9, tzinfo=TZ),
            "scheduled_fill_at": datetime(2026, 7, 23, 11, 45, tzinfo=TZ),
        },
    )
    run = {
        "run_id": "2026-07-23T0900+08:00",
        "decision_at": "2026-07-23T09:00:00+08:00",
        "generated_at": "2026-07-23T09:01:00+08:00",
        "proposed_actions": [pending.model_dump(mode="json")],
        "snapshot": {},
    }
    await upsert_report(
        path,
        document_type="portfolio",
        trade_date="2026-07-23",
        timezone="Asia/Shanghai",
        checkpoint=Checkpoint.OPEN,
        run=run,
        section="portfolio",
        title="2026-07-23 Auto Fin Portfolio",
    )

    snapshot = latest_portfolio_snapshot(
        tmp_path,
        "daily",
        before=datetime(2026, 7, 23, 11, 45, tzinfo=TZ),
        excluding_run_id="another-run",
    )
    assert snapshot is not None
    assert [action.action_id for action in snapshot.proposed_actions] == ["buy-pending"]


def test_daily_cookbook_wires_auto_fin_cc_skill_memory_search_and_crons(monkeypatch):
    for name in (
        "AUTO_FIN_AGENT_BACKEND",
        "AUTO_FIN_PROJECT_PATH",
        "AUTO_FIN_MODEL_NAME",
        "AUTO_FIN_API_KEY",
        "AUTO_FIN_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    config = _load_config("daily_cookbook")
    wrapper = config["components"]["agent_wrapper"]["auto_fin"]
    assert wrapper["backend"] == "claude_code"
    assert wrapper["skills"] == ["tushare-data"]
    assert wrapper["job_tools"] == ["memory_search"]
    assert "memory_search" in config["jobs"]
    assert config["jobs"]["auto_fin"]["parameters"]["properties"]["force"]["default"] is True
    assert config["jobs"]["auto_fin"]["quant_enabled"] is True
    assert config["jobs"]["auto_fin"]["quant_universe_size"] == 50
    assert config["jobs"]["auto_fin"]["backtest_weight"] == pytest.approx(0.45)

    steps = config["jobs"]["auto_fin"]["steps"]
    assert len(steps) == 1
    assert steps[0]["backend"] == "auto_fin_pipeline_step"
    assert steps[0]["agent_wrapper"] == "auto_fin"
    dispatch = steps[0]["dispatch_steps"]
    assert len(dispatch) == 1
    assert dispatch[0]["backend"] == "auto_fin_notification_step"
    assert dispatch[0]["title"] == "ReMe Auto Fin"
    assert config["jobs"]["auto_fin_0900_cron"]["steps"] == steps
    assert config["jobs"]["auto_fin_1145_cron"]["steps"] == steps
    assert config["jobs"]["auto_fin_1445_cron"]["steps"] == steps
    assert config["jobs"]["auto_fin_0900_cron"]["cron"] == "0 9 * * 1-5"
    assert config["jobs"]["auto_fin_1145_cron"]["cron"] == "45 11 * * 1-5"
    assert config["jobs"]["auto_fin_1445_cron"]["cron"] == "45 14 * * 1-5"


def test_portfolio_snapshot_rejects_nav_mismatch():
    with pytest.raises(ValueError, match="nav mismatch"):
        PortfolioSnapshot(nav=1.0, cash_nav=0.5, positions=[])


@pytest.mark.asyncio
async def test_pipeline_writes_four_reports_notifies_each_agent_stage_and_skips_same_run(
    tmp_path: Path,
    monkeypatch,
):
    notified_stages: list[str] = []

    async def fake_send(self):
        notified_stages.append(str(self.context.get("auto_fin_notification_stage")))
        self.context.response.metadata["dingtalk_sent_count"] = 1
        return self.context.response

    monkeypatch.setattr(DingTalkMarkdownSendStep, "execute", fake_send)
    app_context = ApplicationContext(
        workspace_dir=str(tmp_path),
        timezone="Asia/Shanghai",
    )
    wrapper = _AutoFinAnalysisWrapper(app_context=app_context)
    step = AutoFinPipelineStep(
        app_context=app_context,
        agent_wrapper=wrapper,
        dispatch_steps=[{"backend": "auto_fin_notification_step"}],
    )
    step.logger = MagicMock()
    response = await step(
        date="2026-07-23",
        checkpoint="0900",
        trade_dates=["2026-07-22", "2026-07-23", "2026-07-24"],
        research_only=True,
        quant_enabled=False,
        quant_required=False,
    )
    assert response.success
    day_dir = tmp_path / "daily" / "2026-07-23"
    for name in (
        "event_analysis.md",
        "backtest_analysis.md",
        "us_correlation_analysis.md",
        "portfolio.md",
    ):
        assert (day_dir / name).is_file()
    analysis_documents = (
        (EventAnalysisDocument, load_document(day_dir / "event_analysis.md")),
        (BacktestAnalysisDocument, load_document(day_dir / "backtest_analysis.md")),
        (UsCorrelationAnalysisDocument, load_document(day_dir / "us_correlation_analysis.md")),
    )
    rendered_only_fields = {"description", "body", "ranking", "limitations"}
    for document_type, document in analysis_documents:
        document_type.model_validate(document.metadata)
        analysis = document.metadata["runs"][0]["analysis"]
        assert rendered_only_fields.isdisjoint(analysis)
    portfolio = load_document(day_dir / "portfolio.md")
    PortfolioDocument.model_validate(portfolio.metadata)
    assert len(portfolio.metadata["runs"]) == 1
    portfolio_run = portfolio.metadata["runs"][0]
    assert "portfolio_before" not in portfolio_run
    assert "positions" not in portfolio_run
    assert "portfolio_after_mark" not in portfolio_run
    assert "proposed_actions" not in portfolio_run.get("snapshot", {})
    assert notified_stages == ["event", "backtest", "us_correlation", "portfolio"]
    assert PortfolioDocument.model_validate(portfolio.metadata).runs[0].snapshot.nav == 1.0
    day_index = (tmp_path / "daily" / "2026-07-23.md").read_text(encoding="utf-8")
    assert "runs:" not in day_index
    assert "document_type:" not in day_index
    assert not list((tmp_path / "metadata" / "auto-fin" / "locks").glob("*.lock"))
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "checkpoint start run_id=2026-07-23T0900+08:00" in logs
    assert "analysis start stage=" in logs
    assert "settlements applied run_id=2026-07-23T0900+08:00" in logs
    assert "portfolio persistence done run_id=2026-07-23T0900+08:00" in logs
    assert "pipeline done run_id=2026-07-23T0900+08:00 status=COMPLETE" in logs

    rerun = await step(
        date="2026-07-23",
        checkpoint="0900",
        trade_dates=["2026-07-22", "2026-07-23", "2026-07-24"],
        force=False,
        research_only=True,
        quant_enabled=False,
        quant_required=False,
    )
    assert rerun.metadata["skipped"] is True
    assert len(load_document(day_dir / "portfolio.md").metadata["runs"]) == 1
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "pipeline skip existing run_id=2026-07-23T0900+08:00" in logs

    forced_by_default = await step(
        date="2026-07-23",
        checkpoint="0900",
        trade_dates=["2026-07-22", "2026-07-23", "2026-07-24"],
        research_only=True,
        quant_enabled=False,
        quant_required=False,
    )
    assert "skipped" not in forced_by_default.metadata
    assert len(load_document(day_dir / "portfolio.md").metadata["runs"]) == 1
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "checkpoint start run_id=2026-07-23T0900+08:00" in logs
    assert "timezone=Asia/Shanghai force=True" in logs
    assert notified_stages == ["event", "backtest", "us_correlation", "portfolio"]


@pytest.mark.asyncio
async def test_auto_fin_notification_records_success_and_deduplicates(
    tmp_path: Path,
    monkeypatch,
):
    calls = 0

    async def fake_send(self):
        nonlocal calls
        calls += 1
        self.context.response.metadata["dingtalk_sent_count"] = 1
        return self.context.response

    monkeypatch.setattr(DingTalkMarkdownSendStep, "execute", fake_send)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    step = AutoFinNotificationStep(app_context=app_context)
    context = RuntimeContext()
    context.response.metadata.update(
        {
            "notify": True,
            "run_id": "2026-07-23T0900+08:00",
        },
    )

    await step(context)
    await step(context)

    assert calls == 1
    assert context.response.metadata["dingtalk_skipped_duplicate"] is True
    state = tmp_path / "metadata" / "auto-fin" / "notification-state" / "2026-07-23T0900+08:00.json"
    assert state.is_file()


@pytest.mark.asyncio
async def test_auto_fin_notification_deduplicates_each_stage_independently(
    tmp_path: Path,
    monkeypatch,
):
    calls = 0

    async def fake_send(self):
        nonlocal calls
        calls += 1
        self.context.response.metadata["dingtalk_sent_count"] = 1
        return self.context.response

    monkeypatch.setattr(DingTalkMarkdownSendStep, "execute", fake_send)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    step = AutoFinNotificationStep(app_context=app_context)
    context = RuntimeContext()
    context.response.metadata.update(
        {
            "notify": True,
            "run_id": "2026-07-23T0900+08:00",
        },
    )

    await step(context, auto_fin_notification_stage="event")
    await step(context, auto_fin_notification_stage="backtest")
    await step(context, auto_fin_notification_stage="event")

    assert calls == 2
    state_dir = tmp_path / "metadata" / "auto-fin" / "notification-state"
    assert (state_dir / "2026-07-23T0900+08:00.event.json").is_file()
    assert (state_dir / "2026-07-23T0900+08:00.backtest.json").is_file()
