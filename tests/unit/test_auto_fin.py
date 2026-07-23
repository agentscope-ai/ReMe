"""Focused tests for the deterministic Auto Fin contracts and wiring."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.schema import (
    ActionStatus,
    ActionType,
    BacktestAnalysisOutput,
    Checkpoint,
    EventAnalysisOutput,
    InstrumentType,
    PortfolioDocument,
    PortfolioProposalOutput,
    PortfolioSnapshot,
    Position,
    PositionMark,
    ProposedAction,
    RunStatus,
    UsCorrelationAnalysisOutput,
)
from reme.steps.cookbook.auto_fin._common import load_document, upsert_report
from reme.steps.cookbook.auto_fin.analysis import (
    AutoFinBacktestStep,
    AutoFinEventStep,
    AutoFinPortfolioStep,
    AutoFinUsCorrelationStep,
)
from reme.steps.cookbook.auto_fin.ledger import AutoFinLedger, next_trade_date
from reme.steps.cookbook.auto_fin.notification import AutoFinNotificationStep
from reme.steps.cookbook.auto_fin.pipeline import AutoFinPipelineStep
from reme.steps.cookbook.dingtalk.send import DingTalkMarkdownSendStep

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
                "adjustment": "raw_with_explicit_return_adjustment",
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
        adjustment="raw_with_explicit_return_adjustment",
        market_data_complete=True,
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


def test_daily_cookbook_wires_auto_fin_cc_skill_memory_and_crons(monkeypatch):
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
    assert wrapper["job_tools"] == ["memory"]
    assert "memory" in config["jobs"]
    assert config["jobs"]["auto_fin"]["parameters"]["properties"]["force"]["default"] is True

    steps = config["jobs"]["auto_fin"]["steps"]
    assert steps[0] == {
        "backend": "auto_fin_pipeline_step",
        "agent_wrapper": "auto_fin",
    }
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
async def test_pipeline_writes_four_reports_and_skips_same_run(tmp_path: Path):
    app_context = ApplicationContext(
        workspace_dir=str(tmp_path),
        timezone="Asia/Shanghai",
    )
    wrapper = _AutoFinAnalysisWrapper(app_context=app_context)
    step = AutoFinPipelineStep(app_context=app_context, agent_wrapper=wrapper)
    step.logger = MagicMock()
    response = await step(
        date="2026-07-23",
        checkpoint="0900",
        trade_dates=["2026-07-22", "2026-07-23", "2026-07-24"],
        research_only=True,
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
    portfolio = load_document(day_dir / "portfolio.md")
    PortfolioDocument.model_validate(portfolio.metadata)
    assert len(portfolio.metadata["runs"]) == 1
    assert portfolio.metadata["runs"][0]["snapshot"]["nav"] == 1.0
    assert not list((tmp_path / "metadata" / "auto-fin" / "locks").glob("*.lock"))
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "checkpoint start run_id=2026-07-23T0900+08:00" in logs
    assert "analysis start stage=" in logs
    assert "settlements applied run_id=2026-07-23T0900+08:00" in logs
    assert "report persistence done run_id=2026-07-23T0900+08:00 reports=4" in logs
    assert "pipeline done run_id=2026-07-23T0900+08:00 status=COMPLETE" in logs

    rerun = await step(
        date="2026-07-23",
        checkpoint="0900",
        trade_dates=["2026-07-22", "2026-07-23", "2026-07-24"],
        force=False,
        research_only=True,
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
    )
    assert "skipped" not in forced_by_default.metadata
    assert len(load_document(day_dir / "portfolio.md").metadata["runs"]) == 1
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "checkpoint start run_id=2026-07-23T0900+08:00" in logs
    assert "timezone=Asia/Shanghai force=True" in logs


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
