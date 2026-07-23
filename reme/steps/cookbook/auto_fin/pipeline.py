"""Unified, serial Auto Fin checkpoint pipeline."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from ....components import R
from ....schema import (
    BacktestAnalysisRun,
    Checkpoint,
    EventAnalysisRun,
    PortfolioMetrics,
    PortfolioRun,
    PortfolioSnapshot,
    RunStatus,
    UpstreamAnalysis,
    UsCorrelationAnalysisRun,
)
from ...base_step import BaseStep
from ...file_io import refresh_day_index
from ._common import (
    analysis_section,
    checkpoint_lock,
    find_run,
    latest_portfolio_snapshot,
    load_document,
    portfolio_section,
    upsert_report,
    write_atomic,
)
from .analysis import (
    AutoFinBacktestStep,
    AutoFinEventStep,
    AutoFinPortfolioStep,
    AutoFinUsCorrelationStep,
)
from .ledger import AutoFinLedger, next_trade_date


@dataclass(frozen=True)
class _Schedule:
    """Resolved market times for one checkpoint."""

    decision_at: datetime
    data_cutoff: datetime
    market_cutoff: datetime
    settlement_trade_date: date
    settlement_fill_at: datetime
    settlement_fill_basis: str
    scheduled_fill_at: datetime


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


@R.register("auto_fin_pipeline_step")
class AutoFinPipelineStep(BaseStep):
    """Run all Auto Fin analyses serially, then validate and persist one checkpoint."""

    def _value(self, key: str, default: Any = None) -> Any:
        assert self.context is not None
        return self.context.get(key, self.kwargs.get(key, default))

    @staticmethod
    def _strict_date(value: str) -> date:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc
        if parsed.isoformat() != value:
            raise ValueError("date must be YYYY-MM-DD")
        return parsed

    @staticmethod
    def _run_id(trade_date: date, checkpoint: Checkpoint, timezone: ZoneInfo) -> str:
        offset = datetime.combine(trade_date, time(), timezone).isoformat()[-6:]
        return f"{trade_date.isoformat()}T{checkpoint.value}{offset}"

    @staticmethod
    def _schedule(
        trade_date: date,
        previous_trade_date: date,
        checkpoint: Checkpoint,
        timezone: ZoneInfo,
    ) -> _Schedule:
        def at(day: date, hour: int, minute: int = 0) -> datetime:
            return datetime.combine(day, time(hour, minute), timezone)

        if checkpoint is Checkpoint.OPEN:
            return _Schedule(
                decision_at=at(trade_date, 9),
                data_cutoff=at(trade_date, 9),
                market_cutoff=at(previous_trade_date, 15),
                settlement_trade_date=previous_trade_date,
                settlement_fill_at=at(previous_trade_date, 15),
                settlement_fill_basis="PREVIOUS_1500_CLOSE",
                scheduled_fill_at=at(trade_date, 9, 30),
            )
        if checkpoint is Checkpoint.MIDDAY:
            return _Schedule(
                decision_at=at(trade_date, 11, 45),
                data_cutoff=at(trade_date, 11, 45),
                market_cutoff=at(trade_date, 11, 30),
                settlement_trade_date=trade_date,
                settlement_fill_at=at(trade_date, 9, 30),
                settlement_fill_basis="0930_OPEN",
                scheduled_fill_at=at(trade_date, 13),
            )
        return _Schedule(
            decision_at=at(trade_date, 14, 45),
            data_cutoff=at(trade_date, 14, 45),
            market_cutoff=at(trade_date, 14, 45),
            settlement_trade_date=trade_date,
            settlement_fill_at=at(trade_date, 13),
            settlement_fill_basis="1300_OPEN",
            scheduled_fill_at=at(trade_date, 15),
        )

    @staticmethod
    def _fetch_trade_calendar_sync(
        start: date,
        end: date,
    ) -> tuple[list[date], list[dict[str, Any]]]:
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "TUSHARE_TOKEN is required to validate the A-share trading calendar",
            )
        try:
            import tushare as ts
        except ImportError as exc:
            raise RuntimeError(
                "tushare is required for the Auto Fin trading calendar",
            ) from exc
        frame = ts.pro_api(token).trade_cal(
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="exchange,cal_date,is_open,pretrade_date",
        )
        records = frame.to_dict(orient="records")
        open_dates = [
            datetime.strptime(str(row["cal_date"]), "%Y%m%d").date()
            for row in records
            if int(row.get("is_open", 0)) == 1
        ]
        return sorted(open_dates), records

    async def _trade_calendar(
        self,
        trade_date: date,
        manifest_path: Path,
    ) -> list[date]:
        supplied = self._value("trade_dates", None)
        records: list[dict[str, Any]]
        if supplied is not None:
            if not isinstance(supplied, list):
                raise ValueError("trade_dates must be a list of YYYY-MM-DD strings")
            open_dates = [self._strict_date(str(value)) for value in supplied]
            records = [{"cal_date": value.isoformat(), "is_open": 1} for value in open_dates]
            source = "job_parameter"
        else:
            start, end = trade_date - timedelta(days=45), trade_date + timedelta(
                days=45,
            )
            self.logger.info(
                f"[{self.name}] fetch trading calendar source=tushare.trade_cal "
                f"start={start.isoformat()} end={end.isoformat()}",
            )
            open_dates, records = await asyncio.to_thread(
                self._fetch_trade_calendar_sync,
                start,
                end,
            )
            source = "tushare.trade_cal"
        await write_atomic(
            manifest_path,
            _json(
                {
                    "source": source,
                    "fetched_at": datetime.now().astimezone().isoformat(),
                    "trade_date": trade_date.isoformat(),
                    "rows": records,
                },
            )
            + "\n",
        )
        self.logger.info(
            f"[{self.name}] trading calendar ready source={source} open_dates={len(set(open_dates))} "
            f"manifest={manifest_path.relative_to(self.workspace_path).as_posix()}",
        )
        return sorted(set(open_dates))

    async def _run_analysis_step(self, step_type: type[BaseStep]) -> None:
        """Run one fresh analysis Step against this checkpoint's RuntimeContext."""
        assert self.context is not None
        step = step_type(app_context=self.app_context, agent_wrapper=self.agent_wrapper)
        self.logger.info(f"[{self.name}] analysis start stage={step.name}")
        await step(self.context)
        self.logger.info(f"[{self.name}] analysis done stage={step.name}")

    @staticmethod
    def _common_run(
        run_id: str,
        checkpoint: Checkpoint,
        schedule: _Schedule,
        generated_at: datetime,
        status: RunStatus,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "checkpoint": checkpoint.value,
            "status": status.value,
            "decision_at": schedule.decision_at.isoformat(),
            "data_cutoff": schedule.data_cutoff.isoformat(),
            "generated_at": generated_at.isoformat(),
            "stale": False,
        }

    async def _run_pipeline(
        self,
        *,
        trade_date: date,
        checkpoint: Checkpoint,
        timezone: ZoneInfo,
        open_dates: list[date],
        run_id: str,
        force: bool,
    ) -> None:
        assert self.context is not None
        daily_dir = str(self.config_value("daily_dir")).strip("/")
        day_dir = self.workspace_path / daily_dir / trade_date.isoformat()
        self.logger.info(
            f"[{self.name}] pipeline start run_id={run_id} checkpoint={checkpoint.value} "
            f"trade_date={trade_date.isoformat()} force={force}",
        )
        paths = {
            "event": day_dir / "event_analysis.md",
            "backtest": day_dir / "backtest_analysis.md",
            "us": day_dir / "us_correlation_analysis.md",
            "portfolio": day_dir / "portfolio.md",
        }
        if find_run(paths["portfolio"], run_id) is not None and not force:
            rel = paths["portfolio"].relative_to(self.workspace_path).as_posix()
            self.context["auto_fin_portfolio_path"] = rel
            self.context.response.success = True
            self.context.response.answer = f"Skipped: Auto Fin checkpoint already exists at {rel}"
            self.context.response.metadata.update(
                {"run_id": run_id, "skipped": True, "notify": False},
            )
            self.logger.info(
                f"[{self.name}] pipeline skip existing run_id={run_id} portfolio={rel}",
            )
            return

        previous_dates = [value for value in open_dates if value < trade_date]
        if not previous_dates:
            raise RuntimeError(
                f"no previous A-share trading date is available for {trade_date}",
            )
        previous_trade_date = previous_dates[-1]
        schedule = self._schedule(trade_date, previous_trade_date, checkpoint, timezone)
        self.logger.info(
            f"[{self.name}] schedule resolved run_id={run_id} decision_at={schedule.decision_at.isoformat()} "
            f"market_cutoff={schedule.market_cutoff.isoformat()} "
            f"scheduled_fill_at={schedule.scheduled_fill_at.isoformat()}",
        )
        bootstrap = bool(self._value("bootstrap", True))
        snapshot = latest_portfolio_snapshot(
            self.workspace_path,
            daily_dir,
            before=schedule.decision_at,
            excluding_run_id=run_id,
        )
        snapshot_source = "persisted"
        if snapshot is None:
            portfolio_paths = list(
                (self.workspace_path / daily_dir).glob("*/portfolio.md"),
            )
            has_other_run = any(
                run.get("run_id") != run_id
                for path in portfolio_paths
                for run in load_document(path).metadata.get("runs", [])
            )
            has_unusable_document = bool(portfolio_paths) and not any(
                load_document(path).metadata.get("runs", []) for path in portfolio_paths
            )
            if not bootstrap or checkpoint is not Checkpoint.OPEN or has_other_run or has_unusable_document:
                raise RuntimeError("no previous legal portfolio snapshot is available")
            snapshot = PortfolioSnapshot()
            snapshot_source = "bootstrap"
        self.logger.info(
            f"[{self.name}] portfolio snapshot ready run_id={run_id} source={snapshot_source} "
            f"positions={len(snapshot.positions)} nav={snapshot.nav:.6f} cash_nav={snapshot.cash_nav:.6f}",
        )

        run_context = {
            "run_id": run_id,
            "timezone": str(timezone),
            "trade_date": trade_date.isoformat(),
            "previous_trade_date": previous_trade_date.isoformat(),
            "checkpoint": checkpoint.value,
            "decision_at": schedule.decision_at.isoformat(),
            "data_cutoff": schedule.data_cutoff.isoformat(),
            "market_cutoff": schedule.market_cutoff.isoformat(),
            "settlement_fill_at": schedule.settlement_fill_at.isoformat(),
            "settlement_fill_basis": schedule.settlement_fill_basis,
            "scheduled_fill_at": schedule.scheduled_fill_at.isoformat(),
            "workspace": str(self.workspace_path),
            "event_report": str(paths["event"]),
            "backtest_report": str(paths["backtest"]),
            "us_report": str(paths["us"]),
            "portfolio_report": str(paths["portfolio"]),
        }
        open_schedule = self._schedule(
            trade_date,
            previous_trade_date,
            Checkpoint.OPEN,
            timezone,
        )
        open_run_id = self._run_id(trade_date, Checkpoint.OPEN, timezone)
        self.context.update(
            {
                "auto_fin_run_context": run_context,
                "auto_fin_snapshot": snapshot,
                "auto_fin_checkpoint": checkpoint,
                "auto_fin_trade_date": trade_date,
                "auto_fin_open_data_cutoff": open_schedule.data_cutoff,
                "auto_fin_open_run_id": open_run_id,
                "auto_fin_us_path": paths["us"],
                "auto_fin_timezone": timezone,
            },
        )
        await self._run_analysis_step(AutoFinEventStep)
        await self._run_analysis_step(AutoFinBacktestStep)
        event_output = self.context.get("auto_fin_event_output")
        event_error = str(self.context.get("auto_fin_event_error", ""))
        backtest_output = self.context.get("auto_fin_backtest_output")
        backtest_error = str(self.context.get("auto_fin_backtest_error", ""))
        self.logger.info(
            f"[{self.name}] domestic analyses ready run_id={run_id} event_ok={event_output is not None} "
            f"backtest_ok={backtest_output is not None} "
            f"market_data_complete={bool(backtest_output and backtest_output.market_data_complete)}",
        )

        snapshot_before = snapshot.model_copy(deep=True)
        ledger = AutoFinLedger(
            snapshot,
            max_positions=int(self._value("max_positions", 10)),
            slot_weight=float(self._value("slot_weight", 0.1)),
        )
        settlements = []
        if backtest_output is not None and backtest_output.market_data_complete:
            ledger.apply_marks(backtest_output.settlement_marks)
            settlements = ledger.settle(
                ledger.snapshot.proposed_actions,
                trade_date=schedule.settlement_trade_date,
                eligible_sell_date=next_trade_date(
                    schedule.settlement_trade_date,
                    open_dates,
                ),
                fill_at=schedule.settlement_fill_at,
                fill_basis=schedule.settlement_fill_basis,
                market_data_complete=True,
            )
            ledger.apply_marks(backtest_output.position_marks)
        else:
            settlements = ledger.settle(
                ledger.snapshot.proposed_actions,
                trade_date=schedule.settlement_trade_date,
                eligible_sell_date=next_trade_date(
                    schedule.settlement_trade_date,
                    open_dates,
                ),
                fill_at=schedule.settlement_fill_at,
                fill_basis=schedule.settlement_fill_basis,
                market_data_complete=False,
            )
        settlement_counts: dict[str, int] = {}
        for settlement in settlements:
            settlement_counts[settlement.status.value] = settlement_counts.get(settlement.status.value, 0) + 1
        self.logger.info(
            f"[{self.name}] settlements applied run_id={run_id} total={len(settlements)} "
            f"statuses={settlement_counts} positions={len(ledger.snapshot.positions)}",
        )

        self.context["auto_fin_portfolio_snapshot"] = ledger.snapshot
        await self._run_analysis_step(AutoFinUsCorrelationStep)
        us_output = self.context.get("auto_fin_us_output")
        us_error = str(self.context.get("auto_fin_us_error", ""))
        us_generated_at = self.context.get("auto_fin_us_generated_at")

        errors = [value for value in (event_error, backtest_error, us_error) if value]
        if backtest_output is not None and not backtest_output.market_data_complete:
            errors.append("backtest market data is incomplete")
        status = RunStatus.DEGRADED if errors else RunStatus.COMPLETE
        status_log = self.logger.warning if errors else self.logger.info
        status_log(
            f"[{self.name}] upstream status run_id={run_id} status={status.value} "
            f"error_count={len(errors)} us_ok={us_output is not None}",
        )
        generated_at = datetime.now(timezone)
        analyses = {
            "event": {
                "status": (RunStatus.COMPLETE.value if event_output else RunStatus.FAILED.value),
                "error": event_error,
                "analysis": (event_output.model_dump(mode="json") if event_output else None),
            },
            "backtest": {
                "status": (
                    RunStatus.COMPLETE.value
                    if backtest_output and backtest_output.market_data_complete
                    else (RunStatus.DEGRADED.value if backtest_output else RunStatus.FAILED.value)
                ),
                "error": backtest_error,
                "analysis": (backtest_output.model_dump(mode="json") if backtest_output else None),
            },
            "us_correlation": {
                "status": (RunStatus.COMPLETE.value if us_output else RunStatus.FAILED.value),
                "error": us_error,
                "analysis": us_output.model_dump(mode="json") if us_output else None,
                "reused": checkpoint is not Checkpoint.OPEN,
            },
        }
        self.context["auto_fin_analyses"] = analyses
        await self._run_analysis_step(AutoFinPortfolioStep)
        portfolio_output = self.context.get("auto_fin_portfolio_output")
        portfolio_error = str(self.context.get("auto_fin_portfolio_error", ""))
        if portfolio_output is None:
            raise RuntimeError(
                f"portfolio analysis failed schema validation: {portfolio_error}",
            )

        accepted, rejected = ledger.validate_proposals(
            portfolio_output.actions,
            run_id=run_id,
            trade_date=trade_date,
            proposed_at=generated_at,
            scheduled_fill_at=schedule.scheduled_fill_at,
            run_status=status,
            research_only=bool(self._value("research_only", False)),
            position_data_complete=bool(
                backtest_output and backtest_output.market_data_complete,
            ),
        )
        ledger.snapshot.proposed_actions = [action for action in accepted if action.status.value == "PROPOSED"]
        self.logger.info(
            f"[{self.name}] proposals validated run_id={run_id} received={len(portfolio_output.actions)} "
            f"accepted={len(accepted)} rejected={len(rejected)} "
            f"pending={len(ledger.snapshot.proposed_actions)}",
        )
        interval_return = ledger.snapshot.nav / snapshot_before.nav - 1.0
        common = self._common_run(run_id, checkpoint, schedule, generated_at, status)

        event_run = EventAnalysisRun.model_validate(
            {
                **common,
                "status": (RunStatus.COMPLETE if event_output else RunStatus.FAILED).value,
                "analysis": (event_output.model_dump(mode="json") if event_output else None),
                "error": event_error,
            },
        ).model_dump(mode="json")
        backtest_run = BacktestAnalysisRun.model_validate(
            {
                **common,
                "status": (
                    RunStatus.COMPLETE
                    if backtest_output and backtest_output.market_data_complete
                    else RunStatus.DEGRADED if backtest_output else RunStatus.FAILED
                ).value,
                "analysis": (backtest_output.model_dump(mode="json") if backtest_output else None),
                "error": backtest_error,
            },
        ).model_dump(mode="json")
        upstream = {
            "event": UpstreamAnalysis(
                run_id=run_id,
                status=RunStatus(analyses["event"]["status"]),
                data_cutoff=schedule.data_cutoff,
                generated_at=generated_at,
            ).model_dump(mode="json"),
            "backtest": UpstreamAnalysis(
                run_id=run_id,
                status=RunStatus(analyses["backtest"]["status"]),
                data_cutoff=schedule.market_cutoff,
                generated_at=generated_at,
            ).model_dump(mode="json"),
            "us_correlation": UpstreamAnalysis(
                run_id=open_run_id,
                status=RunStatus(analyses["us_correlation"]["status"]),
                data_cutoff=open_schedule.data_cutoff,
                generated_at=us_generated_at or generated_at,
            ).model_dump(mode="json"),
        }
        portfolio_run = PortfolioRun.model_validate(
            {
                **common,
                "portfolio_before": PortfolioMetrics(
                    nav=snapshot_before.nav,
                    cash_nav=snapshot_before.cash_nav,
                    position_count=len(snapshot_before.positions),
                ).model_dump(mode="json"),
                "settlements": [value.model_dump(mode="json") for value in settlements],
                "positions": [value.model_dump(mode="json") for value in ledger.snapshot.positions],
                "portfolio_after_mark": PortfolioMetrics(
                    nav=ledger.snapshot.nav,
                    cash_nav=ledger.snapshot.cash_nav,
                    position_count=len(ledger.snapshot.positions),
                    interval_return=interval_return,
                ).model_dump(mode="json"),
                "proposed_actions": [value.model_dump(mode="json") for value in accepted],
                "rejected_actions": [value.model_dump(mode="json") for value in rejected],
                "upstream": upstream,
                "snapshot": ledger.snapshot.model_dump(mode="json"),
            },
        ).model_dump(mode="json")

        event_body = (
            analysis_section(
                event_output.description,
                event_output.body,
                event_output.limitations,
            )
            if event_output
            else f"事件分析失败：{event_error}"
        )
        backtest_body = (
            analysis_section(
                backtest_output.description,
                backtest_output.body,
                backtest_output.limitations,
            )
            if backtest_output
            else f"回测分析失败：{backtest_error}"
        )
        portfolio_body = portfolio_section(
            snapshot_before,
            ledger.snapshot,
            portfolio_run["settlements"],
            portfolio_run["proposed_actions"],
            portfolio_run["rejected_actions"],
            portfolio_output.body,
            interval_return=interval_return,
            status=status.value,
            us_as_of=us_output.as_of.isoformat() if us_output else "",
        )
        report_args = {
            "trade_date": trade_date.isoformat(),
            "timezone": str(timezone),
            "checkpoint": checkpoint,
        }
        self.logger.info(
            f"[{self.name}] report persistence start run_id={run_id} directory={day_dir}",
        )
        await upsert_report(
            paths["event"],
            document_type="event_analysis",
            run=event_run,
            section=event_body,
            title=f"{trade_date.isoformat()} Event Analysis",
            **report_args,
        )
        await upsert_report(
            paths["backtest"],
            document_type="backtest_analysis",
            run=backtest_run,
            section=backtest_body,
            title=f"{trade_date.isoformat()} Backtest Analysis",
            **report_args,
        )
        if checkpoint is Checkpoint.OPEN and us_output is not None:
            us_run = UsCorrelationAnalysisRun.model_validate(
                {
                    **common,
                    "analysis": us_output.model_dump(mode="json"),
                    "error": "",
                },
            ).model_dump(mode="json")
            await upsert_report(
                paths["us"],
                document_type="us_correlation_analysis",
                run=us_run,
                section=analysis_section(
                    us_output.description,
                    us_output.body,
                    us_output.limitations,
                ),
                title=f"{trade_date.isoformat()} US Correlation Analysis",
                **report_args,
            )
        await upsert_report(
            paths["portfolio"],
            document_type="portfolio",
            run=portfolio_run,
            section=portfolio_body,
            title=f"{trade_date.isoformat()} Auto Fin Portfolio",
            **report_args,
        )
        report_count = 4 if checkpoint is Checkpoint.OPEN and us_output is not None else 3
        self.logger.info(
            f"[{self.name}] report persistence done run_id={run_id} reports={report_count}",
        )
        try:
            await refresh_day_index(
                SimpleNamespace(workspace_path=self.workspace_path),
                trade_date.isoformat(),
                daily_dir,
            )
            self.logger.info(
                f"[{self.name}] refreshed derived daily index run_id={run_id}",
            )
        except (OSError, UnicodeError, ValueError) as exc:
            # portfolio.md is the commit point; a derived daily index can be rebuilt independently.
            self.logger.warning(
                f"[{self.name}] failed to refresh derived daily index: {exc}",
            )

        rel = paths["portfolio"].relative_to(self.workspace_path).as_posix()
        self.context["auto_fin_portfolio_path"] = rel
        self.context.response.success = status is not RunStatus.FAILED
        self.context.response.answer = f"Generated Auto Fin {checkpoint.value} report: {rel}"
        self.context.response.metadata.update(
            {
                "run_id": run_id,
                "status": status.value,
                "portfolio_path": rel,
                "errors": errors,
                "notify": True,
            },
        )
        self.logger.info(
            f"[{self.name}] pipeline done run_id={run_id} status={status.value} portfolio={rel} "
            f"positions={len(ledger.snapshot.positions)} nav={ledger.snapshot.nav:.6f}",
        )

    async def execute(self):
        assert self.context is not None
        timezone = ZoneInfo(
            (
                self.app_context.app_config.timezone
                if self.app_context and self.app_context.app_config.timezone
                else "Asia/Shanghai"
            ),
        )
        raw_date = str(self._value("date", "") or "").strip()
        trade_date = self._strict_date(raw_date) if raw_date else datetime.now(timezone).date()
        try:
            checkpoint = Checkpoint(str(self._value("checkpoint", "")))
        except ValueError as exc:
            raise ValueError("checkpoint must be one of 0900, 1145, 1445") from exc
        run_id = self._run_id(trade_date, checkpoint, timezone)
        force = bool(self._value("force", True))
        self.logger.info(
            f"[{self.name}] checkpoint start run_id={run_id} trade_date={trade_date.isoformat()} "
            f"checkpoint={checkpoint.value} timezone={timezone} force={force}",
        )

        metadata_root = self.workspace_path / str(self.config_value("metadata_dir")) / "auto-fin"
        resource_root = self.workspace_path / str(self.config_value("resource_dir")) / "auto-fin"
        manifest_path = resource_root / trade_date.isoformat() / "manifests" / f"calendar-{checkpoint.value}.json"
        checkpoint_path = metadata_root / "checkpoints" / f"{run_id}.json"
        lock_path = metadata_root / "locks" / f"{run_id}.lock"
        async with checkpoint_lock(lock_path, run_id):
            self.logger.info(f"[{self.name}] checkpoint lock acquired run_id={run_id}")
            open_dates = await self._trade_calendar(trade_date, manifest_path)
            if trade_date not in open_dates:
                self.context.response.success = True
                self.context.response.answer = f"Skipped: {trade_date.isoformat()} is not an A-share trading day"
                self.context.response.metadata.update(
                    {"run_id": run_id, "skipped": True, "notify": False},
                )
                self.logger.info(
                    f"[{self.name}] checkpoint skip non-trading day run_id={run_id}",
                )
                return self.context.response
            await write_atomic(
                checkpoint_path,
                _json(
                    {
                        "run_id": run_id,
                        "status": "RUNNING",
                        "started_at": datetime.now(timezone).isoformat(),
                    },
                )
                + "\n",
            )
            self.logger.info(
                f"[{self.name}] checkpoint state written run_id={run_id} status=RUNNING",
            )
            try:
                await self._run_pipeline(
                    trade_date=trade_date,
                    checkpoint=checkpoint,
                    timezone=timezone,
                    open_dates=open_dates,
                    run_id=run_id,
                    force=force,
                )
            except Exception as exc:
                await write_atomic(
                    checkpoint_path,
                    _json(
                        {
                            "run_id": run_id,
                            "status": "FAILED",
                            "failed_at": datetime.now(timezone).isoformat(),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    + "\n",
                )
                self.logger.exception(
                    f"[{self.name}] checkpoint failed run_id={run_id} error={type(exc).__name__}: {exc}",
                )
                raise
            final_status = self.context.response.metadata.get("status", "SKIPPED")
            await write_atomic(
                checkpoint_path,
                _json(
                    {
                        "run_id": run_id,
                        "status": final_status,
                        "completed_at": datetime.now(timezone).isoformat(),
                    },
                )
                + "\n",
            )
            self.logger.info(
                f"[{self.name}] checkpoint done run_id={run_id} status={final_status}",
            )
        return self.context.response
