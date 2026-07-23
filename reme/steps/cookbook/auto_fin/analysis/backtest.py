"""Backtest and market-analysis step for Auto Fin."""

from __future__ import annotations

from datetime import datetime

from .....components import R
from .....schema import (
    ActionStatus,
    ActionType,
    BacktestAnalysisOutput,
    PortfolioSnapshot,
)
from ._base import AutoFinAnalysisStep, json_text


@R.register("auto_fin_backtest_step")
class AutoFinBacktestStep(AutoFinAnalysisStep):
    """Generate and validate reproducible return intervals."""

    @staticmethod
    def validate_output(output: BacktestAnalysisOutput, cutoff: datetime) -> None:
        """Reject marks outside the checkpoint market-data contract."""
        if output.market_cutoff != cutoff:
            raise ValueError(
                "backtest market_cutoff does not match the checkpoint contract",
            )
        marks = [*output.settlement_marks, *output.position_marks]
        if any(mark.interval_end > cutoff for mark in marks):
            raise ValueError("backtest contains a return interval after market_cutoff")
        ids = [mark.interval_id for mark in marks]
        if len(ids) != len(set(ids)):
            raise ValueError("backtest position mark interval_ids must be unique")

    @staticmethod
    def validate_required_marks(
        snapshot: PortfolioSnapshot,
        output: BacktestAnalysisOutput,
        *,
        settlement_fill_at: datetime,
        market_cutoff: datetime,
    ) -> None:
        """Require exact return intervals for every position affected by settlement."""

        def require_mark(
            marks_by_code: dict[str, list],
            code: str,
            interval_start: datetime,
            interval_end: datetime,
            phase: str,
        ) -> None:
            if not any(
                mark.interval_start == interval_start and mark.interval_end == interval_end
                for mark in marks_by_code.get(code, [])
            ):
                raise ValueError(
                    f"{phase} mark for {code} must cover "
                    f"{interval_start.isoformat()} to {interval_end.isoformat()}",
                )

        settlement_by_code: dict[str, list] = {}
        position_by_code: dict[str, list] = {}
        for mark in output.settlement_marks:
            settlement_by_code.setdefault(mark.code, []).append(mark)
        for mark in output.position_marks:
            position_by_code.setdefault(mark.code, []).append(mark)

        for position in snapshot.positions:
            if position.marked_at is None:
                raise ValueError(f"position {position.code} is missing marked_at")
            if position.marked_at > settlement_fill_at:
                raise ValueError(
                    f"position {position.code} is marked after the settlement fill",
                )
            if position.marked_at < settlement_fill_at:
                require_mark(
                    settlement_by_code,
                    position.code,
                    position.marked_at,
                    settlement_fill_at,
                    "settlement",
                )

        if market_cutoff <= settlement_fill_at:
            return
        proposed = [action for action in snapshot.proposed_actions if action.status is ActionStatus.PROPOSED]
        sell_codes = {action.code for action in proposed if action.action is ActionType.SELL}
        buy_codes = {action.code for action in proposed if action.action is ActionType.BUY}
        post_fill_codes = {position.code for position in snapshot.positions if position.code not in sell_codes}
        post_fill_codes.update(buy_codes)
        for code in post_fill_codes:
            require_mark(
                position_by_code,
                code,
                settlement_fill_at,
                market_cutoff,
                "post-settlement",
            )

    async def execute(self):
        run_context = self.state("run_context")
        snapshot = self.state("snapshot")
        if not isinstance(run_context, dict) or not isinstance(
            snapshot,
            PortfolioSnapshot,
        ):
            raise RuntimeError(
                "Auto Fin run context and snapshot are required before backtest analysis",
            )
        quant_ranking = (self.state("quant_rankings") or {}).get("backtest")
        output, error = await self.reply(
            "backtest_user",
            BacktestAnalysisOutput,
            run_context=json_text(run_context),
            portfolio=json_text(snapshot.model_dump(mode="json")),
            quant_research=json_text(
                quant_ranking.model_dump(mode="json") if quant_ranking is not None else None,
            ),
        )
        if output is not None:
            try:
                market_cutoff = datetime.fromisoformat(
                    str(run_context["market_cutoff"]),
                )
                self.validate_output(output, market_cutoff)
                if output.market_data_complete:
                    self.validate_required_marks(
                        snapshot,
                        output,
                        settlement_fill_at=datetime.fromisoformat(
                            str(run_context["settlement_fill_at"]),
                        ),
                        market_cutoff=market_cutoff,
                    )
            except (KeyError, ValueError) as exc:
                output, error = None, str(exc)
                self.logger.warning(f"[{self.name}] backtest output rejected: {error}")
        if output is not None and quant_ranking is not None:
            output = output.model_copy(update={"ranking": quant_ranking})
        self.set_state("backtest_output", output)
        self.set_state("backtest_error", error)
        self.logger.info(
            f"[{self.name}] backtest analysis done valid={output is not None} "
            f"market_data_complete={bool(output and output.market_data_complete)}",
        )
        assert self.context is not None
        return self.context.response
