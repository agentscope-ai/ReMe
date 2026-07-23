"""US-correlation analysis step for Auto Fin."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .....components import R
from .....schema import Checkpoint, UsCorrelationAnalysisOutput
from .._common import find_run
from ._base import AutoFinAnalysisStep, json_text


@R.register("auto_fin_us_correlation_step")
class AutoFinUsCorrelationStep(AutoFinAnalysisStep):
    """Generate the opening US analysis or reuse it later the same day."""

    @staticmethod
    def validate_output(
        output: UsCorrelationAnalysisOutput,
        *,
        as_of: datetime,
        trade_date: date,
    ) -> None:
        """Require alignment with the A-share opening checkpoint."""
        if output.as_of != as_of or output.a_share_trade_date != trade_date:
            raise ValueError(
                "US correlation analysis is misaligned with the A-share checkpoint",
            )
        if set(output.lookbacks) != {"1D", "5D", "30D"}:
            raise ValueError(
                "US correlation analysis must contain 1D, 5D, and 30D lookbacks",
            )

    async def execute(self):
        run_context = self.state("run_context")
        checkpoint = self.state("checkpoint")
        trade_date = self.state("trade_date")
        open_data_cutoff = self.state("open_data_cutoff")
        if (
            not isinstance(run_context, dict)
            or not isinstance(checkpoint, Checkpoint)
            or not isinstance(trade_date, date)
            or not isinstance(open_data_cutoff, datetime)
        ):
            raise RuntimeError(
                "Auto Fin opening context is missing before US correlation analysis",
            )

        output: UsCorrelationAnalysisOutput | None
        error = ""
        generated_at: datetime | None = None
        if checkpoint is Checkpoint.OPEN:
            self.logger.info(
                f"[{self.name}] generate opening US correlation analysis trade_date={trade_date.isoformat()}",
            )
            quant_ranking = (self.state("quant_rankings") or {}).get("us_correlation")
            output, error = await self.reply(
                "us_user",
                UsCorrelationAnalysisOutput,
                run_context=json_text(run_context),
                quant_research=json_text(
                    quant_ranking.model_dump(mode="json") if quant_ranking is not None else None,
                ),
            )
            if output is not None:
                try:
                    self.validate_output(
                        output,
                        as_of=open_data_cutoff,
                        trade_date=trade_date,
                    )
                    timezone = self.state("timezone")
                    generated_at = datetime.now(
                        timezone if isinstance(timezone, ZoneInfo) else None,
                    )
                except ValueError as exc:
                    output, error = None, str(exc)
                    self.logger.warning(
                        f"[{self.name}] US correlation output rejected: {error}",
                    )
            if output is not None and quant_ranking is not None:
                output = output.model_copy(update={"ranking": quant_ranking})
        else:
            path = self.state("us_path")
            open_run_id = self.state("open_run_id")
            if not isinstance(path, Path) or not isinstance(open_run_id, str):
                raise RuntimeError("Auto Fin opening report context is missing")
            self.logger.info(
                f"[{self.name}] reuse opening US correlation analysis run_id={open_run_id} "
                f"checkpoint={checkpoint.value}",
            )
            saved = find_run(path, open_run_id)
            try:
                output = UsCorrelationAnalysisOutput.model_validate(
                    (saved or {})["analysis"],
                )
                generated_at = datetime.fromisoformat(
                    str((saved or {})["generated_at"]),
                )
                self.validate_output(
                    output,
                    as_of=open_data_cutoff,
                    trade_date=trade_date,
                )
            except (KeyError, TypeError, ValueError) as exc:
                output, error = None, f"09:00 US analysis unavailable: {exc}"
                self.logger.warning(
                    f"[{self.name}] opening US correlation analysis unavailable: {exc}",
                )

        self.set_state("us_output", output)
        self.set_state("us_error", error)
        self.set_state("us_generated_at", generated_at)
        self.logger.info(
            f"[{self.name}] US correlation analysis done valid={output is not None} "
            f"reused={checkpoint is not Checkpoint.OPEN}",
        )
        assert self.context is not None
        return self.context.response
