"""Event-analysis step for Auto Fin."""

from __future__ import annotations

from datetime import date, datetime

from .....components import R
from .....schema import EventAnalysisOutput
from ._base import AutoFinAnalysisStep, json_text


@R.register("auto_fin_event_step")
class AutoFinEventStep(AutoFinAnalysisStep):
    """Generate and validate cutoff-bounded event analysis."""

    @staticmethod
    def validate_output(output: EventAnalysisOutput, cutoff: datetime) -> None:
        """Reject event information unavailable at the checkpoint cutoff."""
        if output.window.end_inclusive != cutoff:
            raise ValueError("event window must end exactly at data_cutoff")
        if any(event.published_at > cutoff or not event.known_before_cutoff for event in output.events):
            raise ValueError(
                "event analysis contains information unavailable at data_cutoff",
            )
        for source in output.sources:
            if isinstance(source.max_timestamp, datetime) and source.max_timestamp > cutoff:
                raise ValueError("event source contains data after data_cutoff")
            if (
                isinstance(source.max_timestamp, date)
                and not isinstance(source.max_timestamp, datetime)
                and source.max_timestamp > cutoff.date()
            ):
                raise ValueError("event source contains data after data_cutoff")

    async def execute(self):
        run_context = self.state("run_context")
        if not isinstance(run_context, dict):
            raise RuntimeError("Auto Fin run context is missing before event analysis")
        output, error = await self.reply(
            "event_user",
            EventAnalysisOutput,
            run_context=json_text(run_context),
        )
        if output is not None:
            try:
                self.validate_output(
                    output,
                    datetime.fromisoformat(str(run_context["data_cutoff"])),
                )
            except (KeyError, ValueError) as exc:
                output, error = None, str(exc)
                self.logger.warning(f"[{self.name}] event output rejected: {error}")
        self.set_state("event_output", output)
        self.set_state("event_error", error)
        self.logger.info(
            f"[{self.name}] event analysis done valid={output is not None}",
        )
        assert self.context is not None
        return self.context.response
