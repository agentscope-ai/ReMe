"""Forecast one selected ETF from calculated historical samples."""

from __future__ import annotations

from ....components import R
from ....schema import (
    AutoFinEtfHistoricalResearch,
    AutoFinEtfSelection,
    AutoFinSelectedEtfAnalysis,
    AutoFinSelectedEvent,
)
from ._base import AutoFinStep


@R.register("auto_fin_market_step")
class AutoFinMarketStep(AutoFinStep):
    """Run and validate market analysis for the current ETF."""

    async def execute(self):
        assert self.context is not None
        item = AutoFinEtfSelection.model_validate(self._required("auto_fin_current_etf"))
        events = [AutoFinSelectedEvent.model_validate(event) for event in self._required("auto_fin_current_events")]
        history = AutoFinEtfHistoricalResearch.model_validate(self._required("auto_fin_current_history"))
        index = int(self._required("auto_fin_current_index"))
        label = f"{item.etf_code}({item.etf_name})"
        event_lines = "\n".join(
            f"- [{event.event_time.isoformat()}] {event.event_title or event.reason}: {event.event_content}"
            for event in events
        )
        analysis, _ = await self._reply(
            "market_user",
            f"auto_fin_market_{index:02d}_{item.etf_code}",
            AutoFinSelectedEtfAnalysis,
            etf_code=item.etf_code,
            etf_name=item.etf_name,
            events=event_lines,
            history_path=str(self._required("auto_fin_current_history_resource")),
            decision_at=str(self._required("auto_fin_decision_at")),
        )
        if (analysis.etf_code, analysis.etf_name) != (item.etf_code, item.etf_name):
            raise ValueError(f"Market Agent changed ETF {label!r}")
        historical_event_times = {event.event_time for event in history.historical_events}
        if any(match.event_time not in historical_event_times for match in analysis.matched_historical_events):
            raise ValueError("Market Agent referenced an unknown historical event")
        analysis.limitations = list(dict.fromkeys([*history.limitations, *analysis.limitations]))
        self.context["auto_fin_current_analysis"] = analysis.model_dump(mode="json")
        return self.context.response
