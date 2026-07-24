"""Orchestrate historical research and market analysis for selected ETFs."""

from __future__ import annotations

from ....components import R
from ....schema import (
    AutoFinEtfHistoricalResearch,
    AutoFinEtfHistoryDetail,
    AutoFinEtfSelection,
    AutoFinSelectedEtfAnalysis,
    AutoFinSelectedEvent,
)
from ._base import AutoFinStep, _write_jsonl

DEFAULT_DISPATCH_STEPS = ["auto_fin_history_search_step", "auto_fin_market_step"]


@R.register("auto_fin_history_step")
class AutoFinHistoryStep(AutoFinStep):
    """Dispatch historical and market steps for each selected ETF."""

    def __init__(self, *args, dispatch_steps=None, **kwargs):
        super().__init__(
            *args,
            dispatch_steps=DEFAULT_DISPATCH_STEPS if dispatch_steps is None else dispatch_steps,
            **kwargs,
        )

    async def execute(self):
        assert self.context is not None
        etfs = [AutoFinEtfSelection.model_validate(item) for item in self._required("auto_fin_etfs")]
        history_details = []
        news_rows = await self._read_jsonl(self.workspace_path / str(self._required("auto_fin_filtered_news")))
        news_by_id = {str(row["news_id"]): row for row in news_rows}
        current_keys = (
            "auto_fin_current_history",
            "auto_fin_current_history_resource",
            "auto_fin_current_analysis",
        )
        self.logger.info(f"[{self.name}] start etfs={len(etfs)}")
        for index, item in enumerate(etfs, 1):
            label = f"{item.etf_code}({item.etf_name})"
            self.logger.info(
                f"[{self.name}] etf start index={index}/{len(etfs)} etf={label!r} events={len(item.events)}",
            )
            events = [
                AutoFinSelectedEvent(
                    reason=event.reason,
                    news_id=event.news_id,
                    event_time=news_by_id[event.news_id]["event_time"],
                    event_title=str(news_by_id[event.news_id].get("title") or ""),
                    event_content=str(
                        news_by_id[event.news_id].get("content") or news_by_id[event.news_id].get("title") or "",
                    ),
                )
                for event in item.events
            ]
            for key in current_keys:
                if key in self.context:
                    del self.context[key]
            await self.dispatch_steps(
                self.dispatch_step_specs,
                agent_wrapper=self.agent_wrapper,
                auto_fin_current_index=index,
                auto_fin_current_etf=item.model_dump(mode="json"),
                auto_fin_current_events=[event.model_dump(mode="json") for event in events],
            )
            history = AutoFinEtfHistoricalResearch.model_validate(self._required("auto_fin_current_history"))
            analysis = AutoFinSelectedEtfAnalysis.model_validate(self._required("auto_fin_current_analysis"))
            detail = AutoFinEtfHistoryDetail(
                etf=item,
                current_events=events,
                historical_research=history,
                market_analysis=analysis,
            )
            history_details.append(detail.model_dump(mode="json"))
            self.logger.info(
                f"[{self.name}] etf done index={index}/{len(etfs)} etf={label!r}",
            )
        for key in current_keys:
            if key in self.context:
                del self.context[key]
        history_path = (
            self.workspace_path / "resource" / str(self._required("auto_fin_date")) / "auto_fin_history_output.jsonl"
        )
        _write_jsonl(history_path, history_details)
        self.context["auto_fin_history_details"] = history_details
        self.context["auto_fin_history_resource"] = str(history_path)
        self.context.response.metadata["analysis_count"] = len(history_details)
        self.logger.info(
            f"[{self.name}] done analyses={len(history_details)} history_resource={history_path}",
        )
        return self.context.response
