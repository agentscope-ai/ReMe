"""Forecast one selected ETF from calculated historical samples."""

from __future__ import annotations

import json

from ....components import R
from ....schema import (
    AutoFinEtfHistoricalResearch,
    AutoFinEtfSelection,
    AutoFinMarketSelection,
    AutoFinSelectedEtfAnalysis,
    AutoFinSelectedEvent,
)
from ._base import AutoFinStep, _write


@R.register("auto_fin_market_step")
class AutoFinMarketStep(AutoFinStep):
    """Collect similarity judgments and calculate one ETF forecast."""

    @staticmethod
    def _calculate_analysis(
        item: AutoFinEtfSelection,
        history: AutoFinEtfHistoricalResearch,
        selection: AutoFinMarketSelection,
    ) -> AutoFinSelectedEtfAnalysis:
        """Build all deterministic market fields from Agent-selected news IDs."""
        history_by_news_id = {event.news_id: event for event in history.historical_events}
        unknown_news_ids = {
            match.news_id for match in selection.matched_historical_events if match.news_id not in history_by_news_id
        }
        if unknown_news_ids:
            raise ValueError(f"Market Agent referenced unknown historical news: {sorted(unknown_news_ids)}")

        selected = [
            (match, min(1.0, max(-1.0, match.similarity)))
            for match in selection.matched_historical_events
            if min(1.0, max(-1.0, match.similarity)) != 0
        ]
        total_similarity = sum(abs(similarity) for _, similarity in selected)
        matches = [
            {
                "reason": match.reason,
                "news_id": match.news_id,
                "event_time": history_by_news_id[match.news_id].event_time,
                "similarity": similarity,
                "weight": abs(similarity) / total_similarity,
            }
            for match, similarity in selected
        ]

        returns = []
        has_missing_horizon = False
        has_direction_conflict = False
        for horizon in range(1, 11):
            available = []
            for match, similarity in selected:
                event = history_by_news_id[match.news_id]
                point = next((point for point in event.future_returns if point.horizon == horizon), None)
                if point is not None:
                    direction = 1.0 if similarity > 0 else -1.0
                    available.append((abs(similarity), direction * point.cumulative_return))
            if not available:
                has_missing_horizon = True
                expected_return = None
            else:
                horizon_similarity = sum(similarity for similarity, _ in available)
                expected_return = (
                    sum(similarity * cumulative_return for similarity, cumulative_return in available)
                    / horizon_similarity
                )
                values = [cumulative_return for _, cumulative_return in available]
                has_direction_conflict |= any(value > 0 for value in values) and any(value < 0 for value in values)
            returns.append({"horizon": horizon, "expected_return": expected_return})

        positive_returns = [point for point in returns if (point["expected_return"] or 0) > 0]
        suggested_holding_days = (
            max(positive_returns, key=lambda point: (point["expected_return"], -point["horizon"]))["horizon"]
            if positive_returns
            else None
        )
        limitations = list(history.limitations)
        if not selected:
            limitations.append("没有匹配的历史事件")
        elif len(selected) < 2:
            limitations.append("相似历史样本少于 2 个")
        if has_missing_horizon:
            limitations.append("部分持有期缺少可用历史收益")
        if has_direction_conflict:
            limitations.append("相似历史样本的收益方向存在分歧")
        if selected and suggested_holding_days is None:
            limitations.append("加权预期收益没有正值")

        return AutoFinSelectedEtfAnalysis.model_validate(
            {
                "etf_code": item.etf_code,
                "etf_name": item.etf_name,
                "matched_historical_events": matches,
                "forecast": {
                    "returns": returns,
                    "suggested_holding_days": suggested_holding_days,
                },
                "limitations": list(dict.fromkeys(limitations)),
            },
        )

    async def execute(self):
        assert self.context is not None
        item = AutoFinEtfSelection.model_validate(self._required("auto_fin_current_etf"))
        events = [AutoFinSelectedEvent.model_validate(event) for event in self._required("auto_fin_current_events")]
        history = AutoFinEtfHistoricalResearch.model_validate(self._required("auto_fin_current_history"))
        index = int(self._required("auto_fin_current_index"))
        event_lines = "\n".join(
            f"- [{event.event_time.isoformat()}] {event.event_title or event.reason}: {event.event_content}"
            for event in events
        )
        selection, selection_path = await self._reply(
            "market_user",
            f"auto_fin_market_{index:02d}_{item.etf_code}",
            AutoFinMarketSelection,
            etf_code=item.etf_code,
            etf_name=item.etf_name,
            events=event_lines,
            history_path=str(self._required("auto_fin_current_history_resource")),
            decision_at=str(self._required("auto_fin_decision_at")),
        )
        analysis = self._calculate_analysis(item, history, selection)
        _write(
            selection_path,
            json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        )
        self.context["auto_fin_current_analysis"] = analysis.model_dump(mode="json")
        return self.context.response
