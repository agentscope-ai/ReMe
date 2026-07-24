"""Research historical analogues and ETF reactions for each topic."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime

from ....components import R
from ....schema import AutoFinHistoricalResearch, AutoFinTopicAnalysis
from .analysis import AutoFinAgentStep


@R.register("auto_fin_history_step")
class AutoFinHistoryStep(AutoFinAgentStep):
    """Run one fresh historical-market Agent for each current topic."""

    async def execute(self):
        assert self.context is not None
        topics = dict(self._required("auto_fin_topics"))
        analyses = []
        window_start = datetime.fromisoformat(str(self._required("auto_fin_window_start")))
        self.logger.info(f"[{self.name}] start topics={len(topics)}")
        for index, (topic, events) in enumerate(topics.items(), 1):
            self.logger.info(
                f"[{self.name}] topic start index={index}/{len(topics)} topic={topic!r} events={len(events)}",
            )
            history = await self._reply(
                "history_search_user",
                AutoFinHistoricalResearch,
                topic=topic,
                events=json.dumps(events, ensure_ascii=False, separators=(",", ":")),
                window_start=str(self._required("auto_fin_window_start")),
                workspace_root=str(self.workspace_path),
            )
            if history.topic != topic:
                raise ValueError(f"History Agent changed topic {topic!r} to {history.topic!r}")
            if any(event.event_time >= window_start for event in history.historical_events):
                raise ValueError("History Agent returned an event inside the current news window")
            self.logger.info(
                f"[{self.name}] history ready topic={topic!r} events={len(history.historical_events)} "
                f"limitations={len(history.limitations)}",
            )
            with tempfile.TemporaryDirectory(prefix="auto-fin-") as temporary_dir:
                analysis = await self._reply(
                    "market_user",
                    AutoFinTopicAnalysis,
                    topic=topic,
                    events=json.dumps(events, ensure_ascii=False, separators=(",", ":")),
                    history=history.model_dump_json(),
                    decision_at=str(self._required("auto_fin_decision_at")),
                    temporary_dir=temporary_dir,
                )
            if analysis.topic != topic:
                raise ValueError(f"Market Agent changed topic {topic!r} to {analysis.topic!r}")
            if analysis.historical_events != history.historical_events:
                raise ValueError("Market Agent changed the verified historical events")
            analysis.limitations = list(dict.fromkeys([*history.limitations, *analysis.limitations]))
            analyses.append(analysis.model_dump(mode="json"))
            self.logger.info(
                f"[{self.name}] topic done index={index}/{len(topics)} topic={topic!r} etfs={len(analysis.etfs)}",
            )
        self.context["auto_fin_topic_analyses"] = analyses
        self.context.response.metadata["analysis_count"] = len(analyses)
        self.logger.info(
            f"[{self.name}] done analyses={len(analyses)} etfs={sum(len(item['etfs']) for item in analyses)}",
        )
        return self.context.response
