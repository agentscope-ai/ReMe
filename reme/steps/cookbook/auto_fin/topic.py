"""Build the current Auto Fin topic timelines."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time

from ....components import R
from ....schema import AutoFinTopicsOutput
from .analysis import AutoFinAgentStep
from .data import _write_jsonl


@R.register("auto_fin_topic_step")
class AutoFinTopicStep(AutoFinAgentStep):
    """Filter the current window and ask one Agent to organize topic timelines."""

    async def _current_news(self, start: datetime, end: datetime) -> list[dict]:
        news = {}
        first_day = date.fromisoformat(str(self._required("auto_fin_news_start")))
        last_day = date.fromisoformat(str(self._required("auto_fin_date")))
        for day in self._days(first_day, last_day):
            for row in await self._read_jsonl(self._news_path(day)):
                published_at = self._published_at(row, start.tzinfo)
                if published_at is None or not start < published_at <= end:
                    continue
                title = str(row.get("title") or "").strip()
                news_id = hashlib.sha256(f"{published_at.isoformat()}|{title}".encode()).hexdigest()[:16]
                news[news_id] = {
                    "news_id": news_id,
                    "event_time": published_at.isoformat(),
                    "title": title,
                    "content": str(row.get("content") or "").strip()[:4000],
                }
        return sorted(news.values(), key=lambda row: (row["event_time"], row["news_id"]))

    async def execute(self):
        assert self.context is not None
        decision_at = datetime.fromisoformat(str(self._required("auto_fin_decision_at")))
        previous = date.fromisoformat(str(self._required("auto_fin_previous_trade_date")))
        window_start = datetime.combine(previous, time(15), decision_at.tzinfo)
        news = await self._current_news(window_start, decision_at)
        self.logger.info(
            f"[{self.name}] start window=({window_start.isoformat()},{decision_at.isoformat()}] news={len(news)}",
        )
        output = await self._reply(
            "topic_user",
            AutoFinTopicsOutput,
            window_start=window_start.isoformat(),
            decision_at=decision_at.isoformat(),
            news=json.dumps(news, ensure_ascii=False, separators=(",", ":")),
        )
        for events in output.topics.values():
            if any(not window_start < event.event_time <= decision_at for event in events):
                raise ValueError("Topic Agent returned an event outside the news window")
        day_dir = self.workspace_path / str(self.config_value("daily_dir")) / decision_at.date().isoformat()
        _write_jsonl(day_dir / "auto_fin_news.jsonl", news)
        self.context["auto_fin_window_start"] = window_start.isoformat()
        self.context["auto_fin_topics"] = output.model_dump(mode="json")["topics"]
        self.context.response.metadata.update({"news_count": len(news), "topic_count": len(output.topics)})
        self.logger.info(
            f"[{self.name}] done topics={len(output.topics)} "
            f"events={sum(len(events) for events in output.topics.values())} path={day_dir / 'auto_fin_news.jsonl'}",
        )
        return self.context.response
