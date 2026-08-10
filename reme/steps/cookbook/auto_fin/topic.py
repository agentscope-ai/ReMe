"""Select CLS news that is semantically related to configured topics."""

from __future__ import annotations

import json

from ....components import R
from ....schema import AutoFinTopicOutput
from ._base import AutoFinStep


@R.register("auto_fin_topic_step")
class AutoFinTopicStep(AutoFinStep):
    """Filter current news in bounded Agent batches without writing files."""

    async def execute(self):
        assert self.context is not None
        news = list(self._required("auto_fin_news"))
        topics = list(self._required("auto_fin_topics"))
        batch_size = max(1, int(self._value("topic_batch_size", 50)))
        selected: set[str] = set()
        for start in range(0, len(news), batch_size):
            batch = [
                {**row, "content": str(row.get("content") or "")[:1000]} for row in news[start : start + batch_size]
            ]
            output = await self._reply(
                "topic_user",
                AutoFinTopicOutput,
                topics=json.dumps(topics, ensure_ascii=False),
                news=json.dumps(batch, ensure_ascii=False),
            )
            selected.update(output.news_ids)
        relevant = [row for row in news if row["news_id"] in selected]
        self.context["auto_fin_selected_news"] = relevant
        self.context.response.metadata["relevant_news_count"] = len(relevant)
        if not relevant:
            reason = f"最近24小时没有与 {', '.join(topics)} 相关的财联社新闻。"
            self.context["auto_fin_skipped"] = True
            self.context.response.answer = reason
            self.context.response.metadata.update({"skipped": True, "skip_reason": reason})
        return self.context.response
