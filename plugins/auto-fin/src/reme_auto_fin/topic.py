"""Select CLS news that is semantically related to configured topics."""

from __future__ import annotations

import json
import re
from time import perf_counter

from .base import AGENT_INPUT_LOG_LIMIT, AGENT_OUTPUT_LOG_LIMIT, AutoFinStep


class AutoFinTopicStep(AutoFinStep):
    """Filter current news in bounded Agent batches without writing files."""

    @staticmethod
    def _parse_news_ids(value: object) -> list[str]:
        if not isinstance(value, str):
            raise ValueError("Auto Fin topic Agent returned no text")
        match = re.search(r"```json\s*(.*?)```", value, re.IGNORECASE | re.DOTALL)
        if match is None:
            raise ValueError("Auto Fin topic Agent returned no JSON code block")
        try:
            ids = json.loads(match.group(1).strip())
        except json.JSONDecodeError as exc:
            raise ValueError("Auto Fin topic Agent returned invalid JSON") from exc
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError("Auto Fin topic Agent must return a JSON array of strings")
        return ids

    async def _select_news_ids(self, prompt: str) -> list[str]:
        """Request and validate plain-text topic IDs, retrying one malformed reply."""
        if self.agent_wrapper is None:
            raise RuntimeError("Auto Fin analysis requires an agent_wrapper")
        self.logger.info(
            f"[{self.name}] agent input prompt=topic_user query={self._preview(prompt, AGENT_INPUT_LOG_LIMIT)}",
        )
        for attempt in range(2):
            started_at = perf_counter()
            result = await self.agent_wrapper.reply(prompt)
            try:
                ids = self._parse_news_ids(result.get("result") if isinstance(result, dict) else None)
            except ValueError as exc:
                if attempt:
                    raise ValueError(f"Auto Fin topic Agent returned invalid news IDs: {exc}") from exc
                self.logger.warning(f"[{self.name}] invalid topic JSON; retrying once: {exc}")
                continue
            self.logger.info(
                f"[{self.name}] agent output prompt=topic_user elapsed={perf_counter() - started_at:.2f}s "
                f"output={self._preview(ids, AGENT_OUTPUT_LOG_LIMIT)}",
            )
            return ids
        raise RuntimeError("Auto Fin topic Agent produced no response")

    async def execute(self):
        """Select relevant news from each batch for the current invocation."""
        assert self.context is not None
        news = list(self._required("auto_fin_news"))
        topics = list(self._required("auto_fin_topics"))
        window_hours = float(self._value("auto_fin_window_hours", 24))
        formatted_hours = f"{window_hours:g}"
        batch_size = max(1, int(self._value("topic_batch_size", 50)))
        selected: set[str] = set()
        for start in range(0, len(news), batch_size):
            batch = [
                {**row, "content": str(row.get("content") or "")[:1000]} for row in news[start : start + batch_size]
            ]
            prompt = self.prompt_format(
                "topic_user",
                topics=json.dumps(topics, ensure_ascii=False),
                news=json.dumps(batch, ensure_ascii=False),
                window_hours=formatted_hours,
            )
            selected.update(await self._select_news_ids(prompt))
        relevant = [row for row in news if row["news_id"] in selected]
        self.context["auto_fin_selected_news"] = relevant
        self.context.response.metadata["relevant_news_count"] = len(relevant)
        if not relevant:
            reason = f"最近{formatted_hours}小时没有与 {', '.join(topics)} 相关的财联社新闻。"
            self.context["auto_fin_skipped"] = True
            self.context.response.answer = reason
            self.context.response.metadata.update({"skipped": True, "skip_reason": reason})
        return self.context.response
