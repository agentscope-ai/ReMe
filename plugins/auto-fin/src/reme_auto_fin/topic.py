"""Select CLS news that is semantically related to configured topics."""

from __future__ import annotations

from collections.abc import Iterator
import json
import re
from time import perf_counter

from .base import AGENT_INPUT_LOG_LIMIT, AGENT_OUTPUT_LOG_LIMIT, AutoFinStep


class AutoFinTopicStep(AutoFinStep):
    """Filter current news in bounded Agent batches without writing files."""

    PROMPT_CHAR_LIMIT = 100_000

    @staticmethod
    def _parse_news_ids(value: object, topics: list[str]) -> dict[str, list[str]]:
        if not isinstance(value, str):
            raise ValueError("Auto Fin topic Agent returned no text")
        match = re.search(r"```json\s*(.*?)```", value, re.IGNORECASE | re.DOTALL)
        if match is None:
            raise ValueError("Auto Fin topic Agent returned no JSON code block")
        try:
            mapping = json.loads(match.group(1).strip())
        except json.JSONDecodeError as exc:
            raise ValueError("Auto Fin topic Agent returned invalid JSON") from exc
        if not isinstance(mapping, dict) or set(mapping) != set(topics):
            raise ValueError("Auto Fin topic Agent must return exactly the configured topic keys")
        if any(not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) for ids in mapping.values()):
            raise ValueError("Auto Fin topic Agent must return string news ID arrays")
        return mapping

    async def _select_news_ids(self, prompt: str, topics: list[str]) -> dict[str, list[str]]:
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
                ids = self._parse_news_ids(result.get("result") if isinstance(result, dict) else None, topics)
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

    def _prompt(self, news: list[dict], topics: list[str], window_hours: str) -> str:
        return self.prompt_format(
            "topic_user",
            topics=json.dumps(topics, ensure_ascii=False),
            news=json.dumps(news, ensure_ascii=False),
            window_hours=window_hours,
            output_example=json.dumps({topic: [] for topic in topics}, ensure_ascii=False),
        )

    def _batches(self, news: list[dict], topics: list[str], window_hours: str) -> Iterator[list[dict]]:
        batch: list[dict] = []
        for row in news:
            item = {**row, "title": str(row.get("title") or "")[:300], "content": str(row.get("content") or "")[:1000]}
            if len(self._prompt([*batch, item], topics, window_hours)) > self.PROMPT_CHAR_LIMIT:
                if not batch:
                    raise ValueError("Auto Fin news item exceeds the topic Agent prompt limit")
                yield batch
                batch = [item]
                if len(self._prompt(batch, topics, window_hours)) > self.PROMPT_CHAR_LIMIT:
                    raise ValueError("Auto Fin news item exceeds the topic Agent prompt limit")
            else:
                batch.append(item)
        if batch:
            yield batch

    async def execute(self):
        """Select relevant news from each batch for the current invocation."""
        assert self.context is not None
        news = list(self._required("auto_fin_news"))
        topics = list(self._required("auto_fin_topics"))
        window_hours = float(self._value("auto_fin_window_hours", 24))
        formatted_hours = f"{window_hours:g}"
        selected: dict[str, set[str]] = {topic: set() for topic in topics}
        batch_count = 0
        for batch in self._batches(news, topics, formatted_hours):
            batch_count += 1
            valid_ids = {row["news_id"] for row in batch}
            prompt = self._prompt(batch, topics, formatted_hours)
            self.logger.info(
                f"[{self.name}] filtering batch={batch_count} news={len(batch)} prompt_chars={len(prompt)} "
                f"first_id={batch[0]['news_id']} last_id={batch[-1]['news_id']}",
            )
            for topic, ids in (await self._select_news_ids(prompt, topics)).items():
                unknown = len(set(ids) - valid_ids)
                if unknown:
                    self.logger.warning(
                        f"[{self.name}] ignored unknown IDs batch={batch_count} topic={topic} count={unknown}",
                    )
                selected[topic].update(valid_ids.intersection(ids))
        by_topic = {topic: [row for row in news if row["news_id"] in selected[topic]] for topic in topics}
        relevant_ids = set().union(*selected.values()) if selected else set()
        relevant = [row for row in news if row["news_id"] in relevant_ids]
        self.context["auto_fin_news_by_topic"] = by_topic
        self.context["auto_fin_selected_news"] = relevant
        self.context.response.metadata["relevant_news_count"] = len(relevant)
        self.context.response.metadata["topic_batch_count"] = batch_count
        self.logger.info(
            f"[{self.name}] topic filtering complete batches={batch_count} selected_unique={len(relevant)} "
            f"per_topic={{{', '.join(f'{topic!r}: {len(rows)}' for topic, rows in by_topic.items())}}}",
        )
        if not relevant:
            reason = f"最近{formatted_hours}小时没有与 {', '.join(topics)} 相关的财联社新闻。"
            self.context["auto_fin_skipped"] = True
            self.context.response.answer = reason
            self.context.response.metadata.update({"skipped": True, "skip_reason": reason})
        return self.context.response
