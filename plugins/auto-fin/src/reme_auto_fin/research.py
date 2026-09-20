"""Research each topic's current news and persist one note per topic."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from .base import AutoFinStep, find_note, normalize_report, normalize_title, read_note
from .schema import AutoFinNote, AutoFinReportOutput

NEWS_LIMIT = 20
SEARCH_LIMIT = 3


class AutoFinResearchStep(AutoFinStep):
    """Research every topic independently so each one earns its own note."""

    @staticmethod
    def _search_budget(run_date: str) -> dict[str, Any]:
        """Return the read-only historical search budget granted to one topic."""
        return {
            "limit": 5,
            "min_score": 0.0,
            "start_date": None,
            "end_date": (date.fromisoformat(run_date) - timedelta(days=1)).isoformat(),
            "max_search_calls": SEARCH_LIMIT,
        }

    async def _research(self, topic: str, related: list[dict], run_date: str) -> AutoFinNote:
        """Research one topic and write its validated note."""
        recent = sorted(related, key=lambda row: (row["event_time"], row["news_id"]), reverse=True)[:NEWS_LIMIT]
        earlier = find_note(self.day_dir, kind="auto-fin-topic", topic=topic)
        context_id = f"auto_fin:{uuid4().hex}"
        self.logger.info(
            f"[{self.name}] research topic={topic} related={len(related)} "
            f"sent={len(recent)} omitted={len(related) - len(recent)}",
        )
        try:
            output = normalize_report(
                await self._reply(
                    "research_user",
                    AutoFinReportOutput,
                    job_tools=["search"],
                    injected_job_kwargs=self._search_budget(run_date),
                    tool_context_id=context_id,
                    decision_at=str(self._required("auto_fin_decision_at")),
                    window_start=str(self._required("auto_fin_window_start")),
                    topic=topic,
                    news=json.dumps(recent, ensure_ascii=False),
                    omitted_news_count=str(len(related) - len(recent)),
                    earlier_note=read_note(earlier),
                ),
            )
        finally:
            if self.app_context is not None:
                self.app_context.metadata.get("__search_call_budgets", {}).pop(context_id, None)
        written = await self._write_report(
            normalize_title(topic, "主题观察"),
            output,
            kind="auto-fin-topic",
            existing=earlier,
            date=run_date,
            topic=topic,
            source_news_ids=[row["news_id"] for row in recent],
        )
        return AutoFinNote(
            topic=topic,
            title=output.title,
            description=output.description,
            body=written.body,
            path=written.path,
        )

    async def execute(self):
        """Write one note per topic that had relevant news, isolating per-topic failures."""
        assert self.context is not None
        self.context["changes"] = []
        if self.context.get("auto_fin_skipped"):
            return self.context.response
        run_date = str(self._required("auto_fin_date"))
        by_topic = self._required("auto_fin_news_by_topic")
        notes: list[AutoFinNote] = []
        failures: list[dict[str, str]] = []
        for topic in self._required("auto_fin_topics"):
            if not by_topic[topic]:
                self.logger.info(f"[{self.name}] skipping topic={topic} reason=no_related_news")
                continue
            try:
                notes.append(await self._research(topic, by_topic[topic], run_date))
            except Exception as exc:
                failures.append({"topic": topic, "error": str(exc)})
                self.logger.warning(f"[{self.name}] topic={topic} failed; continuing with the rest: {exc}")
        if failures and not notes:
            raise RuntimeError(f"Auto Fin research failed for every topic: {failures}")
        self.context["auto_fin_notes"] = notes
        self.context.response.answer = f"Researched {len(notes)} topic(s): {', '.join(note.path for note in notes)}"
        self.context.response.metadata.update(
            {
                "note_paths": [note.path for note in notes],
                "failed_topics": failures,
                "selected_news_count": len(self._required("auto_fin_selected_news")),
            },
        )
        return self.context.response
