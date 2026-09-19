"""Merge the topic notes into one dated brief that links back to each of them."""

from __future__ import annotations

import json
from types import SimpleNamespace

from reme.steps.file_io import refresh_day_index

from .base import AutoFinStep, find_note, normalize_report, normalize_title, read_note
from .schema import AutoFinNote, AutoFinReportOutput


class AutoFinDigestStep(AutoFinStep):
    """Summarize the topic notes and index the day once every note is in place."""

    async def execute(self):
        assert self.context is not None
        if self.context.get("auto_fin_skipped"):
            return self.context.response
        notes: list[AutoFinNote] = self._required("auto_fin_notes")
        run_date = str(self._required("auto_fin_date"))
        earlier = find_note(self.day_dir, kind="auto-fin-digest")
        output = normalize_report(
            await self._reply(
                "digest_user",
                AutoFinReportOutput,
                job_tools=[],
                decision_at=str(self._required("auto_fin_decision_at")),
                window_start=str(self._required("auto_fin_window_start")),
                notes=json.dumps([note.model_dump() for note in notes], ensure_ascii=False),
                earlier_brief=read_note(earlier),
            ),
        )
        path, sources = await self._write_report(
            normalize_title(f"主题新闻观察（{run_date}）", "主题新闻观察"),
            output,
            kind="auto-fin-digest",
            existing=earlier,
            trailer="## 主题详解\n\n" + "\n".join(f"- [[{note.path}]]" for note in notes),
            date=run_date,
            source_notes=[note.path for note in notes],
        )
        await refresh_day_index(
            SimpleNamespace(workspace_path=self.workspace_path),
            run_date,
            str(self.config_value("daily_dir")),
        )
        self.context["markdown_path"] = self.context["auto_fin_digest_path"] = path
        self.context.response.answer = output.body
        self.context.response.metadata.update(
            {
                "markdown_path": path,
                "digest_path": path,
                "source_paths": sources,
                "note_paths": [note.path for note in notes],
                "selected_news_count": len(self._required("auto_fin_selected_news")),
            },
        )
        return self.context.response
