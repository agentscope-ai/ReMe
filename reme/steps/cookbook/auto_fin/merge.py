"""Research current news with ReMe and save a wikilink-backed report."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from ....components import R
from ....schema import AutoFinReportOutput
from ...file_io import refresh_day_index
from ._base import AutoFinStep, _write
from .data import AutoFinDataStep

_TOOLS = ["memory_search", "read"]
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\n]+)\]\]")


@R.register("auto_fin_merge_step")
class AutoFinMergeStep(AutoFinStep):
    """Give one Agent read-only ReMe tools, then validate links in its Markdown."""

    def _report_path(self, run_date: date) -> Path:
        return self.workspace_path / str(self.config_value("daily_dir")) / str(run_date) / "auto_fin.md"

    def _previous_report(self, run_date: date) -> str:
        """Return the most recent report from a *prior* day (yesterday's, typically)."""
        daily = self.workspace_path / str(self.config_value("daily_dir"))
        candidates = []
        for path in daily.glob("*/auto_fin.md"):
            try:
                day = date.fromisoformat(path.parent.name)
            except ValueError:
                continue
            if day < run_date:
                candidates.append((day, path))
        return max(candidates)[1].read_text(encoding="utf-8") if candidates else "无历史推荐。"

    def _current_report(self, run_date: date) -> str:
        """Return today's existing report so intra-day reruns refine it, not replace it."""
        path = self._report_path(run_date)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return "今日暂无更早时段的推荐，本次为当日首次生成。"

    @staticmethod
    def _normalize(output: AutoFinReportOutput) -> AutoFinReportOutput:
        title = re.sub(r"^#+\s*", "", output.title.strip()) or "主题新闻观察"
        description = output.description.strip() or "基于当前新闻与历史记忆的主题研究。"
        body = output.body.strip() or "## 结论\n\n暂无可用结论。"
        if body.startswith("# "):
            body = body.partition("\n")[2].lstrip() or "## 结论\n\n暂无可用结论。"
        return output.model_copy(update={"title": title, "description": description, "body": body})

    def _validate_wikilinks(self, body: str, run_date: date) -> tuple[str, list[str]]:
        """Keep real in-workspace Markdown links and downgrade invalid links to text."""
        source_paths: list[str] = []
        report = self._report_path(run_date).resolve()
        workspace = self.workspace_path.resolve()

        def replace(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            raw_target, separator, raw_alias = inner.partition("|")
            target = raw_target.strip()
            path = target.partition("#")[0].strip()
            alias = (raw_alias.strip() if separator else "") or Path(path).stem.replace("_", " ")
            if not self._valid_wikilink_path(path):
                return alias
            resolved = (workspace / path).resolve()
            try:
                resolved.relative_to(workspace)
            except ValueError:
                return alias
            if not resolved.is_file() or resolved == report:
                return alias
            if path not in source_paths:
                source_paths.append(path)
            return match.group(0)

        return _WIKILINK_RE.sub(replace, body), source_paths

    @staticmethod
    def _valid_wikilink_path(path: str) -> bool:
        parts = Path(path).parts
        return bool(
            path
            and not path.startswith("/")
            and "\\" not in path
            and path.endswith(".md")
            and "." not in parts
            and ".." not in parts
            and not any(character in path for character in "[]|"),
        )

    @staticmethod
    def _ensure_current_source(body: str, current_path: str, source_paths: list[str]) -> tuple[str, list[str]]:
        if current_path in source_paths:
            return body, source_paths
        alias = Path(current_path).stem.replace("_", " ")
        source = f"## 来源\n\n- 今日材料来自 [[{current_path}|{alias}]]。"
        return f"{body.rstrip()}\n\n{source}", [current_path, *source_paths]

    async def execute(self):
        assert self.context is not None
        run_date = date.fromisoformat(str(self._required("auto_fin_date")))
        current_path = str(self._required("auto_fin_news_path"))
        news = AutoFinDataStep.read_news(self.workspace_path / current_path)
        output, output_path = await self._reply(
            "merge_user",
            "auto_fin_merge",
            AutoFinReportOutput,
            job_tools=_TOOLS,
            decision_at=str(self._required("auto_fin_decision_at")),
            topics=json.dumps(self._required("auto_fin_topics"), ensure_ascii=False),
            news=json.dumps(news, ensure_ascii=False),
            current_news_path=current_path,
            previous_report=self._previous_report(run_date),
            current_report=self._current_report(run_date),
        )
        output = self._normalize(output)
        body, source_paths = self._validate_wikilinks(output.body, run_date)
        body, source_paths = self._ensure_current_source(body, current_path, source_paths)
        output = output.model_copy(update={"body": body})
        self._write_output(output_path, output)
        markdown = f"# {output.title}\n\n> {output.description}\n\n{output.body}\n\n"
        markdown += "> 未接入可靠行情数据；本文只提供新闻研究和回顾线索，不提供收益、目标价或买卖建议。\n"
        report = self._report_path(run_date)
        _write(report, markdown)
        await refresh_day_index(
            SimpleNamespace(workspace_path=self.workspace_path),
            str(run_date),
            str(self.config_value("daily_dir")),
        )
        relative = report.relative_to(self.workspace_path).as_posix()
        self.context["markdown_path"] = relative
        self.context["auto_fin_digest_path"] = relative
        self.context.response.answer = output.body
        self.context.response.metadata.update(
            {"markdown_path": relative, "digest_path": relative, "source_paths": source_paths},
        )
        return self.context.response
