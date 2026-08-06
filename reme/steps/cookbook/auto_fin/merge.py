"""Generate and save the final Auto Fin recommendation."""

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


@R.register("auto_fin_merge_step")
class AutoFinMergeStep(AutoFinStep):
    """Call the final tool-free Agent with all evidence already prepared."""

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
        title = re.sub(r"^#+\s*", "", output.title.strip()) or "Auto Fin ETF 结论"
        description = output.description.strip() or "基于当前事件与相似历史表现的 ETF 观察。"
        body = output.body.strip() or "## 结论\n\n暂无可用结论。"
        if body.startswith("# "):
            body = body.partition("\n")[2].lstrip() or "## 结论\n\n暂无可用结论。"
        return AutoFinReportOutput(title=title, description=description, body=body)

    async def execute(self):
        assert self.context is not None
        if self.context.get("auto_fin_skipped"):
            self.context.response.answer = str(self.context.get("auto_fin_skip_reason") or "Auto Fin 已跳过。")
            return self.context.response
        run_date = date.fromisoformat(str(self._required("auto_fin_date")))
        output, output_path = await self._reply(
            "merge_user",
            "auto_fin_merge",
            AutoFinReportOutput,
            decision_at=str(self._required("auto_fin_decision_at")),
            etfs=json.dumps(
                [
                    {"etf_code": code, "etf_name": name}
                    for code, name in dict(self._required("auto_fin_etf_names")).items()
                ],
                ensure_ascii=False,
            ),
            analyses=json.dumps(self._required("auto_fin_analyses"), ensure_ascii=False),
            previous_report=self._previous_report(run_date),
            current_report=self._current_report(run_date),
        )
        output = self._normalize(output)
        self._write_output(output_path, output)
        markdown = (
            f"# {output.title}\n\n> {output.description}\n\n{output.body}\n\n"
            "> 仅为事件研究和持有时间参考，不构成投资建议，不会执行交易。\n"
        )
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
        self.context.response.metadata.update({"markdown_path": relative, "digest_path": relative})
        return self.context.response
