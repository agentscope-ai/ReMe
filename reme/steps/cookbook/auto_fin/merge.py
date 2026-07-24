"""Merge all Auto Fin topic analyses into the final report."""

from __future__ import annotations

import json

import yaml

from ....components import R
from ....schema import AutoFinReportOutput
from .analysis import AutoFinAgentStep
from .data import _write, _write_jsonl


@R.register("auto_fin_merge_step")
class AutoFinMergeStep(AutoFinAgentStep):
    """Ask a fresh Agent for the final Markdown and persist structured frontmatter."""

    async def execute(self):
        assert self.context is not None
        topics = dict(self._required("auto_fin_topics"))
        analyses = list(self._required("auto_fin_topic_analyses"))
        output = await self._reply(
            "merge_user",
            AutoFinReportOutput,
            decision_at=str(self._required("auto_fin_decision_at")),
            window_start=str(self._required("auto_fin_window_start")),
            topics=json.dumps(topics, ensure_ascii=False, indent=2),
            analyses=json.dumps(analyses, ensure_ascii=False, indent=2),
        )
        metadata = {
            "schema_version": "auto-fin/v2",
            "decision_at": self._required("auto_fin_decision_at"),
            "news_window": f"({self._required('auto_fin_window_start')}, {self._required('auto_fin_decision_at')}]",
            "topics": topics,
            "analyses": analyses,
        }
        frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
        limitations = "\n".join(f"- {item}" for item in output.limitations)
        markdown = (
            f"---\n{frontmatter}\n---\n\n# {output.title.strip()}\n\n"
            f"{output.description.strip()}\n\n{output.body.strip()}"
        )
        if limitations:
            markdown += f"\n\n## 限制\n\n{limitations}"
        markdown += "\n\n> 仅为事件研究和持有时间参考，不构成投资建议，不会执行交易。\n"
        day_dir = self.workspace_path / str(self.config_value("daily_dir")) / str(self._required("auto_fin_date"))
        report_path = day_dir / "auto_fin.md"
        _write_jsonl(day_dir / "auto_fin_analysis.jsonl", analyses)
        _write(report_path, markdown)
        relative = report_path.relative_to(self.workspace_path).as_posix()
        self.context["markdown_path"] = relative
        self.context.response.answer = output.body
        self.context.response.metadata.update(
            {"markdown_path": relative, "etf_count": sum(len(x["etfs"]) for x in analyses)},
        )
        return self.context.response
