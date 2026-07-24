"""Merge all selected ETF analyses into the final report."""

from __future__ import annotations

import yaml

from ....components import R
from ....schema import AutoFinEtfHistoryDetail, AutoFinReportOutput
from ._base import AutoFinStep, _write, _write_jsonl


@R.register("auto_fin_merge_step")
class AutoFinMergeStep(AutoFinStep):
    """Ask a fresh Agent for the final Markdown and persist structured frontmatter."""

    async def execute(self):
        assert self.context is not None
        etfs = list(self._required("auto_fin_etfs"))
        history_details = [
            AutoFinEtfHistoryDetail.model_validate(item) for item in self._required("auto_fin_history_details")
        ]
        selected = [item.etf.model_dump(mode="json") for item in history_details]
        if selected != etfs:
            raise ValueError("Auto Fin merge history details must match the selected ETFs")
        details = [item.model_dump(mode="json") for item in history_details]
        analyses = [item.market_analysis.model_dump(mode="json") for item in history_details]
        self.logger.info(
            f"[{self.name}] start etfs={len(etfs)} analyses={len(analyses)}",
        )
        output, _ = await self._reply(
            "merge_user",
            "auto_fin_merge",
            AutoFinReportOutput,
            decision_at=str(self._required("auto_fin_decision_at")),
            window_start=str(self._required("auto_fin_window_start")),
            etfs_path=str(self._required("auto_fin_etfs_resource")),
            history_path=str(self._required("auto_fin_history_resource")),
        )
        metadata = {
            "schema_version": "auto-fin/v8",
            "decision_at": self._required("auto_fin_decision_at"),
            "news_window": f"({self._required('auto_fin_window_start')}, {self._required('auto_fin_decision_at')}]",
            "history_details": details,
            "recommendation": output.final_recommendation,
            "limitations": output.limitations,
        }
        frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
        markdown = (
            f"---\n{frontmatter}\n---\n\n# {output.title.strip()}\n\n"
            f"{output.description.strip()}\n\n{output.body.strip()}"
        )
        markdown += "\n\n> 仅为事件研究和持有时间参考，不构成投资建议，不会执行交易。\n"
        day_dir = self.workspace_path / str(self.config_value("daily_dir")) / str(self._required("auto_fin_date"))
        report_path = day_dir / "auto_fin.md"
        digest_path = day_dir / "auto_fin_brief.md"
        digest = (
            f"# {output.title.strip()}\n\n{output.concise_summary}\n\n"
            "> 仅为事件研究和持有时间参考，不构成投资建议，不会执行交易。\n"
        )
        _write_jsonl(day_dir / "auto_fin_analysis.jsonl", analyses)
        _write(report_path, markdown)
        _write(digest_path, digest)
        relative = report_path.relative_to(self.workspace_path).as_posix()
        digest_relative = digest_path.relative_to(self.workspace_path).as_posix()
        self.context["markdown_path"] = relative
        self.context["auto_fin_digest_path"] = digest_relative
        self.context.response.answer = output.final_recommendation
        self.context.response.metadata.update(
            {"markdown_path": relative, "digest_path": digest_relative, "etf_count": len(analyses)},
        )
        self.logger.info(
            f"[{self.name}] done path={relative} etfs={self.context.response.metadata['etf_count']} "
            f"limitations={len(output.limitations)}",
        )
        return self.context.response
