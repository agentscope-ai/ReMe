"""Merge all selected ETF analyses into the final report."""

from __future__ import annotations

from ....components import R
from ....schema import AutoFinEtfHistoryDetail, AutoFinReportOutput
from ._base import AutoFinStep, _write


@R.register("auto_fin_merge_step")
class AutoFinMergeStep(AutoFinStep):
    """Ask a fresh Agent for the final Markdown and persist it directly."""

    async def execute(self):
        assert self.context is not None
        etfs = list(self._required("auto_fin_etfs"))
        history_details = [
            AutoFinEtfHistoryDetail.model_validate(item) for item in self._required("auto_fin_history_details")
        ]
        selected = [item.etf.model_dump(mode="json") for item in history_details]
        if selected != etfs:
            raise ValueError("Auto Fin merge history details must match the selected ETFs")
        self.logger.info(
            f"[{self.name}] start etfs={len(etfs)}",
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
        markdown = f"# {output.title}\n\n{output.body}\n\n"
        markdown += "> 仅为事件研究和持有时间参考，不构成投资建议，不会执行交易。\n"
        day_dir = self.workspace_path / str(self.config_value("daily_dir")) / str(self._required("auto_fin_date"))
        report_path = day_dir / "auto_fin.md"
        _write(report_path, markdown)
        relative = report_path.relative_to(self.workspace_path).as_posix()
        self.context["markdown_path"] = relative
        self.context["auto_fin_digest_path"] = relative
        self.context.response.answer = output.body
        self.context.response.metadata.update(
            {"markdown_path": relative, "digest_path": relative, "etf_count": len(history_details)},
        )
        self.logger.info(
            f"[{self.name}] done path={relative} etfs={self.context.response.metadata['etf_count']}",
        )
        return self.context.response
