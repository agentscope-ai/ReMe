"""Merge all selected ETF analyses into the final report."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from ....components import R
from ....schema import AutoFinEtfHistoryDetail, AutoFinReportOutput
from ...file_io import refresh_day_index
from ._base import AutoFinStep, _write, _write_jsonl


@R.register("auto_fin_merge_step")
class AutoFinMergeStep(AutoFinStep):
    """Ask a fresh Agent for the final Markdown and persist it directly."""

    @staticmethod
    def _normalize_report(output: AutoFinReportOutput) -> AutoFinReportOutput:
        """Normalize cosmetic report fields and provide safe empty fallbacks."""
        title = re.sub(r"^#+\s*", "", output.title.strip()) or "Auto Fin ETF 结论"
        body = output.body.strip() or "## 结论\n\n暂无可用结论。"
        first_line, separator, remainder = body.partition("\n")
        if first_line.lstrip().startswith("# "):
            body = remainder.lstrip() if separator else "## 结论\n\n暂无可用结论。"
        return AutoFinReportOutput(title=title, body=body)

    @staticmethod
    def _calculation_results(history_details: list[AutoFinEtfHistoryDetail]) -> list[dict]:
        """Return the program-calculated forecast for every analyzed ETF."""
        results = []
        for item in history_details:
            holding_days = item.market_analysis.forecast.suggested_holding_days
            results.append(
                {
                    "etf_code": item.etf.etf_code,
                    "etf_name": item.etf.etf_name,
                    "suggested_holding_days": holding_days,
                    "returns": [point.model_dump(mode="json") for point in item.market_analysis.forecast.returns],
                },
            )
        return results

    async def execute(self):
        assert self.context is not None
        etfs = list(self._required("auto_fin_etfs"))
        history_details = [
            AutoFinEtfHistoryDetail.model_validate(item) for item in self._required("auto_fin_history_details")
        ]
        selected = [item.etf.model_dump(mode="json") for item in history_details]
        if selected != etfs:
            raise ValueError("Auto Fin merge history details must match the selected ETFs")
        analyses = [item.market_analysis.model_dump(mode="json") for item in history_details]
        calculation_results = self._calculation_results(history_details)
        self.logger.info(
            f"[{self.name}] start etfs={len(etfs)}",
        )
        output, output_path = await self._reply(
            "merge_user",
            "auto_fin_merge",
            AutoFinReportOutput,
            decision_at=str(self._required("auto_fin_decision_at")),
            window_start=str(self._required("auto_fin_window_start")),
            etfs_path=str(self._required("auto_fin_etfs_resource")),
            history_path=str(self._required("auto_fin_history_resource")),
            calculation_results=json.dumps(calculation_results, ensure_ascii=False),
        )
        normalized_output = self._normalize_report(output)
        if normalized_output != output:
            _write(
                output_path,
                json.dumps(normalized_output.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n",
            )
        output = normalized_output
        markdown = f"# {output.title}\n\n{output.body}\n\n"
        markdown += "> 仅为事件研究和持有时间参考，不构成投资建议，不会执行交易。\n"
        day_dir = self.workspace_path / str(self.config_value("daily_dir")) / str(self._required("auto_fin_date"))
        report_path = day_dir / "auto_fin.md"
        _write_jsonl(day_dir / "auto_fin_analysis.jsonl", analyses)
        _write(report_path, markdown)
        relative = report_path.relative_to(self.workspace_path).as_posix()
        await refresh_day_index(
            SimpleNamespace(workspace_path=self.workspace_path),
            str(self._required("auto_fin_date")),
            str(self.config_value("daily_dir")),
        )
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
