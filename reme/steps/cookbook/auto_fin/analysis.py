"""Analyze prepared Auto Fin data and deliver its Markdown report."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date, datetime, time
from typing import Any

import yaml
from pydantic import BaseModel

from ....components import R
from ....schema import AutoFinDecisionOutput, AutoFinResearchPlan, AutoFinThemePlan
from .data import _AutoFinStep, _write, _write_jsonl


@R.register("auto_fin_analysis_step")
class AutoFinAnalysisStep(_AutoFinStep):
    """Analyze prepared cache data and write the report."""

    async def _reply(self, prompt_name: str, model: type[BaseModel], **values: str):
        if self.agent_wrapper is None:
            raise RuntimeError("Auto Fin requires an agent_wrapper with memory_search")
        result = await self.agent_wrapper.reply(
            self.prompt_format(prompt_name, **values),
            output_schema=model,
        )
        value = result.get("structured_output")
        return value if isinstance(value, model) else model.model_validate(value)

    def _prepared_context(self) -> tuple[date, datetime, date, list[date]]:
        assert self.context is not None
        required = (
            "auto_fin_trade_date",
            "auto_fin_decision_at",
            "auto_fin_start_date",
            "auto_fin_etf_dates",
            "auto_fin_lookback_days",
        )
        missing = [key for key in required if self.context.get(key) is None]
        if missing:
            raise RuntimeError(f"Auto Fin data step did not prepare: {', '.join(missing)}")
        return (
            date.fromisoformat(str(self.context["auto_fin_trade_date"])),
            datetime.fromisoformat(str(self.context["auto_fin_decision_at"])),
            date.fromisoformat(str(self.context["auto_fin_start_date"])),
            [date.fromisoformat(str(value)) for value in self.context["auto_fin_etf_dates"]],
        )

    async def _news(
        self,
        start: date,
        trade_date: date,
        window_start: datetime,
        decision_at: datetime,
    ) -> list[dict]:
        news = {}
        for day in self._days(start, trade_date):
            for row in await self._read_jsonl(self._cache_path("news", day)):
                published_at = self._published_at(row, decision_at.tzinfo)
                if published_at is None or not window_start < published_at <= decision_at:
                    continue
                title = str(row.get("title") or "").strip()
                news_id = hashlib.sha256(f"{published_at.isoformat()}|{title}".encode()).hexdigest()[:16]
                news[news_id] = {
                    "record_type": "auto_fin_news",
                    "news_id": news_id,
                    "published_at": published_at.isoformat(),
                    "title": title,
                    "source": str(row.get("src") or ""),
                    "content": str(row.get("content") or "").strip()[:4000],
                }
        return sorted(news.values(), key=lambda row: (row["published_at"], row["news_id"]))

    def _dataset_sync(self, dataset: str, days: list[date], ts_code: str | None = None) -> list[dict]:
        import polars as pl  # pylint: disable=import-outside-toplevel

        paths = [self._cache_path(dataset, day) for day in days]
        frames = [pl.read_csv(path) for path in paths]
        if not frames:
            return []
        frame = pl.concat(frames, how="diagonal_relaxed")
        if ts_code is not None:
            frame = frame.filter(pl.col("ts_code").str.to_uppercase() == ts_code.upper())
        return frame.to_dicts()

    async def _dataset(self, dataset: str, days: list[date], ts_code: str | None = None) -> list[dict]:
        return await asyncio.to_thread(self._dataset_sync, dataset, days, ts_code)

    def _data_location(self, start: date, trade_date: date, etf_dates: list[date]) -> str:
        def relative(path):
            return path.relative_to(self.workspace_path).as_posix()

        return json.dumps(
            {
                "workspace_root": str(self.workspace_path),
                "news_jsonl": [relative(self._cache_path("news", day)) for day in self._days(start, trade_date)],
                "fund_daily_csv": [relative(self._cache_path("fund_daily", day)) for day in etf_dates],
                "fund_adj_csv": [relative(self._cache_path("fund_adj", day)) for day in etf_dates],
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _adjusted_history(daily: list[dict[str, Any]], factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        factor_by_date = {
            str(row.get("trade_date")): float(row["adj_factor"])
            for row in factors
            if row.get("trade_date") and row.get("adj_factor") is not None
        }
        history = []
        for row in daily:
            trade_day = str(row.get("trade_date") or "")
            factor = factor_by_date.get(trade_day)
            close = row.get("close")
            if not trade_day or factor is None or close is None:
                continue
            history.append(
                {
                    "trade_date": trade_day,
                    "close": float(close),
                    "adj_factor": factor,
                    "adjusted_close": float(close) * factor,
                    "amount": row.get("amount"),
                },
            )
        history.sort(key=lambda row: row["trade_date"])
        previous = None
        for row in history:
            row["close_return"] = row["adjusted_close"] / previous - 1.0 if previous not in (None, 0) else None
            previous = row["adjusted_close"]
        return history

    @staticmethod
    def _case_stat(history: list[dict[str, Any]], case_date: date) -> dict[str, Any] | None:
        target = case_date.strftime("%Y%m%d")
        index = next((position for position, row in enumerate(history) if row["trade_date"] >= target), None)
        if index is None or index == 0:
            return None
        baseline = history[index - 1]["adjusted_close"]
        result: dict[str, Any] = {
            "case_trade_date": case_date.isoformat(),
            "realized_trade_date": datetime.strptime(history[index]["trade_date"], "%Y%m%d").date().isoformat(),
        }
        for offset in (0, 1, 3, 5):
            position = index + offset
            if position < len(history):
                result[f"close_return_d{offset}"] = history[position]["adjusted_close"] / baseline - 1.0
        return result

    async def _theme_data(
        self,
        theme: AutoFinThemePlan,
        etf_dates: list[date],
        previous: date,
    ) -> dict[str, Any]:
        daily, factors = await asyncio.gather(
            self._dataset("fund_daily", etf_dates, theme.etf_code),
            self._dataset("fund_adj", etf_dates, theme.etf_code),
        )
        history = self._adjusted_history(
            daily,
            factors,
        )
        case_stats = []
        for case in (item for item in theme.historical_cases if item.trade_date <= previous):
            stat = self._case_stat(history, case.trade_date)
            if stat:
                case_stats.append({**case.model_dump(mode="json"), **stat})
        latest = history[-1] if history else None
        prior = history[-2] if len(history) > 1 else None
        return {
            "theme": theme.theme,
            "direction": theme.direction,
            "etf_code": theme.etf_code,
            "etf_name": theme.etf_name,
            "latest": {
                "trade_date": latest["trade_date"] if latest else None,
                "price": latest["close"] if latest else None,
                "pre_close": prior["close"] if prior else None,
                "pct_change": latest["close_return"] if latest else None,
            },
            "historical_cases": case_stats,
            "recent_close_returns": history[-20:],
            "_history": history,
        }

    @staticmethod
    def _markdown(
        trade_date: date,
        decision_at: datetime,
        window_start: datetime,
        output: AutoFinDecisionOutput,
        datasets: list[dict[str, Any]],
    ) -> str:
        metadata = yaml.safe_dump(
            {
                "document_type": "auto_fin_news_case",
                "trade_date": trade_date.isoformat(),
                "decision_at": decision_at.isoformat(),
                "news_window": f"({window_start.isoformat()}, {decision_at.isoformat()}]",
            },
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        latest = {item["etf_code"]: item["latest"].get("pct_change") for item in datasets}
        lines = [
            "---",
            metadata,
            "---",
            "",
            f"# {trade_date.isoformat()} Auto Fin 09:30",
            "",
            output.description.strip(),
            "",
            output.body.strip(),
            "",
            "## 参考建议",
            "",
            "| 主题 | ETF | 最近收盘涨跌 | Price-in | 建议 | 置信度 | 理由 |",
            "|---|---|---:|---|---|---:|---|",
        ]
        for item in output.recommendations:
            pct = latest.get(item.etf_code)
            pct_text = f"{pct:.2%}" if pct is not None else "-"
            lines.append(
                f"| {item.theme} | {item.etf_name} ({item.etf_code}) | {pct_text} | "
                f"{item.price_in} | {item.action} | {item.confidence:.0%} | {item.reason} |",
            )
        if output.limitations:
            lines.extend(["", "## 限制", "", *[f"- {item}" for item in output.limitations]])
        lines.extend(["", "> 仅为新闻案例研究，不考虑仓位，不会执行真实交易。", ""])
        return "\n".join(lines)

    async def execute(self):
        assert self.context is not None
        if self.context.get("auto_fin_skip"):
            return self.context.response
        trade_date, decision_at, start, etf_dates = self._prepared_context()
        previous = max(etf_dates)
        window_start = datetime.combine(previous, time(15), decision_at.tzinfo)
        day_dir = self.workspace_path / str(self.config_value("daily_dir")) / trade_date.isoformat()
        report_path = day_dir / "auto_fin.md"

        news = await self._news(start, trade_date, window_start, decision_at)
        data_location = self._data_location(start, trade_date, etf_dates)
        _write_jsonl(day_dir / "auto_fin_news.jsonl", news)
        universe = await self._dataset("fund_daily", [previous])
        plan = await self._reply(
            "plan_user",
            AutoFinResearchPlan,
            trade_date=trade_date.isoformat(),
            decision_at=decision_at.isoformat(),
            window_start=window_start.isoformat(),
            lookback_days=str(self.context["auto_fin_lookback_days"]),
            data_location=data_location,
            news=json.dumps(news, ensure_ascii=False, separators=(",", ":")),
            etf_universe=json.dumps(universe, ensure_ascii=False, separators=(",", ":")),
        )
        news_ids = {item["news_id"] for item in news}
        universe_codes = {str(row.get("ts_code", "")).upper() for row in universe}
        for theme in plan.themes:
            if not set(theme.news_ids).issubset(news_ids):
                raise ValueError(f"theme {theme.theme!r} references unknown current news")
            if theme.etf_code.upper() not in universe_codes:
                raise ValueError(f"ETF {theme.etf_code!r} is absent from the prepared universe")
            if any(case.trade_date >= trade_date for case in theme.historical_cases):
                raise ValueError("memory cases must be strictly earlier than the current trade date")

        datasets = [await self._theme_data(theme, etf_dates, previous) for theme in plan.themes]
        serializable = [{key: value for key, value in item.items() if key != "_history"} for item in datasets]
        etf_rows = []
        for item in datasets:
            etf_rows.append(
                {
                    "record_type": "auto_fin_latest_etf",
                    **{key: item[key] for key in ("theme", "direction", "etf_code", "etf_name", "latest")},
                },
            )
            etf_rows.extend(
                {
                    "record_type": "auto_fin_etf_daily",
                    "theme": item["theme"],
                    "etf_code": item["etf_code"],
                    **daily_bar,
                }
                for daily_bar in item["_history"]
            )
        _write_jsonl(day_dir / "auto_fin_etf.jsonl", etf_rows)

        output = await self._reply(
            "decision_user",
            AutoFinDecisionOutput,
            trade_date=trade_date.isoformat(),
            decision_at=decision_at.isoformat(),
            data_location=data_location,
            news=json.dumps(news, ensure_ascii=False, indent=2),
            plan=plan.model_dump_json(indent=2),
            etf_data=json.dumps(serializable, ensure_ascii=False, indent=2),
        )
        allowed = {(item.theme, item.etf_code) for item in plan.themes}
        if any((item.theme, item.etf_code) not in allowed for item in output.recommendations):
            raise ValueError("recommendations must use the planned theme and representative ETF")

        plans = {(item.theme, item.etf_code): item for item in plan.themes}
        data = {(item["theme"], item["etf_code"]): item for item in serializable}
        case_rows = []
        for item in output.recommendations:
            theme = plans[(item.theme, item.etf_code)]
            case_rows.append(
                {
                    "record_type": "auto_fin_case",
                    "trade_date": trade_date.isoformat(),
                    "decision_at": decision_at.isoformat(),
                    "theme": item.theme,
                    "direction": theme.direction,
                    "news": [row for row in news if row["news_id"] in theme.news_ids],
                    "etf_code": item.etf_code,
                    "etf_name": item.etf_name,
                    "latest": data[(item.theme, item.etf_code)]["latest"],
                    "price_in": item.price_in,
                    "action": item.action,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "historical_evidence": item.historical_evidence,
                    "invalidation_condition": item.invalidation_condition,
                },
            )
        _write_jsonl(day_dir / "auto_fin_cases.jsonl", case_rows)
        _write(report_path, self._markdown(trade_date, decision_at, window_start, output, serializable))

        relative_report = report_path.relative_to(self.workspace_path).as_posix()
        self.context["markdown_path"] = relative_report
        self.context.response.answer = output.body
        self.context.response.metadata.update(
            {
                "trade_date": trade_date.isoformat(),
                "decision_at": decision_at.isoformat(),
                "news_count": len(news),
                "case_count": len(case_rows),
                "markdown_path": relative_report,
            },
        )
        return self.context.response


AutoFinPipelineStep = AutoFinAnalysisStep
