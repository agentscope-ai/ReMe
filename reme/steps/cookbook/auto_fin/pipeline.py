"""Two-step cached news and ETF research for Auto Fin."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel

from ....components import R
from ....components.outbound_proxy import BaseOutboundProxy
from ....enumeration import ComponentEnum
from ....schema import AutoFinDecisionOutput, AutoFinResearchPlan, AutoFinThemePlan
from ....utils.tushare import create_tushare_api
from ...base_step import BaseStep, Ref


def _clean(value: Any) -> Any:
    """Convert dataframe values into strict JSON values."""
    if value is None or isinstance(value, (str, int, bool)):
        cleaned = value
    elif isinstance(value, float):
        cleaned = value if math.isfinite(value) else None
    elif isinstance(value, (date, datetime)):
        cleaned = value.isoformat()
    elif isinstance(value, dict):
        cleaned = {str(key): _clean(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        cleaned = [_clean(item) for item in value]
    elif hasattr(value, "item"):
        cleaned = _clean(value.item())
    else:
        cleaned = str(value)
    return cleaned


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict(orient="records")
        except TypeError:
            value = value.to_dicts()
    return [_clean(item) for item in value]


def _write(path: Path, text: str) -> None:
    """Atomically replace one generated artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    _write(path, text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object: {path}")
            records.append(value)
    return records


class _AutoFinStep(BaseStep):
    """Shared schedule, cache, and TuShare helpers."""

    outbound_proxy: BaseOutboundProxy | None = Ref(
        BaseOutboundProxy,
        ComponentEnum.OUTBOUND_PROXY,
        optional=True,
    )

    def _value(self, key: str, default: Any = None) -> Any:
        assert self.context is not None
        return self.context.get(key, self.kwargs.get(key, default))

    @property
    def _proxy_url(self) -> str | None:
        return self.outbound_proxy.http_url if self.outbound_proxy is not None else None

    @property
    def _cache_root(self) -> Path:
        return self.workspace_path / "metadata" / "auto-fin" / "cache"

    def _cache_path(self, dataset: str, day: date) -> Path:
        return self._cache_root / dataset / f"{day.isoformat()}.jsonl"

    def _schedule(self) -> tuple[date, datetime, ZoneInfo]:
        timezone = ZoneInfo(str(self.config_value("timezone")))
        now_value = self._value("now")
        now = datetime.fromisoformat(str(now_value)) if now_value is not None else datetime.now(timezone)
        now = now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)
        requested = str(self._value("date", "")).strip()
        trade_date = date.fromisoformat(requested) if requested else now.date()
        if trade_date != now.date():
            raise ValueError(
                "Auto Fin only supports the current trade date to preserve the 09:30 cutoff",
            )
        decision_at = datetime.combine(trade_date, time(9, 30), timezone)
        if now < decision_at:
            raise ValueError(
                f"09:30 decision has not been reached: now={now.isoformat()}",
            )
        return trade_date, decision_at, timezone

    def _lookback_days(self) -> int:
        value = int(self._value("lookback_days", 60))
        if value < 1:
            raise ValueError("lookback_days must be at least 1")
        return value

    @staticmethod
    def _days(start: date, end: date) -> list[date]:
        return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]

    @staticmethod
    def _published_at(row: dict[str, Any], timezone: ZoneInfo) -> datetime | None:
        value = row.get("pub_time") or row.get("published_at") or row.get("datetime")
        if not value:
            return None
        text = str(value).strip()
        parsed = None
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(text[:19], pattern)
                break
            except ValueError:
                continue
        if parsed is None:
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
        return parsed.replace(tzinfo=timezone) if parsed.tzinfo is None else parsed.astimezone(timezone)

    @staticmethod
    def _is_valid_cache(path: Path, *, allow_empty: bool = True) -> bool:
        if not path.is_file():
            return False
        try:
            records = _read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return allow_empty or bool(records)

    async def _fetch_records(self, endpoint: str, **kwargs) -> list[dict[str, Any]]:
        """Call TuShare or an injected test provider."""
        provider = self._value("tushare_provider")
        if provider is not None:
            value = provider(endpoint, **kwargs)
            if asyncio.iscoroutine(value):
                value = await value
            return _records(value)
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is required for Auto Fin")
        api = create_tushare_api(token, proxy_url=self._proxy_url)
        value = await asyncio.to_thread(getattr(api, endpoint), **kwargs)
        return _records(value)

    async def _fetch_paginated(
        self,
        endpoint: str,
        limit: int,
        **kwargs,
    ) -> list[dict[str, Any]]:
        rows = []
        offset = 0
        while True:
            page = await self._fetch_records(
                endpoint,
                **kwargs,
                offset=offset,
                limit=limit,
            )
            rows.extend(page)
            if len(page) < limit:
                return rows
            offset += limit

    async def _trade_dates(self, trade_date: date, start: date) -> list[date]:
        supplied = self._value("trade_dates")
        if supplied is not None:
            dates = [date.fromisoformat(str(value)) for value in supplied]
        else:
            rows = await self._fetch_records(
                "trade_cal",
                exchange="SSE",
                start_date=start.strftime("%Y%m%d"),
                end_date=trade_date.strftime("%Y%m%d"),
                fields="cal_date,is_open",
            )
            dates = [
                datetime.strptime(str(row["cal_date"]), "%Y%m%d").date()
                for row in rows
                if int(row.get("is_open", 0)) == 1
            ]
        dates = sorted(set(dates))
        if trade_date not in dates:
            raise ValueError(f"{trade_date.isoformat()} is not an A-share trade date")
        if not any(day < trade_date for day in dates):
            raise ValueError(
                "Auto Fin requires at least one previous A-share trade date",
            )
        return dates


@R.register("auto_fin_data_step")
class AutoFinDataStep(_AutoFinStep):
    """Fill missing daily TuShare cache files for the configured lookback."""

    async def _fetch_news_window(
        self,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        rows = await self._fetch_records(
            "major_news",
            src="",
            start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
            fields="title,pub_time,src,content",
        )
        if len(rows) < 400 or end - start <= timedelta(minutes=1):
            return rows
        midpoint = start + (end - start) / 2
        left, right = await asyncio.gather(
            self._fetch_news_window(start, midpoint),
            self._fetch_news_window(midpoint, end),
        )
        return left + right

    async def _cache_news(self, day: date, decision_at: datetime) -> bool:
        path = self._cache_path("news", day)
        if self._is_valid_cache(path):
            return False
        timezone = decision_at.tzinfo
        assert isinstance(timezone, ZoneInfo)
        start = datetime.combine(day, time.min, timezone)
        end = decision_at if day == decision_at.date() else start + timedelta(days=1)
        rows = await self._fetch_news_window(start, end)
        rows = [
            row
            for row in rows
            if (published_at := self._published_at(row, timezone)) is not None
            and start <= published_at
            and (published_at <= end if day == decision_at.date() else published_at < end)
        ]
        _write_jsonl(path, rows)
        return True

    async def _cache_etf(self, dataset: str, day: date) -> bool:
        path = self._cache_path(dataset, day)
        if self._is_valid_cache(path, allow_empty=False):
            return False
        fields = (
            "ts_code,trade_date,close,pre_close,pct_chg,amount"
            if dataset == "fund_daily"
            else "ts_code,trade_date,adj_factor"
        )
        if dataset == "fund_adj":
            rows = await self._fetch_paginated(
                dataset,
                2000,
                trade_date=day.strftime("%Y%m%d"),
                fields=fields,
            )
        else:
            rows = await self._fetch_records(
                dataset,
                trade_date=day.strftime("%Y%m%d"),
                fields=fields,
            )
        if not rows:
            raise RuntimeError(
                f"TuShare returned no {dataset} data for open date {day.isoformat()}",
            )
        _write_jsonl(path, rows)
        return True

    async def execute(self):
        assert self.context is not None
        trade_date, decision_at, _ = self._schedule()
        report_path = self.workspace_path / str(self.config_value("daily_dir")) / trade_date.isoformat() / "auto_fin.md"
        relative_report = report_path.relative_to(self.workspace_path).as_posix()
        skip_analysis = report_path.is_file() and not bool(self._value("force", False))
        if skip_analysis:
            self.context["auto_fin_skip"] = True
            self.context["markdown_path"] = relative_report
            self.context.response.metadata.update(
                {"skipped": True, "markdown_path": relative_report},
            )

        lookback_days = self._lookback_days()
        start = trade_date - timedelta(days=lookback_days - 1)
        trade_dates = await self._trade_dates(trade_date, start)
        etf_dates = [day for day in trade_dates if start <= day < trade_date]
        downloaded = {"news": 0, "fund_daily": 0, "fund_adj": 0}
        for day in self._days(start, trade_date):
            downloaded["news"] += int(await self._cache_news(day, decision_at))
        for day in etf_dates:
            downloaded["fund_daily"] += int(await self._cache_etf("fund_daily", day))
            downloaded["fund_adj"] += int(await self._cache_etf("fund_adj", day))

        self.context.update(
            {
                "auto_fin_trade_date": trade_date.isoformat(),
                "auto_fin_decision_at": decision_at.isoformat(),
                "auto_fin_start_date": start.isoformat(),
                "auto_fin_trade_dates": [day.isoformat() for day in trade_dates],
                "auto_fin_etf_dates": [day.isoformat() for day in etf_dates],
                "auto_fin_lookback_days": lookback_days,
            },
        )
        self.context.response.metadata.update(
            {
                "trade_date": trade_date.isoformat(),
                "lookback_days": lookback_days,
                "data_downloaded": downloaded,
            },
        )
        return self.context.response


@R.register("auto_fin_analysis_step")
class AutoFinAnalysisStep(_AutoFinStep):
    """Analyze prepared cache data, write the report, and dispatch delivery."""

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
        )
        missing = [key for key in required if self.context.get(key) is None]
        if missing:
            raise RuntimeError(
                f"Auto Fin data step did not prepare: {', '.join(missing)}",
            )
        return (
            date.fromisoformat(str(self.context["auto_fin_trade_date"])),
            datetime.fromisoformat(str(self.context["auto_fin_decision_at"])),
            date.fromisoformat(str(self.context["auto_fin_start_date"])),
            [date.fromisoformat(str(value)) for value in self.context["auto_fin_etf_dates"]],
        )

    def _news(
        self,
        start: date,
        trade_date: date,
        window_start: datetime,
        decision_at: datetime,
    ) -> list[dict]:
        news = {}
        for day in self._days(start, trade_date):
            for row in _read_jsonl(self._cache_path("news", day)):
                published_at = self._published_at(row, decision_at.tzinfo)
                if published_at is None or not window_start < published_at <= decision_at:
                    continue
                title = str(row.get("title") or "").strip()
                news_id = hashlib.sha256(
                    f"{published_at.isoformat()}|{title}".encode(),
                ).hexdigest()[:16]
                news[news_id] = {
                    "record_type": "auto_fin_news",
                    "news_id": news_id,
                    "published_at": published_at.isoformat(),
                    "title": title,
                    "source": str(row.get("src") or ""),
                    "content": str(row.get("content") or "").strip()[:4000],
                }
        return sorted(
            news.values(),
            key=lambda row: (row["published_at"], row["news_id"]),
        )

    def _dataset(
        self,
        dataset: str,
        days: list[date],
        ts_code: str | None = None,
    ) -> list[dict]:
        rows = []
        for day in days:
            for row in _read_jsonl(self._cache_path(dataset, day)):
                if ts_code is None or str(row.get("ts_code", "")).upper() == ts_code.upper():
                    rows.append(row)
        return rows

    @staticmethod
    def _adjusted_history(
        daily: list[dict[str, Any]],
        factors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
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
    def _case_stat(
        history: list[dict[str, Any]],
        case_date: date,
    ) -> dict[str, Any] | None:
        target = case_date.strftime("%Y%m%d")
        index = next(
            (position for position, row in enumerate(history) if row["trade_date"] >= target),
            None,
        )
        if index is None or index == 0:
            return None
        baseline = history[index - 1]["adjusted_close"]
        result: dict[str, Any] = {
            "case_trade_date": case_date.isoformat(),
            "realized_trade_date": datetime.strptime(
                history[index]["trade_date"],
                "%Y%m%d",
            )
            .date()
            .isoformat(),
        }
        for offset in (0, 1, 3, 5):
            position = index + offset
            if position < len(history):
                result[f"close_return_d{offset}"] = history[position]["adjusted_close"] / baseline - 1.0
        return result

    def _theme_data(
        self,
        theme: AutoFinThemePlan,
        etf_dates: list[date],
        previous: date,
    ) -> dict[str, Any]:
        history = self._adjusted_history(
            self._dataset("fund_daily", etf_dates, theme.etf_code),
            self._dataset("fund_adj", etf_dates, theme.etf_code),
        )
        valid_cases = [case for case in theme.historical_cases if case.trade_date <= previous]
        case_stats = []
        for case in valid_cases:
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
            lines.extend(
                ["", "## 限制", "", *[f"- {item}" for item in output.limitations]],
            )
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

        news = self._news(start, trade_date, window_start, decision_at)
        _write_jsonl(day_dir / "auto_fin_news.jsonl", news)
        universe = self._dataset("fund_daily", [previous])
        plan = await self._reply(
            "plan_user",
            AutoFinResearchPlan,
            trade_date=trade_date.isoformat(),
            decision_at=decision_at.isoformat(),
            window_start=window_start.isoformat(),
            lookback_days=str(self.context["auto_fin_lookback_days"]),
            news=json.dumps(news, ensure_ascii=False, separators=(",", ":")),
            etf_universe=json.dumps(
                universe,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        news_ids = {item["news_id"] for item in news}
        universe_codes = {str(row.get("ts_code", "")).upper() for row in universe}
        for theme in plan.themes:
            if not set(theme.news_ids).issubset(news_ids):
                raise ValueError(
                    f"theme {theme.theme!r} references unknown current news",
                )
            if theme.etf_code.upper() not in universe_codes:
                raise ValueError(
                    f"ETF {theme.etf_code!r} is absent from the prepared universe",
                )
            if any(case.trade_date >= trade_date for case in theme.historical_cases):
                raise ValueError(
                    "memory cases must be strictly earlier than the current trade date",
                )

        datasets = [self._theme_data(theme, etf_dates, previous) for theme in plan.themes]
        serializable = [{key: value for key, value in item.items() if key != "_history"} for item in datasets]
        etf_rows = []
        for item in datasets:
            etf_rows.append(
                {
                    "record_type": "auto_fin_latest_etf",
                    **{
                        key: item[key]
                        for key in (
                            "theme",
                            "direction",
                            "etf_code",
                            "etf_name",
                            "latest",
                        )
                    },
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
            news=json.dumps(news, ensure_ascii=False, indent=2),
            plan=plan.model_dump_json(indent=2),
            etf_data=json.dumps(serializable, ensure_ascii=False, indent=2),
        )
        allowed = {(item.theme, item.etf_code) for item in plan.themes}
        if any((item.theme, item.etf_code) not in allowed for item in output.recommendations):
            raise ValueError(
                "recommendations must use the planned theme and representative ETF",
            )

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
        _write(
            report_path,
            self._markdown(trade_date, decision_at, window_start, output, serializable),
        )
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
        if self.dispatch_step_specs:
            await self.dispatch_steps(self.dispatch_step_specs)
        return self.context.response


# Compatibility alias for callers importing the previous one-step class name.
AutoFinPipelineStep = AutoFinAnalysisStep
