"""Prepare complete local TuShare caches for Auto Fin."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ....components import R
from ....components.outbound_proxy import BaseOutboundProxy
from ....enumeration import ComponentEnum
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


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace one tabular cache."""
    fields = list(dict.fromkeys(key for record in records for key in record))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object: {path}")
            records.append(value)
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class _AutoFinStep(BaseStep):
    """Shared cache and cutoff helpers."""

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

    def _cache_path(self, dataset: str, day: date) -> Path:
        filenames = {
            "news": "auto_fin_news_data.jsonl",
            "fund_daily": "auto_fin_fund_daily.csv",
            "fund_adj": "auto_fin_fund_adj.csv",
        }
        return self.workspace_path / str(self.config_value("daily_dir")) / day.isoformat() / filenames[dataset]

    def _schedule(self) -> tuple[date, datetime]:
        timezone = ZoneInfo(str(self.config_value("timezone")))
        now_value = self._value("now")
        now = datetime.fromisoformat(str(now_value)) if now_value is not None else datetime.now(timezone)
        now = now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)
        requested = str(self._value("date", "")).strip()
        trade_date = date.fromisoformat(requested) if requested else now.date()
        if trade_date != now.date():
            raise ValueError("Auto Fin only supports the current trade date to preserve the 09:30 cutoff")
        decision_at = datetime.combine(trade_date, time(9, 30), timezone)
        if now < decision_at:
            raise ValueError(f"09:30 decision has not been reached: now={now.isoformat()}")
        return trade_date, decision_at

    def _lookback_days(self) -> int:
        value = int(self._value("lookback_days", 60))
        if value < 1:
            raise ValueError("lookback_days must be at least 1")
        return value

    @staticmethod
    def _days(start: date, end: date) -> list[date]:
        return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]

    @staticmethod
    def _published_at(row: dict[str, Any], timezone) -> datetime | None:
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
            records = _read_csv(path) if path.suffix == ".csv" else _read_jsonl(path)
        except (OSError, ValueError, csv.Error, json.JSONDecodeError):
            return False
        return allow_empty or bool(records)

    async def _fetch_records(self, endpoint: str, **kwargs) -> list[dict[str, Any]]:
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

    async def _fetch_paginated(self, endpoint: str, limit: int, **kwargs) -> list[dict[str, Any]]:
        rows = []
        offset = 0
        while True:
            page = await self._fetch_records(endpoint, **kwargs, offset=offset, limit=limit)
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
            raise ValueError("Auto Fin requires at least one previous A-share trade date")
        return dates


@R.register("auto_fin_data_step")
class AutoFinDataStep(_AutoFinStep):
    """Fill missing daily TuShare cache files for the configured lookback."""

    async def _fetch_news_window(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
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
            rows = await self._fetch_records(dataset, trade_date=day.strftime("%Y%m%d"), fields=fields)
        if not rows:
            raise RuntimeError(f"TuShare returned no {dataset} data for open date {day.isoformat()}")
        _write_csv(path, rows)
        return True

    async def execute(self):
        assert self.context is not None
        trade_date, decision_at = self._schedule()
        report_path = self.workspace_path / str(self.config_value("daily_dir")) / trade_date.isoformat() / "auto_fin.md"
        relative_report = report_path.relative_to(self.workspace_path).as_posix()
        if report_path.is_file() and not bool(self._value("force", False)):
            self.context["auto_fin_skip"] = True
            self.context.response.metadata.update({"skipped": True, "markdown_path": relative_report})

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
