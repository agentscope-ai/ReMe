"""Download the news required by Auto Fin."""

from __future__ import annotations

import asyncio
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

NEWS_DAYS = 360


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return _clean(value.item()) if hasattr(value, "item") else str(value)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _write(path, "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records))


class AutoFinStep(BaseStep):
    """Shared Auto Fin helpers."""

    outbound_proxy: BaseOutboundProxy | None = Ref(BaseOutboundProxy, ComponentEnum.OUTBOUND_PROXY, optional=True)

    def _value(self, key: str, default: Any = None) -> Any:
        assert self.context is not None
        return self.context.get(key, self.kwargs.get(key, default))

    @property
    def _proxy_url(self) -> str | None:
        return self.outbound_proxy.http_url if self.outbound_proxy is not None else None

    def _news_path(self, day: date) -> Path:
        daily_dir = str(self.config_value("daily_dir"))
        return self.workspace_path / daily_dir / day.isoformat() / "auto_fin_news_data.jsonl"

    @staticmethod
    def _days(start: date, end: date) -> list[date]:
        return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]

    @staticmethod
    def _read_jsonl_sync(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"JSONL records must be objects: {path}")
        return rows

    @classmethod
    async def _read_jsonl(cls, path: Path) -> list[dict[str, Any]]:
        return await asyncio.to_thread(cls._read_jsonl_sync, path)

    @staticmethod
    def _published_at(row: dict[str, Any], timezone: ZoneInfo) -> datetime | None:
        value = row.get("pub_time") or row.get("published_at") or row.get("datetime")
        if not value:
            return None
        text = str(value).strip()
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(text[:19], pattern)
                return parsed.replace(tzinfo=timezone)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone) if parsed.tzinfo is None else parsed.astimezone(timezone)

    def _schedule(self) -> tuple[date, datetime]:
        timezone = ZoneInfo(str(self.config_value("timezone")))
        now_value = self._value("now")
        now = datetime.fromisoformat(str(now_value)) if now_value is not None else datetime.now(timezone)
        now = now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)
        requested = str(self._value("date", "")).strip()
        run_date = date.fromisoformat(requested) if requested else now.date()
        if run_date != now.date():
            raise ValueError("Auto Fin only supports the current date")
        return run_date, now

    async def _fetch(self, endpoint: str, **kwargs) -> list[dict[str, Any]]:
        provider = self._value("tushare_provider")
        if provider is not None:
            value = provider(endpoint, **kwargs)
            return _records(await value if asyncio.iscoroutine(value) else value)
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is required for Auto Fin")
        api = create_tushare_api(token, proxy_url=self._proxy_url)
        return _records(await asyncio.to_thread(getattr(api, endpoint), **kwargs))

    async def _previous_trade_date(self, run_date: date) -> date:
        supplied = self._value("trade_dates")
        if supplied is not None:
            dates = [date.fromisoformat(str(value)) for value in supplied]
        else:
            start = run_date - timedelta(days=30)
            rows = await self._fetch(
                "trade_cal",
                exchange="SSE",
                start_date=start.strftime("%Y%m%d"),
                end_date=run_date.strftime("%Y%m%d"),
                fields="cal_date,is_open",
            )
            dates = [
                datetime.strptime(str(row["cal_date"]), "%Y%m%d").date()
                for row in rows
                if int(row.get("is_open", 0)) == 1
            ]
        previous = [day for day in dates if day < run_date]
        if not previous:
            raise ValueError("Auto Fin requires a previous A-share trade date")
        return max(previous)

    async def _valid_news(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            rows = await self._read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return all(str(row.get("src") or "") == "财联社" for row in rows)

    async def _fetch_news(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        rows = await self._fetch(
            "major_news",
            src="财联社",
            start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
            fields="title,pub_time,src,content",
        )
        if len(rows) < 400 or end - start <= timedelta(minutes=1):
            return rows
        midpoint = start + (end - start) / 2
        left, right = await asyncio.gather(self._fetch_news(start, midpoint), self._fetch_news(midpoint, end))
        return left + right

    async def _cache_news(self, day: date, decision_at: datetime, refresh: bool) -> bool:
        path = self._news_path(day)
        if not refresh and await self._valid_news(path):
            self.logger.debug(f"[{self.name}] news cache hit date={day.isoformat()} path={path}")
            return False
        timezone = decision_at.tzinfo
        assert isinstance(timezone, ZoneInfo)
        start = datetime.combine(day, time.min, timezone)
        end = decision_at if day == decision_at.date() else start + timedelta(days=1)
        rows = await self._fetch_news(start, end)
        rows = [
            row
            for row in rows
            if (published_at := self._published_at(row, timezone)) is not None
            and start <= published_at
            and (published_at <= end if day == decision_at.date() else published_at < end)
            and str(row.get("src") or "") == "财联社"
        ]
        _write_jsonl(path, rows)
        self.logger.debug(f"[{self.name}] news written date={day.isoformat()} records={len(rows)} path={path}")
        return True


@R.register("auto_fin_data_step")
class AutoFinDataStep(AutoFinStep):
    """Fill missing 360-day news files and always refresh today's news."""

    async def execute(self):
        assert self.context is not None
        run_date, decision_at = self._schedule()
        news_days = int(self._value("lookback_days", NEWS_DAYS))
        if news_days < 1:
            raise ValueError("lookback_days must be at least 1")
        start = run_date - timedelta(days=news_days - 1)
        previous_trade_date = await self._previous_trade_date(run_date)
        force = bool(self._value("force", False))
        self.logger.info(
            f"[{self.name}] start date={run_date.isoformat()} range={start.isoformat()}..{run_date.isoformat()} "
            f"days={news_days} force={force} decision_at={decision_at.isoformat()}",
        )
        downloaded = 0
        for day in self._days(start, run_date):
            downloaded += int(await self._cache_news(day, decision_at, force or day == run_date))
        self.context.update(
            {
                "auto_fin_date": run_date.isoformat(),
                "auto_fin_decision_at": decision_at.isoformat(),
                "auto_fin_news_start": start.isoformat(),
                "auto_fin_previous_trade_date": previous_trade_date.isoformat(),
            },
        )
        self.context.response.metadata.update({"date": run_date.isoformat(), "news_downloaded": downloaded})
        self.logger.info(
            f"[{self.name}] done downloaded={downloaded} cached={news_days - downloaded} "
            f"previous_trade_date={previous_trade_date.isoformat()}",
        )
        return self.context.response
