"""Fetch the latest 24 hours of CLS telegraph news into runtime context."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
import hashlib
from typing import Any

import httpx

from ....components import R
from ._base import AutoFinStep, _plain_text

API_URL = "https://www.cls.cn/v1/roll/get_roll_list"
HEADERS = {
    "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cls.cn/telegraph",
}
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")
DEFAULT_TOPICS = ("黄金", "机器人", "半导体")
WINDOW = timedelta(hours=24)


@R.register("auto_fin_data_step")
class AutoFinDataStep(AutoFinStep):
    """Fetch and normalize one rolling day of CLS news without writing files."""

    def _schedule(self) -> tuple[date, datetime]:
        raw_now = str(self._value("now", "")).strip()
        now = datetime.fromisoformat(raw_now) if raw_now else datetime.now(SHANGHAI_TIMEZONE)
        now = now.replace(tzinfo=SHANGHAI_TIMEZONE) if now.tzinfo is None else now.astimezone(SHANGHAI_TIMEZONE)
        raw_date = str(self._value("date", "")).strip()
        run_date = date.fromisoformat(raw_date) if raw_date else now.date()
        if run_date != now.date():
            raise ValueError("Auto Fin only supports the current date")
        return run_date, now

    @staticmethod
    def _signed_params(last_time: int) -> dict[str, str | int]:
        params: dict[str, str | int] = {
            "refresh_type": 1,
            "rn": 50,
            "last_time": last_time,
            "app": "CailianpressWeb",
            "os": "web",
            "sv": "8.7.9",
        }
        raw = "&".join(f"{key}={params[key]}" for key in sorted(params, key=str.upper))
        params["sign"] = hashlib.md5(hashlib.sha1(raw.encode()).hexdigest().encode()).hexdigest()
        return params

    async def _request_page(self, client: httpx.AsyncClient, last_time: int) -> list[dict[str, Any]]:
        response = await client.get(API_URL, params=self._signed_params(last_time))
        response.raise_for_status()
        payload = response.json()
        if payload.get("errno") != 0:
            raise RuntimeError(f"CLS API error {payload.get('errno')}: {payload.get('msg', '')}")
        rows = (payload.get("data") or {}).get("roll_data") or []
        if not isinstance(rows, list):
            raise RuntimeError("CLS API returned invalid roll_data")
        return rows

    def _window(self) -> timedelta:
        hours = float(self._value("window_hours", WINDOW.total_seconds() / 3600))
        if hours <= 0:
            raise ValueError("Auto Fin window_hours must be greater than zero")
        return timedelta(hours=hours)

    async def _fetch_recent(self, decision_at: datetime, window: timedelta) -> list[dict[str, str]]:
        cutoff = decision_at - window
        cursor = int(decision_at.timestamp())
        interval = max(0.0, float(self._value("request_interval", 10)))
        max_retries = max(1, int(self._value("max_retries", 3)))
        records: dict[str, dict[str, str]] = {}
        async with httpx.AsyncClient(headers=HEADERS, timeout=httpx.Timeout(20, connect=5)) as client:
            while True:
                for attempt in range(max_retries):
                    try:
                        rows = await self._request_page(client, cursor)
                        break
                    except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                        if attempt + 1 == max_retries:
                            raise RuntimeError(f"CLS request failed after {max_retries} attempts: {exc}") from exc
                        await asyncio.sleep(2**attempt)
                    finally:
                        if interval:
                            await asyncio.sleep(interval)
                if not rows:
                    raise RuntimeError("CLS API returned no news before the 24-hour window was covered")
                timestamps = [int(row["ctime"]) for row in rows if str(row.get("ctime", "")).isdigit()]
                if not timestamps:
                    raise RuntimeError("CLS API page contained no valid timestamps")
                oldest = min(timestamps)
                if oldest >= cursor:
                    raise RuntimeError("CLS pagination did not move backward")
                for row in rows:
                    normalized = self._normalize(row, cutoff, decision_at)
                    if normalized is not None:
                        records.setdefault(normalized["news_id"], normalized)
                if oldest <= int(cutoff.timestamp()):
                    break
                cursor = oldest
        return sorted(records.values(), key=lambda row: (row["event_time"], row["news_id"]))

    @staticmethod
    def _normalize(row: dict[str, Any], start: datetime, end: datetime) -> dict[str, str] | None:
        try:
            news_id = str(int(row["id"]))
            published_at = datetime.fromtimestamp(int(row["ctime"]), SHANGHAI_TIMEZONE)
        except (KeyError, TypeError, ValueError, OSError):
            return None
        if not start <= published_at <= end:
            return None
        content = _plain_text(str(row.get("content") or row.get("brief") or ""))
        title = _plain_text(str(row.get("title") or row.get("brief") or content))
        if not title and not content:
            return None
        return {
            "news_id": news_id,
            "event_time": published_at.isoformat(),
            "title": title,
            "content": content,
        }

    @staticmethod
    def _topics(value: Any) -> list[str]:
        values = value if isinstance(value, list) else str(value or "").replace("，", ",").split(",")
        topics = list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
        return topics or list(DEFAULT_TOPICS)

    async def execute(self):
        assert self.context is not None
        run_date, decision_at = self._schedule()
        window = self._window()
        news = await self._fetch_recent(decision_at, window)
        topics = self._topics(self._value("topics"))
        self.context.update(
            {
                "auto_fin_date": run_date.isoformat(),
                "auto_fin_decision_at": decision_at.isoformat(),
                "auto_fin_window_start": (decision_at - window).isoformat(),
                "auto_fin_window_hours": window.total_seconds() / 3600,
                "auto_fin_topics": topics,
                "auto_fin_news": news,
            },
        )
        self.context.response.metadata.update(
            {
                "date": run_date.isoformat(),
                "fetched_news_count": len(news),
                "window_hours": window.total_seconds() / 3600,
                "topics": topics,
            },
        )
        return self.context.response
