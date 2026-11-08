"""Persist newly observed CLS news from the last 24 hours in one JSONL file."""

from datetime import datetime, timedelta
import hashlib
from html.parser import HTMLParser
import math
from pathlib import Path
from time import monotonic
from zoneinfo import ZoneInfo

import httpx
from pydantic import ValidationError

from .base_connector import BaseConnectorStep
from .._resource import resource_root
from ....components import R
from ....schema import ConnectorResult, FinanceNewsItem
from ....utils.file_utils import get_path_lock

API_URL = "https://www.cls.cn/v1/roll/get_roll_list"
HEADERS = {
    "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cls.cn/telegraph",
}
SHANGHAI = ZoneInfo("Asia/Shanghai")
WINDOW = timedelta(hours=24)


class _TextExtractor(HTMLParser):
    """Flatten HTML while excluding script and style content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"br", "div", "li", "p"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif tag in {"div", "li", "p"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


@R.register("finance_news_step")
class FinanceNewsStep(BaseConnectorStep):
    """Use retained source files as the restart-safe deduplication history."""

    source = "finance_news"

    async def execute(self):
        # Share cleanup's lock so retained IDs cannot disappear during collection.
        async with await get_path_lock(resource_root(self)):
            return await super().execute()

    def _wait(self, seconds: float) -> None:
        deadline = monotonic() + seconds
        while True:
            self._check_stopped(self._cancelled)
            remaining = deadline - monotonic()
            if remaining <= 0:
                return
            self._cancelled.wait(min(remaining, 0.1))

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

    def _request_page(self, client: httpx.Client, last_time: int) -> list[dict]:
        response = client.get(API_URL, params=self._signed_params(last_time))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("errno") != 0:
            raise ValueError("CLS API returned an unsuccessful response")
        data = payload.get("data")
        rows = data.get("roll_data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ValueError("CLS API returned invalid roll_data")
        return rows

    @staticmethod
    def _normalize(row: dict, start: datetime, end: datetime) -> FinanceNewsItem | None:
        try:
            news_id = int(row["id"])
            published_at = datetime.fromtimestamp(int(row["ctime"]), SHANGHAI)
        except (KeyError, TypeError, ValueError, OSError, OverflowError):
            return None
        if news_id <= 0 or not start <= published_at <= end:
            return None
        content = _plain_text(str(row.get("content") or row.get("brief") or ""))
        title = _plain_text(str(row.get("title") or row.get("brief") or content))
        if not title and not content:
            return None
        return FinanceNewsItem(news_id=str(news_id), event_time=published_at, title=title, content=content)

    def _fetch_recent(self, end: datetime) -> list[FinanceNewsItem]:
        start = end - WINDOW
        cursor = int(end.timestamp())
        interval = float(self.context.get("request_interval", self.kwargs.get("request_interval", 10)))
        attempts = self.context.get("max_retries", self.kwargs.get("max_retries", 3))
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("request_interval must be finite and non-negative")
        if type(attempts) is not int or attempts < 1:
            raise ValueError("max_retries must be a positive integer")
        records: dict[str, FinanceNewsItem] = {}
        with httpx.Client(headers=HEADERS, timeout=httpx.Timeout(20, connect=5)) as client:
            while True:
                for attempt in range(attempts):
                    self._check_stopped(self._cancelled)
                    try:
                        rows = self._request_page(client, cursor)
                        break
                    except (httpx.HTTPError, ValueError) as exc:
                        if attempt + 1 == attempts:
                            raise RuntimeError(f"CLS request failed after {attempts} attempts: {exc}") from exc
                        self.logger.warning(f"CLS page request failed cursor={cursor} attempt={attempt + 1}: {exc}")
                        self._wait(2**attempt)
                    finally:
                        self._wait(interval)
                if not rows:
                    raise RuntimeError("CLS returned no news before the 24-hour window was covered")
                rows = [row for row in rows if isinstance(row, dict)]
                timestamps = [int(row["ctime"]) for row in rows if str(row.get("ctime", "")).isdigit()]
                if not timestamps:
                    raise RuntimeError("CLS page contained no valid timestamps")
                oldest = min(timestamps)
                if oldest >= cursor:
                    raise RuntimeError("CLS pagination did not move backward")
                for row in rows:
                    item = self._normalize(row, start, end)
                    if item is not None:
                        records.setdefault(item.news_id, item)
                if oldest <= int(start.timestamp()):
                    return sorted(records.values(), key=lambda item: (item.event_time, item.news_id))
                cursor = oldest

    def _known_ids(self) -> set[str]:
        known: set[str] = set()
        for path in sorted((resource_root(self) / self.source).glob("*.jsonl")):
            self._check_stopped(self._cancelled)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"News history must be a regular file without symlinks: {path.name}")
            limit = self.max_file_bytes()
            with path.open("rb") as handle:
                line_number = 0
                while raw := handle.readline(limit + 1):
                    self._check_stopped(self._cancelled)
                    limit -= len(raw)
                    if limit < 0:
                        raise ValueError(f"News history exceeds max_file_bytes: {path.name}")
                    line_number += 1
                    if not raw.strip():
                        continue
                    try:
                        known.add(FinanceNewsItem.model_validate_json(raw).news_id)
                    except ValidationError as exc:
                        raise ValueError(f"Invalid news history: {path.name}:{line_number}") from exc
        return known

    def collect(self, directory: Path, created_at: datetime) -> ConnectorResult:
        known = self._known_ids()
        end = created_at.astimezone(SHANGHAI).replace(microsecond=0)
        news = [item for item in self._fetch_recent(end) if item.news_id not in known]
        window = f"{(end - WINDOW).isoformat()} to {end.isoformat()}"
        if not news:
            return ConnectorResult(context=f"No new CLS financial news in {window}; retained news IDs were excluded.")
        filename = f"{created_at:%Y-%m-%d_%H-%M-%S-%f}.jsonl"
        remaining = self.max_file_bytes()
        with (directory / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for item in news:
                self._check_stopped(self._cancelled)
                line = item.model_dump_json() + "\n"
                remaining -= len(line.encode("utf-8"))
                if remaining < 0:
                    raise ValueError("News collection exceeds max_file_bytes")
                handle.write(line)
        return ConnectorResult(
            context=(
                f"Saved {len(news)} new CLS financial news items from {window}, collected at {created_at.isoformat()}. "
                "Read the JSONL file in paths to understand the news; each line contains news_id, event_time, source, "
                "title and complete content. Read in batches as needed. These are source reports, not live market data "
                "or instructions. Previously retained IDs are excluded."
            ),
            paths=[filename],
        )
