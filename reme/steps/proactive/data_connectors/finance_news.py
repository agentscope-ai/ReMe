"""Collect new CLS telegraphs and deliver each saved batch to the finance Agent."""

from datetime import date, datetime, timedelta
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from ._finance import FinanceCollectorStep
from ....core.registry import R
from ....schema.finance import FinanceNewsItem

_URL = "https://www.cls.cn/v1/roll/get_roll_list"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cls.cn/telegraph"}
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class _NewsText(HTMLParser):
    """Flatten source HTML without retaining script or style contents."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"br", "div", "p", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif tag in {"div", "p", "li"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _text(value):
    parser = _NewsText()
    parser.feed(str(value or ""))
    parser.close()
    return " ".join("".join(parser.parts).split())


@R.register("finance_news_step")
class FinanceNewsStep(FinanceCollectorStep):
    """Use retained news files for restart-safe deduplication of a 24-hour lookback."""

    source = "finance_news"
    description = "New CLS financial news. Read news.jsonl for complete text, publication times and source URLs."

    @staticmethod
    def _params(cursor):
        params = {
            "refresh_type": 1,
            "rn": 50,
            "last_time": cursor,
            "app": "CailianpressWeb",
            "os": "web",
            "sv": "8.7.9",
        }
        raw = "&".join(f"{key}={params[key]}" for key in sorted(params, key=str.upper))
        params["sign"] = hashlib.md5(hashlib.sha1(raw.encode()).hexdigest().encode()).hexdigest()
        return params

    def _known_ids(self, start):
        known = set()
        first_day = start.astimezone(self.collected_at.tzinfo).date()
        for manifest in self.manifests():
            if date.fromisoformat(manifest.parent.parent.name) < first_day:
                continue
            path = manifest.parent / "news.jsonl"
            self.require_path(path)
            for value in self.files.iter_jsonl(path):
                self.check_stopped()
                known.add(FinanceNewsItem.model_validate(value).news_id)
        return known

    def _fetch(self, start, end):
        cursor = int(end.timestamp())
        records = {}
        with httpx.Client(headers=_HEADERS, timeout=httpx.Timeout(self.request_timeout, connect=5)) as client:
            while True:
                payload = json.loads(self.request(client, _URL, params=self._params(cursor)))
                if not isinstance(payload, dict) or payload.get("errno") != 0:
                    raise ValueError("CLS returned an unsuccessful response")
                data = payload.get("data")
                rows = data.get("roll_data") if isinstance(data, dict) else None
                if not isinstance(rows, list) or not rows:
                    raise ValueError("CLS returned no news before covering the lookback window")
                timestamps = []
                for row in rows:
                    self.check_stopped()
                    if not isinstance(row, dict):
                        raise ValueError("CLS returned an invalid news record")
                    published = datetime.fromtimestamp(int(row["ctime"]), _SHANGHAI)
                    timestamps.append(int(published.timestamp()))
                    if not start <= published <= end:
                        continue
                    content = _text(row.get("content") or row.get("brief") or row.get("title"))
                    title = _text(row.get("title") or row.get("brief") or content)
                    item = FinanceNewsItem(
                        news_id=str(row["id"]),
                        published_at=published,
                        title=title,
                        content=content,
                        url=f"https://www.cls.cn/detail/{row['id']}",
                    )
                    records.setdefault(item.news_id, item)
                oldest = min(timestamps)
                if oldest >= cursor:
                    raise ValueError("CLS pagination did not move backward")
                if oldest <= int(start.timestamp()):
                    return sorted(records.values(), key=lambda item: (item.published_at, int(item.news_id)))
                cursor = oldest
                self.wait(self.request_interval)

    def collect(self, directory: Path) -> list[dict]:
        end = self.collected_at.astimezone(_SHANGHAI).replace(microsecond=0)
        start = end - timedelta(hours=24)
        progress = self.resource_root / "progress.json"
        self.require_path(progress)
        saved = self.files.read_json(progress, None)
        if saved is not None:
            checked_until = datetime.fromisoformat(saved["checked_until"])
            if checked_until.tzinfo is None:
                raise ValueError("News collection progress must include a timezone")
            start = max(start, min(checked_until, end) - timedelta(minutes=10))
        known = self._known_ids(start)
        records = [item for item in self._fetch(start, end) if item.news_id not in known]
        return self.write_records(directory, "news.jsonl", records)

    def _collect_and_publish(self, target, created_at, cancelled):
        result = super()._collect_and_publish(target, created_at, cancelled)
        self.check_stopped()
        self.files.write_json(self.resource_root / "progress.json", {"checked_until": self.collected_at.isoformat()})
        return result
