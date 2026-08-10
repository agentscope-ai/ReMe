"""Prepare free, local CLS news data for Auto Fin."""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ....components import R
from ._base import SHANGHAI_TIMEZONE, AutoFinStep, _news_id, _plain_text, _write

NEWS_FILENAME = "auto_fin_news.md"
DEFAULT_NEWS_FILE = Path("datasets/cls_news_last_7_days.jsonl")


@R.register("auto_fin_data_step")
class AutoFinDataStep(AutoFinStep):
    """Convert a locally downloaded CLS JSONL feed into indexed daily notes."""

    def _schedule(self) -> tuple[date, datetime]:
        value = str(self._value("now", "")).strip()
        now = datetime.fromisoformat(value) if value else datetime.now(SHANGHAI_TIMEZONE)
        if now.tzinfo is not None:
            now = now.astimezone(SHANGHAI_TIMEZONE).replace(tzinfo=None)
        requested = str(self._value("date", "")).strip()
        run_date = date.fromisoformat(requested) if requested else now.date()
        if run_date != now.date():
            raise ValueError("Auto Fin only supports the current date")
        return run_date, now

    def _news_path(self, day: date) -> Path:
        return self.workspace_path / str(self.config_value("daily_dir")) / day.isoformat() / NEWS_FILENAME

    def _source_path(self) -> Path:
        assert self.context is not None
        raw = str(
            self.context.get("news_file") or self.kwargs.get("news_file") or os.getenv("AUTO_FIN_NEWS_FILE", ""),
        ).strip()
        path = Path(raw) if raw else DEFAULT_NEWS_FILE
        return path if path.is_absolute() else Path.cwd() / path

    def _load_source(self) -> list[dict[str, Any]]:
        path = self._source_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Auto Fin CLS news file not found: {path}. "
                "Provide a local CLS JSONL file or set AUTO_FIN_NEWS_FILE.",
            )
        return self._read_jsonl_sync(path)

    @staticmethod
    def _source_published_at(row: dict[str, Any]) -> datetime | None:
        try:
            return datetime.fromtimestamp(int(row["ctime"]), SHANGHAI_TIMEZONE).replace(tzinfo=None)
        except (KeyError, TypeError, ValueError, OSError):
            return AutoFinStep._published_at(row)

    def _write_news(self, day: date, decision_at: datetime, source_rows: list[dict[str, Any]]) -> str:
        start = datetime.combine(day, time.min)
        end = decision_at if day == decision_at.date() else start + timedelta(days=1)
        records: dict[str, dict[str, str]] = {}
        for row in source_rows:
            published_at = self._source_published_at(row)
            if published_at is None:
                continue
            if not start <= published_at <= end or (day != decision_at.date() and published_at == end):
                continue
            content = str(row.get("content") or row.get("brief") or "")
            normalized = {"src": "财联社", "content": content}
            news_id = str(row.get("id") or "").strip() or _news_id(normalized, published_at)
            records.setdefault(
                news_id,
                {
                    "news_id": news_id,
                    "event_time": published_at.isoformat(),
                    "title": _plain_text(str(row.get("title") or row.get("brief") or content)),
                    "content": _plain_text(content),
                },
            )
        ordered = sorted(records.values(), key=lambda row: (row["event_time"], row["news_id"]))
        path = self._news_path(day)
        change = "modified" if path.exists() else "added"
        _write(path, self._render_news(day, ordered))
        return change

    @staticmethod
    def _render_news(day: date, rows: list[dict[str, str]]) -> str:
        blocks = [f"# 财联社新闻 {day.isoformat()}\n"]
        for row in rows:
            blocks.append(
                "\n".join(
                    [
                        f"## {row['title'] or '无标题'}",
                        "",
                        f"- news_id: `{row['news_id']}`",
                        f"- 时间: {row['event_time']}",
                        "- 来源: 财联社",
                        "",
                        row["content"] or row["title"],
                        "",
                    ],
                ),
            )
        return "\n".join(blocks).rstrip() + "\n"

    @staticmethod
    def read_news(path: Path) -> list[dict[str, str]]:
        """Parse an Auto Fin news Markdown file written by `_render_news`."""
        text = path.read_text(encoding="utf-8")
        rows = []
        for block in text.split("\n## ")[1:]:
            lines = block.splitlines()
            if len(lines) < 5:
                continue
            news_line = next((line for line in lines if line.startswith("- news_id: `")), "")
            time_line = next((line for line in lines if line.startswith("- 时间: ")), "")
            news_id = news_line.removeprefix("- news_id: `").removesuffix("`").strip()
            event_time = time_line.removeprefix("- 时间: ").strip()
            content_start = next(
                (index + 1 for index, line in enumerate(lines) if line == "- 来源: 财联社"),
                len(lines),
            )
            content = "\n".join(lines[content_start:]).strip()
            if news_id and event_time:
                rows.append(
                    {
                        "news_id": news_id,
                        "event_time": event_time,
                        "title": lines[0].strip(),
                        "content": content,
                    },
                )
        return rows

    async def execute(self):
        assert self.context is not None
        run_date, decision_at = self._schedule()
        lookback = int(self._value("news_lookback_days", 7))
        if lookback < 1:
            raise ValueError("news_lookback_days must be positive")
        news_start = run_date - timedelta(days=lookback - 1)
        source_rows = self._load_source()
        changes = []
        for day in self._days(news_start, run_date):
            path = self._news_path(day)
            if day != run_date and path.is_file():
                continue
            change = self._write_news(day, decision_at, source_rows)
            changes.append({"change": change, "path": str(path)})
        topics = self._topics(self._value("topics", ""))
        self.context.update(
            {
                "changes": changes,
                "auto_fin_date": run_date.isoformat(),
                "auto_fin_decision_at": decision_at.isoformat(),
                "auto_fin_news_start": news_start.isoformat(),
                "auto_fin_topics": topics,
                "auto_fin_news_path": self._news_path(run_date).relative_to(self.workspace_path).as_posix(),
            },
        )
        self.context.response.metadata.update(
            {"date": run_date.isoformat(), "news_downloaded": len(changes), "topics": topics},
        )
        return self.context.response

    @staticmethod
    def _topics(value: Any) -> list[str]:
        values = value if isinstance(value, list) else str(value or "").replace("，", ",").split(",")
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
