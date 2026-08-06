"""Prepare local news and ETF market data for Auto Fin."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ....components import R
from ._base import SHANGHAI_TIMEZONE, AutoFinStep, _news_id, _plain_text, _write, _write_jsonl

DEFAULT_ETF_CODES = [
    "518880.SH",
    "159516.SZ",
    "512800.SH",
    "512890.SH",
    "159992.SZ",
    "159530.SZ",
    "159869.SZ",
    "512690.SH",
    "159755.SZ",
    "159611.SZ",
    "159652.SZ",
    "512760.SH",
    "159732.SZ",
]
NEWS_FILENAME = "auto_fin_news.md"


@R.register("auto_fin_data_step")
class AutoFinDataStep(AutoFinStep):
    """Skip closed markets, then overwrite today's news and all configured ETF data."""

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

    async def _is_trade_day(self, day: date) -> bool:
        rows = await self._fetch(
            "trade_cal",
            exchange="SSE",
            start_date=day.strftime("%Y%m%d"),
            end_date=day.strftime("%Y%m%d"),
            fields="cal_date,is_open",
        )
        return any(
            str(row.get("cal_date")) == day.strftime("%Y%m%d") and int(row.get("is_open", 0)) == 1 for row in rows
        )

    def _news_path(self, day: date) -> Path:
        return self.workspace_path / str(self.config_value("daily_dir")) / day.isoformat() / NEWS_FILENAME

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
        middle = start + (end - start) / 2
        left, right = await asyncio.gather(self._fetch_news(start, middle), self._fetch_news(middle, end))
        return left + right

    async def _write_news(self, day: date, decision_at: datetime) -> str:
        start = datetime.combine(day, time.min)
        end = decision_at if day == decision_at.date() else start + timedelta(days=1)
        records: dict[str, dict[str, str]] = {}
        for row in await self._fetch_news(start, end):
            published_at = self._published_at(row)
            if published_at is None or str(row.get("src") or "") != "财联社":
                continue
            if not start <= published_at <= end or (day != decision_at.date() and published_at == end):
                continue
            news_id = _news_id(row, published_at)
            records.setdefault(
                news_id,
                {
                    "news_id": news_id,
                    "event_time": published_at.isoformat(),
                    "title": _plain_text(str(row.get("title") or "")),
                    "content": _plain_text(str(row.get("content") or "")),
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

    async def _fetch_all(self, endpoint: str, code: str, end: date) -> list[dict[str, Any]]:
        """Page backwards because TuShare fund endpoints cap one response."""
        rows_by_date: dict[str, dict[str, Any]] = {}
        end_date = end
        while True:
            page = await self._fetch(
                endpoint,
                ts_code=code,
                start_date="19900101",
                end_date=end_date.strftime("%Y%m%d"),
            )
            for row in page:
                if trade_date := str(row.get("trade_date") or ""):
                    rows_by_date[trade_date] = row
            if len(page) < 2000:
                break
            dates = [
                datetime.strptime(str(row["trade_date"]), "%Y%m%d").date() for row in page if row.get("trade_date")
            ]
            if not dates or min(dates) <= date(1990, 1, 1):
                break
            next_end = min(dates) - timedelta(days=1)
            if next_end >= end_date:
                raise RuntimeError(f"TuShare pagination did not advance for {endpoint} {code}")
            end_date = next_end
        return [rows_by_date[key] for key in sorted(rows_by_date)]

    async def _cache_etfs(self, codes: list[str], run_date: date) -> dict[str, str]:
        basics = await self._fetch(
            "etf_basic",
            list_status="L",
            fields="ts_code,csname,extname,cname,list_status",
        )
        names = {
            str(row.get("ts_code") or "")
            .strip()
            .upper(): str(
                row.get("csname") or row.get("extname") or row.get("cname") or "",
            )
            .strip()
            for row in basics
        }
        missing = [code for code in codes if not names.get(code)]
        if missing:
            raise ValueError(f"TuShare returned no ETF name for: {', '.join(missing)}")

        fin_dir = self.workspace_path / str(self.config_value("resource_dir")) / "fin"
        mapping = [{"etf_code": code, "etf_name": names[code]} for code in codes]
        _write(
            fin_dir / "etfs.json",
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        )
        for code in codes:
            daily, factors = await asyncio.gather(
                self._fetch_all("fund_daily", code, run_date),
                self._fetch_all("fund_adj", code, run_date),
            )
            factor_by_date = {str(row.get("trade_date")): row.get("adj_factor") for row in factors}
            merged = [{**row, "adj_factor": factor_by_date.get(str(row.get("trade_date")))} for row in daily]
            _write_jsonl(fin_dir / f"{code}.jsonl", merged)
        return {code: names[code] for code in codes}

    async def execute(self):
        assert self.context is not None
        run_date, decision_at = self._schedule()
        if not await self._is_trade_day(run_date):
            reason = f"{run_date.isoformat()} 不是交易日，Auto Fin 已跳过。"
            self.context["auto_fin_skipped"] = True
            self.context["auto_fin_skip_reason"] = reason
            self.context.response.answer = reason
            self.context.response.metadata.update({"date": run_date.isoformat(), "skipped": True})
            return self.context.response

        lookback = int(self._value("news_lookback_days", 60))
        if lookback < 1:
            raise ValueError("news_lookback_days must be positive")
        news_start = run_date - timedelta(days=lookback - 1)
        changes = []
        for day in self._days(news_start, run_date):
            path = self._news_path(day)
            if day != run_date and path.is_file():
                continue
            change = await self._write_news(day, decision_at)
            changes.append({"change": change, "path": str(path)})

        codes = [str(code).strip().upper() for code in self._value("etf_codes", DEFAULT_ETF_CODES)]
        names = await self._cache_etfs(codes, run_date)
        self.context.update(
            {
                "changes": changes,
                "auto_fin_date": run_date.isoformat(),
                "auto_fin_decision_at": decision_at.isoformat(),
                "auto_fin_news_start": news_start.isoformat(),
                "auto_fin_etf_names": names,
            },
        )
        self.context.response.metadata.update({"date": run_date.isoformat(), "news_downloaded": len(changes)})
        return self.context.response
