"""Download the news required by Auto Fin."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ....components import R
from ._base import SHANGHAI_TIMEZONE, AutoFinStep, _news_id, _write_jsonl


@R.register("auto_fin_data_step")
class AutoFinDataStep(AutoFinStep):
    """Fill missing daily news files and always refresh today's news."""

    def _schedule(self) -> tuple[date, datetime]:
        now_value = self._value("now")
        now = datetime.fromisoformat(str(now_value)) if now_value is not None else datetime.now(SHANGHAI_TIMEZONE)
        if now.tzinfo is not None and now.utcoffset() is not None:
            now = now.astimezone(SHANGHAI_TIMEZONE).replace(tzinfo=None)
        requested = str(self._value("date", "")).strip()
        run_date = date.fromisoformat(requested) if requested else now.date()
        if run_date != now.date():
            raise ValueError("Auto Fin only supports the current date")
        return run_date, now

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
        except (OSError, ValueError) as exc:
            self.logger.warning(
                f"[{self.name}] invalid news cache path={path} error={type(exc).__name__}: {exc}",
            )
            return False
        valid = all(str(row.get("src") or "") == "财联社" for row in rows)
        if not valid:
            self.logger.warning(f"[{self.name}] invalid news cache source path={path}")
        return valid

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
        self.logger.debug(
            f"[{self.name}] news fetch split start={start.isoformat()} end={end.isoformat()} "
            f"midpoint={midpoint.isoformat()} records={len(rows)}",
        )
        left, right = await asyncio.gather(self._fetch_news(start, midpoint), self._fetch_news(midpoint, end))
        self.logger.debug(
            f"[{self.name}] news fetch split done start={start.isoformat()} end={end.isoformat()} "
            f"records={len(left) + len(right)}",
        )
        return left + right

    async def _cache_news(self, day: date, decision_at: datetime, refresh: bool) -> bool:
        path = self._news_path(day)
        if not refresh and await self._valid_news(path):
            self.logger.debug(f"[{self.name}] news cache hit date={day.isoformat()} path={path}")
            return False
        start = datetime.combine(day, time.min)
        end = decision_at if day == decision_at.date() else start + timedelta(days=1)
        candidates = []
        for row in await self._fetch_news(start, end):
            published_at = self._published_at(row)
            in_range = (
                published_at is not None
                and start <= published_at
                and (published_at <= end if day == decision_at.date() else published_at < end)
            )
            if in_range and str(row.get("src") or "") == "财联社":
                candidates.append((published_at, _news_id(row, published_at), row))
        news = {}
        for _published_at, news_id, row in sorted(candidates, key=lambda item: item[:2]):
            news.setdefault(news_id, {**row, "news_id": news_id})
        _write_jsonl(path, list(news.values()))
        self.logger.debug(f"[{self.name}] news written date={day.isoformat()} records={len(news)} path={path}")
        return True

    async def execute(self):
        assert self.context is not None
        run_date, decision_at = self._schedule()
        news_days = int(self._value("lookback_days"))
        progress_interval = int(self._value("progress_interval"))
        if news_days < 1:
            raise ValueError("lookback_days must be at least 1")
        if progress_interval < 1:
            raise ValueError("progress_interval must be at least 1")
        start = run_date - timedelta(days=news_days - 1)
        force = bool(self._value("force", False))
        self.logger.info(
            f"[{self.name}] start date={run_date.isoformat()} range={start.isoformat()}..{run_date.isoformat()} "
            f"days={news_days} force={force} decision_at={decision_at.isoformat()}",
        )
        previous_trade_date = await self._previous_trade_date(run_date)
        self.logger.info(
            f"[{self.name}] trade date resolved date={run_date.isoformat()} "
            f"previous_trade_date={previous_trade_date.isoformat()}",
        )
        downloaded = 0
        for processed, day in enumerate(self._days(start, run_date), start=1):
            downloaded += int(await self._cache_news(day, decision_at, force or day == run_date))
            if processed % progress_interval == 0 and processed < news_days:
                self.logger.info(
                    f"[{self.name}] progress processed={processed}/{news_days} downloaded={downloaded} "
                    f"cached={processed - downloaded} last_date={day.isoformat()}",
                )
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
