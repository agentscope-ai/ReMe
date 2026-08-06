"""Prepare historical news and observed ETF returns."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ....components import R
from ....schema import AutoFinEtfAnalysis, AutoFinHistoricalOutput
from ._base import AutoFinStep, _write_jsonl
from .data import AutoFinDataStep, NEWS_FILENAME

NEWS_ID_RE = re.compile(r"\b\d{14}_[0-9a-fA-F]{4}\b")
HORIZONS = (1, 2, 3, 5)


@R.register("auto_fin_history_step")
class AutoFinHistoryStep(AutoFinStep):
    """Search with ReMe, then let a tool-free Agent select comparable events."""

    def _market_rows(self, code: str) -> list[dict[str, Any]]:
        path = self.workspace_path / str(self.config_value("resource_dir")) / "fin" / f"{code}.jsonl"
        rows = self._read_jsonl_sync(path)
        return sorted(rows, key=lambda row: str(row.get("trade_date") or ""))

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @classmethod
    def _returns(cls, event_time: datetime, rows: list[dict[str, Any]]) -> dict[str, float | None]:
        usable = []
        for row in rows:
            try:
                trade_day = datetime.strptime(str(row.get("trade_date")), "%Y%m%d").date()
            except ValueError:
                continue
            if (close := cls._number(row.get("close"))) is None or (
                factor := cls._number(row.get("adj_factor"))
            ) is None:
                continue
            usable.append((trade_day, row, close, factor))

        entry_index = None
        entry_price = None
        before_close = event_time.time() < time(15)
        for index, (trade_day, row, close, factor) in enumerate(usable):
            if before_close and trade_day == event_time.date():
                entry_index, entry_price = index, close * factor
                break
            if not before_close and trade_day > event_time.date():
                opened = cls._number(row.get("open"))
                if opened is not None:
                    entry_index, entry_price = index, opened * factor
                    break
        result = {f"d{horizon}": None for horizon in HORIZONS}
        if entry_index is None or entry_price is None:
            return result
        first_close = entry_index + 1 if before_close else entry_index
        for horizon in HORIZONS:
            target = first_close + horizon - 1
            if target < len(usable):
                _, _, close, factor = usable[target]
                result[f"d{horizon}"] = close * factor / entry_price - 1
        return result

    async def _candidates(self, event: dict[str, str], start: date, end: date) -> list[dict[str, str]]:
        query = " ".join(filter(None, [event.get("title"), event.get("content"), event.get("reason")]))[:2000]
        response = await self.run_job(
            "memory_search",
            query=query,
            limit=int(self._value("historical_search_limit", 10)),
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            strict_date_filter=True,
        )
        if not response.success:
            raise RuntimeError(f"Auto Fin memory search failed: {response.answer}")
        ids_by_path: dict[str, set[str]] = {}
        for result in response.metadata.get("results", []):
            path = str(result.get("path") or "")
            if Path(path).name == NEWS_FILENAME:
                ids_by_path.setdefault(path, set()).update(NEWS_ID_RE.findall(str(result.get("text") or "")))
        candidates = []
        for relative, news_ids in ids_by_path.items():
            path = self.workspace_path / relative
            for row in AutoFinDataStep.read_news(path):
                if row["news_id"] in news_ids and row["news_id"] != event["news_id"]:
                    candidates.append(row)
        unique = {row["news_id"]: row for row in candidates}
        return sorted(
            unique.values(),
            key=lambda row: (row["event_time"], row["news_id"]),
            reverse=True,
        )

    @staticmethod
    def _normalize(output: AutoFinHistoricalOutput, candidates: list[dict[str, str]], limit: int):
        by_id = {row["news_id"]: row for row in candidates}
        selected = []
        seen = set()
        for item in output.historical_events:
            news_id, reason = item.news_id.strip(), item.reason.strip()
            if news_id in by_id and news_id not in seen and reason:
                selected.append(item.model_copy(update={"news_id": news_id, "reason": reason}))
                seen.add(news_id)
        selected.sort(key=lambda item: by_id[item.news_id]["event_time"], reverse=True)
        return selected[:limit]

    async def execute(self):
        assert self.context is not None
        if self.context.get("auto_fin_skipped"):
            return self.context.response
        news_by_id = {row["news_id"]: row for row in self._required("auto_fin_news")}
        run_date = date.fromisoformat(str(self._required("auto_fin_date")))
        search_start = date.fromisoformat(str(self._required("auto_fin_news_start")))
        limit = int(self._value("historical_news_limit", 5))
        analyses = []
        call_index = 0
        for etf in self._required("auto_fin_etfs"):
            market_rows = self._market_rows(etf["etf_code"])
            events = []
            for reference in etf["events"]:
                current = {
                    **news_by_id[reference["news_id"]],
                    "reason": reference["reason"],
                }
                candidates = await self._candidates(current, search_start, run_date - timedelta(days=1))
                call_index += 1
                output, _ = await self._reply(
                    "history_user",
                    f"auto_fin_history_{call_index:03d}",
                    AutoFinHistoricalOutput,
                    etf=json.dumps(
                        {"etf_code": etf["etf_code"], "etf_name": etf["etf_name"]},
                        ensure_ascii=False,
                    ),
                    current_event=json.dumps(current, ensure_ascii=False),
                    candidates=json.dumps(candidates, ensure_ascii=False),
                )
                historical = []
                candidates_by_id = {row["news_id"]: row for row in candidates}
                for match in self._normalize(output, candidates, limit):
                    row = candidates_by_id[match.news_id]
                    historical.append(
                        {
                            **row,
                            "reason": match.reason,
                            "direction": match.direction,
                            "returns": self._returns(datetime.fromisoformat(row["event_time"]), market_rows),
                        },
                    )
                events.append({**current, "historical_events": historical})
            analyses.append(
                AutoFinEtfAnalysis.model_validate(
                    {
                        "etf_code": etf["etf_code"],
                        "etf_name": etf["etf_name"],
                        "events": events,
                    },
                ).model_dump(mode="json"),
            )
        path = self.workspace_path / str(self.config_value("resource_dir")) / str(run_date) / "auto_fin_analysis.jsonl"
        _write_jsonl(path, analyses)
        self.context["auto_fin_analyses"] = analyses
        self.context.response.metadata["analysis_count"] = len(analyses)
        return self.context.response
