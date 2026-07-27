"""Build the current Auto Fin topic timelines."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time
from html.parser import HTMLParser
from typing import Any

from ....components import R
from ....schema import AutoFinEtfsOutput
from ._base import AutoFinStep, _news_id, _write_jsonl

NEWS_TITLE_MAX_CHARS = 200
NEWS_CONTENT_MAX_CHARS = 1200
NEWS_TOTAL_CONTENT_MAX_CHARS = 60_000
ETF_CANDIDATE_LIMIT = 150
ETF_OUTPUT_LIMIT = 20


class _NewsTextExtractor(HTMLParser):
    """Extract visible text without retaining markup, links, or image URLs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag in {"script", "style"}:
            self.ignored_depth += 1
        elif tag in {"br", "div", "h1", "h2", "h3", "h4", "li", "p"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in {"div", "h1", "h2", "h3", "h4", "li", "p"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _NewsTextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def _normalized_key(value: Any) -> str:
    """Normalize display labels used only for deterministic deduplication."""
    return re.sub(r"\s+", "", str(value or "")).casefold()


@R.register("auto_fin_topic_step")
class AutoFinTopicStep(AutoFinStep):
    """Match current news to a small set of liquid, representative ETFs."""

    async def _current_news(self, start: datetime, end: datetime) -> list[dict]:
        news = {}
        title_limit = max(0, int(self._value("news_title_max_chars", NEWS_TITLE_MAX_CHARS)))
        content_limit = max(0, int(self._value("news_content_max_chars", NEWS_CONTENT_MAX_CHARS)))
        total_content_limit = max(
            0,
            int(self._value("news_total_content_max_chars", NEWS_TOTAL_CONTENT_MAX_CHARS)),
        )
        first_day = date.fromisoformat(str(self._required("auto_fin_news_start")))
        last_day = date.fromisoformat(str(self._required("auto_fin_date")))
        for day in self._days(first_day, last_day):
            for row in await self._read_jsonl(self._news_path(day)):
                published_at = self._published_at(row)
                if published_at is None or not start < published_at <= end:
                    continue
                news_id = str(row.get("news_id") or _news_id(row, published_at))
                news.setdefault(
                    news_id,
                    {
                        "news_id": news_id,
                        "event_time": published_at.isoformat(),
                        "title": str(row.get("title") or "").strip()[:title_limit],
                        "content": _plain_text(str(row.get("content") or "")),
                    },
                )
        rows = sorted(news.values(), key=lambda row: (row["event_time"], row["news_id"]))
        per_news_limit = min(content_limit, total_content_limit // len(rows)) if rows else 0
        for row in rows:
            row["content"] = row["content"][:per_news_limit]
        return rows

    async def _filtered_etfs(self, trade_date: date) -> list[dict[str, str]]:
        basics = await self._fetch(
            "etf_basic",
            list_status="L",
            fields="ts_code,csname,extname,cname,index_code,index_name,list_status",
        )
        daily = await self._fetch(
            "fund_daily",
            trade_date=trade_date.strftime("%Y%m%d"),
            fields="ts_code,trade_date,amount",
        )

        basic_by_code: dict[str, dict[str, Any]] = {}
        for row in basics:
            code = str(row.get("ts_code") or row.get("code") or "").strip().upper()
            name = str(
                row.get("csname") or row.get("name") or row.get("extname") or row.get("cname") or "",
            ).strip()
            if code and name and str(row.get("list_status") or "L").upper() == "L":
                basic_by_code.setdefault(code, {**row, "code": code, "name": name})

        amount_by_code: dict[str, float] = {}
        for row in daily:
            code = str(row.get("ts_code") or row.get("code") or "").strip().upper()
            try:
                amount = float(row.get("amount"))
            except (TypeError, ValueError):
                continue
            if code in basic_by_code and math.isfinite(amount) and amount >= 0:
                amount_by_code[code] = max(amount, amount_by_code.get(code, -math.inf))

        ranked = sorted(
            ({**basic_by_code[code], "amount": amount} for code, amount in amount_by_code.items()),
            key=lambda row: (-row["amount"], row["code"]),
        )
        selected: list[dict[str, str]] = []
        seen_names: set[str] = set()
        seen_indexes: set[str] = set()
        limit = max(0, int(self._value("etf_candidate_limit", ETF_CANDIDATE_LIMIT)))
        if not limit:
            return selected
        for row in ranked:
            name_key = _normalized_key(row["name"])
            index_keys = {
                key
                for key in (
                    _normalized_key(row.get("index_code")),
                    _normalized_key(row.get("index_name")),
                )
                if key
            }
            if name_key in seen_names or index_keys & seen_indexes:
                continue
            selected.append({"code": row["code"], "name": row["name"]})
            seen_names.add(name_key)
            seen_indexes.update(index_keys)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _repair_news_ids(
        output: AutoFinEtfsOutput,
        news: list[dict[str, Any]],
    ) -> tuple[AutoFinEtfsOutput, dict[str, str]]:
        """Repair a mistyped timestamp only when the content hash is unambiguous."""
        news_ids = {str(row["news_id"]) for row in news}
        ids_by_suffix: dict[str, list[str]] = {}
        for news_id in news_ids:
            _, separator, suffix = news_id.rpartition("_")
            if separator and suffix:
                ids_by_suffix.setdefault(suffix, []).append(news_id)

        data = output.model_dump(mode="json")
        repairs: dict[str, str] = {}
        for item in data["etfs"]:
            for event in item["events"]:
                event["reason"] = event["reason"].strip()
                news_id = event["news_id"].strip()
                event["news_id"] = news_id
                if news_id in news_ids:
                    continue
                _, separator, suffix = news_id.rpartition("_")
                candidates = ids_by_suffix.get(suffix, []) if separator else []
                if len(candidates) == 1:
                    event["news_id"] = candidates[0]
                    repairs[news_id] = candidates[0]
        return AutoFinEtfsOutput.model_validate(data), repairs

    @staticmethod
    def _normalize_selection(
        output: AutoFinEtfsOutput,
        news: list[dict[str, Any]],
        etfs: list[dict[str, str]],
    ) -> tuple[AutoFinEtfsOutput, bool]:
        """Canonicalize, filter, deduplicate, sort, and limit Agent selections."""
        news_order = {str(row["news_id"]): index for index, row in enumerate(news)}
        news_ids = set(news_order)
        candidates = {str(row["code"]).strip().upper(): str(row["name"]).strip() for row in etfs}
        selected: dict[str, dict[str, Any]] = {}
        for item in output.etfs:
            code = item.etf_code.strip().upper()
            name = candidates.get(code)
            if name is None:
                continue
            normalized = selected.setdefault(
                code,
                {"etf_code": code, "etf_name": name, "events": [], "seen_news_ids": set()},
            )
            for event in item.events:
                if not event.reason or event.news_id not in news_ids or event.news_id in normalized["seen_news_ids"]:
                    continue
                normalized["events"].append(event.model_dump(mode="json"))
                normalized["seen_news_ids"].add(event.news_id)

        normalized_items = []
        for item in selected.values():
            events = sorted(item["events"], key=lambda event: news_order[event["news_id"]])
            if events:
                normalized_items.append(
                    {
                        "etf_code": item["etf_code"],
                        "etf_name": item["etf_name"],
                        "events": events,
                    },
                )
            if len(normalized_items) >= ETF_OUTPUT_LIMIT:
                break
        normalized_output = AutoFinEtfsOutput.model_validate({"etfs": normalized_items})
        changed = normalized_output.model_dump(mode="json") != output.model_dump(mode="json")
        return normalized_output, changed

    async def execute(self):
        assert self.context is not None
        decision_at = datetime.fromisoformat(str(self._required("auto_fin_decision_at")))
        previous = date.fromisoformat(str(self._required("auto_fin_previous_trade_date")))
        window_start = datetime.combine(previous, time(15))
        news = await self._current_news(window_start, decision_at)
        self.logger.info(
            f"[{self.name}] start window=({window_start.isoformat()},{decision_at.isoformat()}] news={len(news)} "
            f"content_chars={sum(len(row['content']) for row in news)}",
        )
        resource_dir = self.workspace_path / "resource" / decision_at.date().isoformat()
        news_path = resource_dir / "filtered_news.jsonl"
        etf_path = resource_dir / "filtered_etf.jsonl"
        etfs = await self._filtered_etfs(previous)
        _write_jsonl(news_path, news)
        _write_jsonl(etf_path, etfs)
        output, output_path = await self._reply(
            "topic_user",
            "auto_fin_topic",
            AutoFinEtfsOutput,
            output_suffix=".jsonl",
            jsonl_field="etfs",
            window_start=window_start.isoformat(),
            decision_at=decision_at.isoformat(),
            filtered_news_path=str(news_path),
            filtered_etf_path=str(etf_path),
        )
        output, repairs = self._repair_news_ids(output, news)
        output, normalized = self._normalize_selection(output, news, etfs)
        if repairs:
            self.logger.warning(f"[{self.name}] repaired mistyped news IDs: {repairs}")
        if normalized:
            self.logger.info(f"[{self.name}] normalized ETF selections")
        if repairs or normalized:
            _write_jsonl(output_path, output.model_dump(mode="json")["etfs"])
        self.context["auto_fin_window_start"] = window_start.isoformat()
        self.context["auto_fin_etfs"] = output.model_dump(mode="json")["etfs"]
        self.context["auto_fin_etfs_resource"] = str(output_path)
        self.context["auto_fin_filtered_news"] = str(news_path)
        self.context.response.metadata.update({"news_count": len(news), "etf_count": len(output.etfs)})
        self.logger.info(
            f"[{self.name}] done etfs={len(output.etfs)} "
            f"events={sum(len(item.events) for item in output.etfs)} news_path={news_path} etf_path={etf_path}",
        )
        return self.context.response
