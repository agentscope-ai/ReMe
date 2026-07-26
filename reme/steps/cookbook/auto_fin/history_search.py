"""Find historical events relevant to one selected ETF."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ....components import R
from ....schema import (
    AutoFinEtfHistoricalEvents,
    AutoFinEtfHistoricalResearch,
    AutoFinEtfSelection,
    AutoFinHistoricalEvent,
    AutoFinHistoricalEventReference,
    AutoFinMarketSample,
    AutoFinSelectedEvent,
)
from ...index._dedup import _ToolContextDedupMixin
from ._base import AutoFinStep, _write
from .topic import _plain_text


@R.register("auto_fin_history_search_step")
class AutoFinHistorySearchStep(AutoFinStep):
    """Find historical events and calculate their adjusted ETF returns."""

    @staticmethod
    def _trade_date(value: Any) -> date | None:
        text = str(value or "").replace("-", "")
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return None

    @staticmethod
    def _positive_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def _historical_source_candidates(self, source_path_value: str, news_id: str) -> list[Path]:
        """Return the declared source and a date-derived fallback within the workspace."""
        workspace = self.workspace_path.resolve()
        relative_path = Path(source_path_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Historical source path must be workspace-relative: {source_path_value}")
        source_path = (workspace / relative_path).resolve()
        try:
            source_path.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Historical source path is outside the workspace: {source_path_value}") from exc
        if source_path.name != "auto_fin_news_data.jsonl":
            raise ValueError(f"Historical source must be an Auto Fin news file: {source_path_value}")

        candidates = [source_path]
        news_date = news_id.partition("_")[0][:8]
        try:
            parsed_date = datetime.strptime(news_date, "%Y%m%d").date()
        except ValueError:
            parsed_date = None
        if parsed_date is not None:
            inferred_path = workspace / "daily" / parsed_date.isoformat() / "auto_fin_news_data.jsonl"
            if inferred_path not in candidates:
                candidates.append(inferred_path)
        return candidates

    async def _resolve_historical_event(
        self,
        reference: AutoFinHistoricalEventReference,
        current_news_ids: set[str],
        window_start: datetime,
        rows_by_path: dict[Path, list[dict[str, Any]]],
    ) -> AutoFinHistoricalEvent:
        """Resolve one Agent-selected identity from user-owned source files."""
        workspace = self.workspace_path.resolve()
        if reference.news_id in current_news_ids:
            raise ValueError(f"History Agent returned a current news item: {reference.news_id}")

        candidates = self._historical_source_candidates(reference.source_path, reference.news_id)
        matches: list[tuple[Path, dict[str, Any]]] = []
        for source_path in candidates:
            if not source_path.is_file():
                continue
            rows = rows_by_path.get(source_path)
            if rows is None:
                rows = await self._read_jsonl(source_path)
                rows_by_path[source_path] = rows
            matches.extend((source_path, row) for row in rows if str(row.get("news_id") or "") == reference.news_id)
        if len(matches) != 1:
            raise ValueError(
                f"Historical news_id must resolve exactly once from {reference.source_path} "
                f"or its ID-derived daily file: {reference.news_id}",
            )

        source_path, row = matches[0]
        event_time = self._published_at(row)
        if event_time is None:
            raise ValueError(f"Historical news has no valid publication time: {reference.news_id}")
        if event_time >= window_start:
            raise ValueError(f"History Agent returned an event inside the current news window: {reference.news_id}")
        event_title = str(row.get("title") or "").strip()
        event_content = _plain_text(str(row.get("content") or event_title))
        if not event_title or not event_content:
            raise ValueError(f"Historical news has no usable title or content: {reference.news_id}")

        return AutoFinHistoricalEvent(
            reason=reference.reason,
            news_id=reference.news_id,
            source_path=source_path.relative_to(workspace).as_posix(),
            event_time=event_time,
            event_title=event_title,
            event_content=event_content,
        )

    async def _resolve_historical_events(
        self,
        references: list[AutoFinHistoricalEventReference],
        current_news_ids: set[str],
        window_start: datetime,
    ) -> tuple[list[AutoFinHistoricalEvent], list[str]]:
        """Resolve valid references and report invalid ones without stopping the ETF."""
        rows_by_path: dict[Path, list[dict[str, Any]]] = {}
        events_by_news_id: dict[str, AutoFinHistoricalEvent] = {}
        limitations = []
        for reference in references:
            try:
                event = await self._resolve_historical_event(
                    reference,
                    current_news_ids,
                    window_start,
                    rows_by_path,
                )
            except (OSError, ValueError) as exc:
                limitation = f"跳过无法解析的历史新闻 {reference.news_id}: {exc}"
                self.logger.warning(f"[{self.name}] {limitation}")
                limitations.append(limitation)
                continue
            events_by_news_id.setdefault(reference.news_id, event)
        resolved = sorted(events_by_news_id.values(), key=lambda event: (event.event_time, event.news_id))
        return resolved, limitations

    async def _calculate_samples(
        self,
        etf_code: str,
        events: list[AutoFinHistoricalEvent],
        decision_at: datetime,
    ) -> tuple[list[AutoFinMarketSample], list[str]]:
        if not events:
            return [], []
        start_date = min(event.event_time.date() for event in events).strftime("%Y%m%d")
        end_date = decision_at.date().strftime("%Y%m%d")
        daily, factors = await asyncio.gather(
            self._fetch("fund_daily", ts_code=etf_code, start_date=start_date, end_date=end_date),
            self._fetch("fund_adj", ts_code=etf_code, start_date=start_date, end_date=end_date),
        )
        factors_by_date = {
            trade_date: self._positive_float(row.get("adj_factor"))
            for row in factors
            if (trade_date := self._trade_date(row.get("trade_date"))) is not None
        }
        daily_by_date = {
            trade_date: {
                "trade_date": trade_date,
                "open": self._positive_float(row.get("open")),
                "close": self._positive_float(row.get("close")),
                "adj_factor": factors_by_date.get(trade_date),
            }
            for row in daily
            if (trade_date := self._trade_date(row.get("trade_date"))) is not None
            and datetime.combine(trade_date, time(15, 0)) <= decision_at
        }
        rows = [daily_by_date[trade_date] for trade_date in sorted(daily_by_date)]
        row_indexes = {row["trade_date"]: index for index, row in enumerate(rows)}
        samples = []
        limitations = []
        for event in events:
            event_time = event.event_time
            event_date = event_time.date()
            row_index = row_indexes.get(event_date)
            price_type = None
            if row_index is not None and event_time.time() < time(9, 30):
                price_type = "open"
            elif row_index is not None and event_time.time() < time(15, 0):
                price_type = "close"
            else:
                row_index = next(
                    (index for index, row in enumerate(rows) if row["trade_date"] > event_date),
                    None,
                )
                price_type = "open" if row_index is not None else None

            if row_index is None or price_type is None:
                limitations.append(f"{event_time.isoformat()} 之后没有已完成的 ETF 日线")
                samples.append(
                    AutoFinMarketSample(
                        event_time=event.event_time,
                        reaction_summary="事件之后没有可用的已完成日线。",
                    ),
                )
                continue

            entry_row = rows[row_index]
            raw_price = entry_row[price_type]
            adj_factor = entry_row["adj_factor"]
            entry_clock = time(9, 30) if price_type == "open" else time(15, 0)
            entry_time = datetime.combine(entry_row["trade_date"], entry_clock)
            if raw_price is None or adj_factor is None:
                limitations.append(f"{entry_row['trade_date']} 缺少有效的 {price_type} 或 ETF 复权因子")
                samples.append(
                    AutoFinMarketSample(
                        event_time=event.event_time,
                        reaction_summary="买入点缺少有效价格或复权因子。",
                    ),
                )
                continue

            future_returns = []
            first_close_index = row_index if price_type == "open" else row_index + 1
            adjusted_entry = raw_price * adj_factor
            for future_row in rows[first_close_index : first_close_index + 10]:
                raw_close = future_row["close"]
                close_factor = future_row["adj_factor"]
                if raw_close is None or close_factor is None:
                    limitations.append(f"{future_row['trade_date']} 缺少有效 close 或 ETF 复权因子")
                    break
                future_returns.append(
                    {
                        "horizon": len(future_returns) + 1,
                        "trade_date": future_row["trade_date"],
                        "raw_close": raw_close,
                        "adj_factor": close_factor,
                        "cumulative_return": raw_close * close_factor / adjusted_entry - 1,
                    },
                )
            if len(future_returns) < 10:
                limitations.append(f"{event_time.isoformat()} 只有 {len(future_returns)} 个已完成的未来收盘点")
            samples.append(
                AutoFinMarketSample.model_validate(
                    {
                        "event_time": event.event_time,
                        "entry": {
                            "entry_time": entry_time,
                            "trade_date": entry_row["trade_date"],
                            "price_type": price_type,
                            "raw_price": raw_price,
                            "adj_factor": adj_factor,
                        },
                        "future_returns": future_returns,
                        "reaction_summary": f"按复权日线计算了 {len(future_returns)} 个未来收盘点。",
                    },
                ),
            )
        return samples, list(dict.fromkeys(limitations))

    async def execute(self):
        assert self.context is not None
        item = AutoFinEtfSelection.model_validate(self._required("auto_fin_current_etf"))
        events = [AutoFinSelectedEvent.model_validate(event) for event in self._required("auto_fin_current_events")]
        index = int(self._required("auto_fin_current_index"))
        window_start = datetime.fromisoformat(str(self._required("auto_fin_window_start")))
        decision_at = datetime.fromisoformat(str(self._required("auto_fin_decision_at")))
        search_events = [event.model_dump(mode="json", exclude={"news_id"}) for event in events]
        label = f"{item.etf_code}({item.etf_name})"
        tool_context_id = f"auto_fin_history_{index:02d}_{item.etf_code}_{uuid4().hex}"
        try:
            history, history_path = await self._reply(
                "history_search_user",
                f"auto_fin_history_{index:02d}_{item.etf_code}",
                AutoFinEtfHistoricalEvents,
                tool_context_id=tool_context_id,
                etf_code=item.etf_code,
                etf_name=item.etf_name,
                events=str(search_events),
                window_start=window_start.isoformat(),
                workspace_root=str(self.workspace_path),
            )
        finally:
            if self.app_context is not None:
                contexts = self.app_context.metadata.get(_ToolContextDedupMixin.TOOL_CONTEXTS_KEY)
                if isinstance(contexts, dict):
                    contexts.pop(tool_context_id, None)
                    if not contexts:
                        self.app_context.metadata.pop(_ToolContextDedupMixin.TOOL_CONTEXTS_KEY, None)
        if (history.etf_code, history.etf_name) != (item.etf_code, item.etf_name):
            raise ValueError(f"History Agent changed ETF {label!r}")
        resolved_events, resolution_limitations = await self._resolve_historical_events(
            history.historical_events,
            {event.news_id for event in events},
            window_start,
        )
        samples, market_limitations = await self._calculate_samples(
            item.etf_code,
            resolved_events,
            decision_at,
        )
        enriched_events = [
            event.model_copy(
                update={
                    "market_entry": sample.entry,
                    "future_returns": sample.future_returns,
                },
            )
            for event, sample in zip(resolved_events, samples, strict=True)
        ]
        enriched_history = AutoFinEtfHistoricalResearch(
            etf_code=history.etf_code,
            etf_name=history.etf_name,
            historical_events=enriched_events,
            limitations=list(dict.fromkeys([*resolution_limitations, *market_limitations])),
        )
        _write(
            history_path,
            json.dumps(enriched_history.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        )
        self.context["auto_fin_current_history"] = enriched_history.model_dump(mode="json")
        self.context["auto_fin_current_history_resource"] = str(history_path)
        self.logger.info(
            f"[{self.name}] ready etf={label!r} events={len(enriched_events)} "
            f"limitations={len(enriched_history.limitations)}",
        )
        return self.context.response
