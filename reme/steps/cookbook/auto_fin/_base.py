"""Shared helpers for the Auto Fin workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from ....components.outbound_proxy import BaseOutboundProxy
from ....enumeration import ComponentEnum
from ....utils.tushare import create_tushare_api
from ...base_step import BaseStep, Ref

AGENT_INPUT_LOG_LIMIT = 2000
AGENT_OUTPUT_LOG_LIMIT = 4000
SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _news_hash(row: dict[str, Any]) -> str:
    src = str(row.get("src") or "")
    content = str(row.get("content") or "")
    return hashlib.sha256(f"{src}{content}".encode()).hexdigest()[:4]


def _news_id(row: dict[str, Any], published_at: datetime) -> str:
    return f"{published_at:%Y%m%d%H%M%S}_{_news_hash(row)}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records),
    )


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


class AutoFinStep(BaseStep):
    """Shared Auto Fin helpers."""

    outbound_proxy: BaseOutboundProxy | None = Ref(BaseOutboundProxy, ComponentEnum.OUTBOUND_PROXY, optional=True)

    def _value(self, key: str, default: Any = None) -> Any:
        assert self.context is not None
        return self.context.get(key, self.kwargs.get(key, default))

    def _required(self, key: str) -> Any:
        assert self.context is not None
        if (value := self.context.get(key)) is None:
            raise RuntimeError(f"Auto Fin data is missing: {key}")
        return value

    async def _reply(
        self,
        prompt_name: str,
        resource_name: str,
        model: type[BaseModel],
        *,
        output_suffix: str = ".json",
        jsonl_field: str | None = None,
        tool_context_id: str | None = None,
        **values: str,
    ) -> tuple[BaseModel, Path]:
        """Send a complete prompt directly and persist its structured reply."""
        if self.agent_wrapper is None:
            raise RuntimeError("Auto Fin analysis requires an agent_wrapper")
        if Path(resource_name).name != resource_name:
            raise ValueError(f"Invalid Auto Fin resource name: {resource_name}")
        if output_suffix not in {".json", ".jsonl"}:
            raise ValueError(f"Invalid Auto Fin output suffix: {output_suffix}")

        prompt = self.prompt_format(prompt_name, **values)
        output_path = (
            self.workspace_path
            / "resource"
            / str(self._required("auto_fin_date"))
            / f"{resource_name}_output{output_suffix}"
        )
        started_at = perf_counter()
        serialized_input = json.dumps(prompt, ensure_ascii=False)
        input_preview, input_truncated = self._text_preview(serialized_input, AGENT_INPUT_LOG_LIMIT)
        self.logger.info(
            f"[{self.name}] agent input prompt={prompt_name} schema={model.__name__} "
            f"query_chars={len(prompt)} truncated={str(input_truncated).lower()} query={input_preview}",
        )
        agent_kwargs: dict[str, Any] = {"output_schema": model}
        if tool_context_id:
            agent_kwargs["tool_context_id"] = tool_context_id
        result = await self.agent_wrapper.reply(prompt, **agent_kwargs)
        if not isinstance(result, dict):
            raise TypeError("Auto Fin Agent reply must be a dictionary")
        value = result.get("structured_output")
        if value is None:
            raise ValueError(f"Auto Fin Agent returned no structured output: {self._preview(result)}")
        output = value if isinstance(value, model) else model.model_validate(value)
        payload = output.model_dump(mode="json")
        serialized_output = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if jsonl_field is None:
            _write(output_path, f"{serialized_output}\n")
        else:
            records = payload.get(jsonl_field)
            if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
                raise ValueError(f"Auto Fin JSONL field must contain objects: {jsonl_field}")
            _write_jsonl(output_path, records)
        output_preview, output_truncated = self._text_preview(serialized_output, AGENT_OUTPUT_LOG_LIMIT)
        self.logger.info(
            f"[{self.name}] agent output prompt={prompt_name} schema={model.__name__} "
            f"elapsed={perf_counter() - started_at:.2f}s chars={len(serialized_output)} "
            f"resource={output_path} truncated={str(output_truncated).lower()} output={output_preview}",
        )
        return output, output_path

    @staticmethod
    def _text_preview(text: str, limit: int) -> tuple[str, bool]:
        limit = max(0, limit)
        truncated = len(text) > limit
        return (f"{text[:limit]}...<truncated>" if truncated else text), truncated

    @staticmethod
    def _preview(value: Any, limit: int = 1000) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= limit else f"{text[:limit]}...<truncated>"

    @property
    def _proxy_url(self) -> str | None:
        return self.outbound_proxy.http_url if self.outbound_proxy is not None else None

    async def _fetch(self, endpoint: str, **kwargs) -> list[dict[str, Any]]:
        provider = self._value("tushare_provider")
        details = " ".join(
            f"{key}={kwargs[key]}" for key in ("exchange", "src", "start_date", "end_date") if key in kwargs
        )
        provider_name = "injected" if provider is not None else "sdk"
        started_at = perf_counter()
        self.logger.debug(
            f"[{self.name}] tushare fetch start endpoint={endpoint} provider={provider_name} "
            f"proxy={bool(self._proxy_url)} {details}",
        )
        try:
            if provider is not None:
                value = provider(endpoint, **kwargs)
                rows = _records(await value if asyncio.iscoroutine(value) else value)
            else:
                token = os.getenv("TUSHARE_TOKEN", "").strip()
                if not token:
                    raise RuntimeError("TUSHARE_TOKEN is required for Auto Fin")
                api = create_tushare_api(token, proxy_url=self._proxy_url)
                rows = _records(await asyncio.to_thread(getattr(api, endpoint), **kwargs))
        except Exception:
            self.logger.exception(
                f"[{self.name}] tushare fetch failed endpoint={endpoint} elapsed={perf_counter() - started_at:.2f}s "
                f"{details}",
            )
            raise
        self.logger.debug(
            f"[{self.name}] tushare fetch done endpoint={endpoint} records={len(rows)} "
            f"elapsed={perf_counter() - started_at:.2f}s {details}",
        )
        return rows

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
    def _published_at(row: dict[str, Any]) -> datetime | None:
        value = row.get("pub_time") or row.get("published_at") or row.get("datetime")
        if not value:
            return None
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y%m%d %H:%M:%S")
            except ValueError:
                return None
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            parsed = parsed.astimezone(SHANGHAI_TIMEZONE).replace(tzinfo=None)
        return parsed
