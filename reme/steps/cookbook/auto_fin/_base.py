"""Shared helpers for the Auto Fin workflow."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from ...base_step import BaseStep

AGENT_INPUT_LOG_LIMIT = 2000
AGENT_OUTPUT_LOG_LIMIT = 4000
SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"br", "div", "li", "p"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif tag in {"div", "li", "p"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


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


class AutoFinStep(BaseStep):
    """Shared Auto Fin helpers."""

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
        job_tools: list[str] | None = None,
        **values: str,
    ) -> tuple[BaseModel, Path]:
        """Send a complete prompt directly and persist its structured reply."""
        if self.agent_wrapper is None:
            raise RuntimeError("Auto Fin analysis requires an agent_wrapper")
        if Path(resource_name).name != resource_name:
            raise ValueError(f"Invalid Auto Fin resource name: {resource_name}")
        prompt = self.prompt_format(prompt_name, **values)
        output_path = (
            self.workspace_path / "resource" / str(self._required("auto_fin_date")) / f"{resource_name}_output.json"
        )
        started_at = perf_counter()
        serialized_input = json.dumps(prompt, ensure_ascii=False)
        input_preview, input_truncated = self._text_preview(serialized_input, AGENT_INPUT_LOG_LIMIT)
        self.logger.info(
            f"[{self.name}] agent input prompt={prompt_name} schema={model.__name__} "
            f"query_chars={len(prompt)} truncated={str(input_truncated).lower()} query={input_preview}",
        )
        reply_kwargs: dict[str, Any] = {"output_schema": model}
        if job_tools:
            reply_kwargs["job_tools"] = job_tools
        result = await self.agent_wrapper.reply(prompt, **reply_kwargs)
        if not isinstance(result, dict):
            raise TypeError("Auto Fin Agent reply must be a dictionary")
        value = result.get("structured_output")
        if value is None:
            raise ValueError(f"Auto Fin Agent returned no structured output: {self._preview(result)}")
        output = value if isinstance(value, model) else model.model_validate(value)
        serialized_output = self._write_output(output_path, output)
        output_preview, output_truncated = self._text_preview(serialized_output, AGENT_OUTPUT_LOG_LIMIT)
        self.logger.info(
            f"[{self.name}] agent output prompt={prompt_name} schema={model.__name__} "
            f"elapsed={perf_counter() - started_at:.2f}s chars={len(serialized_output)} "
            f"resource={output_path} truncated={str(output_truncated).lower()} output={output_preview}",
        )
        return output, output_path

    @staticmethod
    def _write_output(path: Path, model: BaseModel) -> str:
        """Persist a model as compact JSON and return the serialized text."""
        serialized = json.dumps(model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        _write(path, f"{serialized}\n")
        return serialized

    @staticmethod
    def _text_preview(text: str, limit: int) -> tuple[str, bool]:
        limit = max(0, limit)
        truncated = len(text) > limit
        return (f"{text[:limit]}...<truncated>" if truncated else text), truncated

    @staticmethod
    def _preview(value: Any, limit: int = 1000) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return AutoFinStep._text_preview(text, limit)[0]

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
