"""Shared helpers for the Auto Fin workflow."""

from __future__ import annotations

from html.parser import HTMLParser
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from reme.steps import BaseStep

AGENT_INPUT_LOG_LIMIT = 2000
AGENT_OUTPUT_LOG_LIMIT = 4000


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
        model: type[BaseModel],
        job_tools: list[str] | None = None,
        injected_job_kwargs: dict[str, Any] | None = None,
        **values: str,
    ) -> BaseModel:
        if self.agent_wrapper is None:
            raise RuntimeError("Auto Fin analysis requires an agent_wrapper")
        prompt = self.prompt_format(prompt_name, **values)
        started_at = perf_counter()
        self.logger.info(
            f"[{self.name}] agent input prompt={prompt_name} schema={model.__name__} "
            f"query={self._preview(prompt, AGENT_INPUT_LOG_LIMIT)}",
        )
        kwargs: dict[str, Any] = {"output_schema": model}
        if job_tools:
            kwargs["job_tools"] = job_tools
        if injected_job_kwargs:
            kwargs["injected_job_kwargs"] = injected_job_kwargs
        result = await self.agent_wrapper.reply(prompt, **kwargs)
        if not isinstance(result, dict) or result.get("structured_output") is None:
            raise ValueError(f"Auto Fin Agent returned no structured output: {self._preview(result)}")
        value = result["structured_output"]
        output = value if isinstance(value, model) else model.model_validate(value)
        output_preview = self._preview(output.model_dump(), AGENT_OUTPUT_LOG_LIMIT)
        self.logger.info(
            f"[{self.name}] agent output prompt={prompt_name} schema={model.__name__} "
            f"elapsed={perf_counter() - started_at:.2f}s output={output_preview}",
        )
        return output

    @staticmethod
    def _preview(value: Any, limit: int = 1000) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return f"{text[:limit]}...<truncated>" if len(text) > limit else text
