"""Shared state and file helpers for daily-paper steps."""

import json
import os
import re
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import aiofiles
import frontmatter
from pydantic import BaseModel

from ...base_step import BaseStep
from ...file_io._file_io import get_path_lock

_STATE_PREFIX = "daily_paper_"
_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_OutputT = TypeVar("_OutputT", bound=BaseModel)


def structured_output(result: dict[str, Any], model: type[_OutputT]) -> _OutputT:
    """Validate an agent wrapper's structured output."""
    value = result.get("structured_output")
    return value if isinstance(value, model) else model.model_validate(value)


def strip_frontmatter(body: str) -> str:
    """Remove one model-generated YAML frontmatter block."""
    return _FRONTMATTER_PATTERN.sub("", body.strip(), count=1).strip()


async def write_atomic(path: Path, content: str | bytes) -> None:
    """Write through a sibling temporary file under the repository path lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = await get_path_lock(path)
    async with lock:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload = content.encode("utf-8") if isinstance(content, str) else content
        try:
            async with aiofiles.open(temp_path, "wb") as stream:
                await stream.write(payload)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


async def write_markdown(path: Path, body: str, metadata: dict[str, Any]) -> None:
    """Serialize a frontmatter Markdown document atomically."""
    rendered = frontmatter.dumps(frontmatter.Post(body.strip(), **metadata))
    await write_atomic(path, rendered if rendered.endswith("\n") else f"{rendered}\n")


class DailyPaperStep(BaseStep):
    """Shared helpers for steps in one daily-paper RuntimeContext."""

    def _skip(self) -> bool:
        assert self.context is not None
        return bool(self.context.get(f"{_STATE_PREFIX}skip", False))

    def _value(self, key: str, default: Any) -> Any:
        assert self.context is not None
        return self.context.get(key, self.kwargs.get(key, default))

    def _state(self, key: str) -> Any:
        assert self.context is not None
        return self.context.get(f"{_STATE_PREFIX}{key}")

    def _set_state(self, key: str, value: Any) -> None:
        assert self.context is not None
        self.context[f"{_STATE_PREFIX}{key}"] = value

    def _run_day(self) -> str:
        value = self._state("run_date")
        if not value:
            raise RuntimeError("daily-paper run date is not initialized")
        return str(value)

    def _manifest_path(self, day: str) -> Path:
        return self.workspace_path / self.config_value("metadata_dir") / "daily_paper" / f"{day}.json"

    async def _write_manifest(self, payload: dict[str, Any]) -> Path:
        path = self._manifest_path(self._run_day())
        await write_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        self._set_state("manifest", payload)
        self.logger.info(
            f"[{self.name}] manifest written path={self.to_workspace_relative(path)} "
            f"status={payload.get('status', '')}",
        )
        return path
