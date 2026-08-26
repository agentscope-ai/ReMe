"""Shared state and file helpers for daily-paper steps."""

import datetime as dt
import os
import re
import zoneinfo
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import aiofiles
import frontmatter
from pydantic import BaseModel

from reme.steps import BaseStep
from reme.steps.file_io import get_path_lock, validate_filename_component

# Number of papers selected, analyzed, and digested each run. Shared across steps.
PAPER_COUNT = 3
_STATE_PREFIX = "daily_paper_"
_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_MARKDOWN_HEADING_PATTERN = re.compile(r"^#+\s*")
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_CHINESE_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")
_OutputT = TypeVar("_OutputT", bound=BaseModel)


def structured_output(result: dict[str, Any], model: type[_OutputT]) -> _OutputT:
    """Validate an agent wrapper's structured output."""
    value = result.get("structured_output")
    return value if isinstance(value, model) else model.model_validate(value)


def strip_frontmatter(body: str) -> str:
    """Remove one model-generated YAML frontmatter block."""
    return _FRONTMATTER_PATTERN.sub("", body.strip(), count=1).strip()


def replace_surrogates(value: str) -> str:
    """Replace invalid Unicode surrogate code points with replacement characters."""
    return _SURROGATE_PATTERN.sub("\ufffd", value)


def _replace_surrogates_recursive(value: Any) -> Any:
    """Replace surrogates in strings nested in frontmatter metadata."""
    if isinstance(value, str):
        return replace_surrogates(value)
    if isinstance(value, dict):
        return {_replace_surrogates_recursive(key): _replace_surrogates_recursive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_surrogates_recursive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_surrogates_recursive(item) for item in value)
    return value


def normalize_chinese_title(raw: str, fallback: str) -> str:
    """Return one safe Chinese title that can also be used as the filename stem."""
    title = replace_surrogates(str(raw or "").strip())
    title = _MARKDOWN_HEADING_PATTERN.sub("", title)
    if title.lower().endswith(".md"):
        title = title[:-3]
    title = _UNSAFE_FILENAME_CHARS.sub("-", title)
    title = re.sub(r"\s+", " ", title).strip(" .-")
    if not title or not _CHINESE_PATTERN.search(title):
        title = fallback
    title = _UNSAFE_FILENAME_CHARS.sub("-", title).strip(" .-")
    if error := validate_filename_component(title, kind="title"):
        raise ValueError(f"Unable to produce a safe daily-paper title from {raw!r}: {error}")
    return title


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string for note metadata."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def now(timezone: str | None = None) -> dt.datetime:
    """Return the current time in an IANA timezone, falling back to local time."""
    if not timezone:
        return dt.datetime.now()
    try:
        return dt.datetime.now(zoneinfo.ZoneInfo(timezone))
    except (KeyError, zoneinfo.ZoneInfoNotFoundError):
        return dt.datetime.now()


def iter_note_metadata(day_dir: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(path, frontmatter metadata)`` for each readable Markdown note in a day."""
    if not day_dir.is_dir():
        return
    for path in sorted(day_dir.glob("*.md")):
        try:
            yield path, frontmatter.load(path).metadata
        except (OSError, UnicodeError, ValueError):
            continue


def resolve_unique_note_path(
    day_dir: Path,
    title: str,
    *,
    taken: set[str],
    taken_suffix: str,
    disk_suffix: str,
    existing: Path | None,
) -> tuple[str, Path]:
    """Disambiguate a note title against already-used titles and on-disk collisions."""
    if title in taken:
        title = f"{title}{taken_suffix}"
    path = day_dir / f"{title}.md"
    if path.exists() and path != existing:
        title = f"{title}{disk_suffix}"
        path = day_dir / f"{title}.md"
    return title, path


async def write_atomic(path: Path, content: str | bytes) -> None:
    """Write through a sibling temporary file under the repository path lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = await get_path_lock(path)
    async with lock:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload = replace_surrogates(content).encode("utf-8") if isinstance(content, str) else content
        try:
            async with aiofiles.open(temp_path, "wb") as stream:
                await stream.write(payload)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


async def write_markdown(path: Path, body: str, metadata: dict[str, Any]) -> None:
    """Serialize a frontmatter Markdown document atomically."""
    safe_body = replace_surrogates(body).strip()
    safe_metadata = _replace_surrogates_recursive(metadata)
    rendered = frontmatter.dumps(frontmatter.Post(safe_body, **safe_metadata))
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
