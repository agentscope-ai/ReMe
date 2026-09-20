"""Shared Markdown, context, and Agent helpers for the Auto Fin workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, NamedTuple
from uuid import uuid4

import aiofiles
import frontmatter
import yaml
from pydantic import BaseModel

from reme.steps import BaseStep
from reme.steps.file_io import get_path_lock, validate_filename_component

from .schema import AutoFinReportOutput

AGENT_INPUT_LOG_LIMIT = 2000
AGENT_OUTPUT_LOG_LIMIT = 4000
NOTE_CHAR_LIMIT = 30_000
TITLE_BYTE_LIMIT = 180

_WIKILINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")
_HYBRID_WIKILINK = re.compile(r"(?P<wikilink>\[\[(?P<inner>[^\[\]\n]+)\]\])\((?P<destination>[^()\n]+)\)")
_HEADING = re.compile(r"^#+\s*")
# A note's stem has to survive two consumers: the filesystem, which rejects
# `<>:"/\|?*` and control characters, and `WikilinkHandler`, whose targets stop
# at `[`, `]` and `#`. Leaving the latter in place turns `- [[<path>]]` into a
# link that resolves to some other note (a trailing `#` reads as an anchor)
# or to no link at all.
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\[\]#\x00-\x1f]')

FIRST_RUN_NOTICE = "今日暂无更早时段的推荐，本次为当日首次生成。"
DISCLAIMER = "> 未接入可靠行情数据；本文只提供新闻研究和回顾线索，不提供收益、目标价或买卖建议。"


class WrittenReport(NamedTuple):
    """One persisted note, described by what actually reached the file."""

    path: str
    """Workspace-relative POSIX path of the note."""

    sources: list[str]
    """Workspace-relative paths of the links that survived validation."""

    body: str
    """The validated body as written, with dangling wikilinks downgraded to plain text."""


class _TextExtractor(HTMLParser):
    """Flatten HTML into whitespace-separated text."""

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


def plain_text(value: str) -> str:
    """Return one HTML fragment as collapsed plain text."""
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601 for note frontmatter."""
    return datetime.now(timezone.utc).isoformat()


def normalize_title(raw: str, fallback: str) -> str:
    """Return a topic or Agent label as a safe, filesystem- and wikilink-safe filename stem."""
    title = _UNSAFE_FILENAME.sub("-", _HEADING.sub("", str(raw or "").strip()))
    title = re.sub(r"\s+", " ", title).strip(" .-")
    if title.lower().endswith(".md"):
        title = title[:-3].strip(" .-")
    title = _truncate(title or fallback).strip(" .-") or fallback
    if error := validate_filename_component(title, kind="title"):
        raise ValueError(f"Unable to produce a safe Auto Fin title from {raw!r}: {error}")
    return title


def _truncate(title: str) -> str:
    """Fit a title into the byte budget one filename component accepts, without splitting a character."""
    kept: list[str] = []
    size = 0
    for character in title:
        size += len(character.encode())
        if size > TITLE_BYTE_LIMIT:
            break
        kept.append(character)
    return "".join(kept)


def normalize_report(output: AutoFinReportOutput) -> AutoFinReportOutput:
    """Strip a duplicated H1 and fill in the fallbacks for one Agent report."""
    body = output.body.strip()
    if body.startswith("# "):
        body = body.partition("\n")[2].strip()
    return output.model_copy(
        update={
            "title": _HEADING.sub("", output.title.strip()) or "主题新闻观察",
            "description": output.description.strip() or "基于当前新闻与历史记忆的主题研究。",
            "body": body or "## 结论\n\n暂无可用结论。",
        },
    )


def normalize_hybrid_wikilinks(body: str) -> str:
    """Drop a redundant Markdown destination from an unambiguous wikilink hybrid."""

    def replace(match: re.Match[str]) -> str:
        inner = match.group("inner").strip()
        raw_target = inner.partition("|")[0].strip()
        destination = match.group("destination").strip().removeprefix("<").removesuffix(">").strip()
        return (
            match.group("wikilink")
            if destination in {raw_target, raw_target.partition("#")[0].strip()}
            else match.group(0)
        )

    return _HYBRID_WIKILINK.sub(replace, body)


def _valid_source_path(path: str) -> bool:
    """Return whether one wikilink target could be a workspace-relative Markdown path."""
    return bool(
        path
        and not path.startswith("/")
        and "\\" not in path
        and path.endswith(".md")
        and "." not in Path(path).parts
        and ".." not in Path(path).parts
        and not any(character in path for character in "[]|"),
    )


def validate_wikilinks(body: str, workspace: Path, exclude: Path) -> tuple[str, list[str]]:
    """Keep real in-workspace Markdown links and downgrade invalid links to plain text."""
    workspace = workspace.resolve()
    exclude = exclude.resolve()
    sources: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw_target, separator, raw_alias = match.group(1).strip().partition("|")
        path = raw_target.strip().partition("#")[0].strip()
        alias = (raw_alias.strip() if separator else "") or Path(path).stem.replace("_", " ")
        if not _valid_source_path(path):
            return alias
        resolved = (workspace / path).resolve()
        if not resolved.is_relative_to(workspace) or not resolved.is_file() or resolved == exclude:
            return alias
        if path not in sources:
            sources.append(path)
        return match.group(0)

    return _WIKILINK.sub(replace, body), sources


def resolve_note_path(day_dir: Path, title: str, *, existing: Path | None) -> tuple[str, Path]:
    """Return a title and path that neither reuse nor overwrite an unrelated note."""
    path = day_dir / f"{title}.md"
    index = 2
    while path != existing and path.exists():
        path = day_dir / f"{title}（{index}）.md"
        index += 1
    return path.stem, path


def find_note(day_dir: Path, *, kind: str, **matches: Any) -> Path | None:
    """Return this workflow's note in one day directory whose frontmatter matches.

    Every Markdown file in the directory is a candidate, including notes the user
    is editing by hand, so an unreadable or malformed one is skipped rather than
    allowed to abort the run.
    """
    for path in sorted(day_dir.glob("*.md")) if day_dir.is_dir() else ():
        try:
            metadata = frontmatter.load(path).metadata
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            continue
        if metadata.get("kind") == kind and all(metadata.get(key) == value for key, value in matches.items()):
            return path
    return None


def read_note(path: Path | None) -> str:
    """Return an earlier note's body for intra-day refinement, or the first-run notice."""
    if path is None:
        return FIRST_RUN_NOTICE
    try:
        return frontmatter.load(path).content.strip()[:NOTE_CHAR_LIMIT] or FIRST_RUN_NOTICE
    except (OSError, UnicodeError, ValueError, yaml.YAMLError):
        return FIRST_RUN_NOTICE


async def write_markdown(path: Path, body: str, metadata: dict[str, Any]) -> None:
    """Serialize one frontmatter Markdown document atomically under its path lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = frontmatter.dumps(frontmatter.Post(body.strip(), **metadata))
    async with await get_path_lock(path):
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            async with aiofiles.open(temporary, "w", encoding="utf-8") as stream:
                await stream.write(rendered if rendered.endswith("\n") else f"{rendered}\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class AutoFinStep(BaseStep):
    """Shared helpers for the steps in one Auto Fin RuntimeContext."""

    def _value(self, key: str, default: Any = None) -> Any:
        assert self.context is not None
        return self.context.get(key, self.kwargs.get(key, default))

    def _required(self, key: str) -> Any:
        assert self.context is not None
        if (value := self.context.get(key)) is None:
            raise RuntimeError(f"Auto Fin data is missing: {key}")
        return value

    @property
    def day_dir(self) -> Path:
        """Return the dated directory that holds this run's notes."""
        return self.workspace_path / str(self.config_value("daily_dir")) / str(self._required("auto_fin_date"))

    def _track(self, path: Path, existing: Path | None) -> None:
        """Record this write, and any note it replaced, for the auto_tag step."""
        assert self.context is not None
        changes = list(self.context.get("changes") or [])
        if existing is not None and existing != path:
            existing.unlink(missing_ok=True)
            changes.append({"change": "deleted", "path": existing.relative_to(self.workspace_path).as_posix()})
        changes.append(
            {
                "change": "modified" if existing == path else "added",
                "path": path.relative_to(self.workspace_path).as_posix(),
            },
        )
        self.context["changes"] = changes

    async def _write_report(
        self,
        filename: str,
        output: AutoFinReportOutput,
        *,
        kind: str,
        existing: Path | None = None,
        trailer: str = "",
        **metadata: Any,
    ) -> WrittenReport:
        """Write one report note, returning its path, valid sources, and validated body."""
        title, path = resolve_note_path(self.day_dir, filename, existing=existing)
        body, sources = validate_wikilinks(normalize_hybrid_wikilinks(output.body), self.workspace_path, path)
        await write_markdown(
            path,
            "\n\n".join(section for section in (body, trailer, DISCLAIMER) if section),
            {
                "name": title,
                "title": output.title,
                "description": output.description,
                "kind": kind,
                "generated_at": utc_now_iso(),
                **metadata,
            },
        )
        self._track(path, existing)
        relative = path.relative_to(self.workspace_path).as_posix()
        self.logger.info(f"[{self.name}] wrote kind={kind} path={relative} chars={len(body)} sources={len(sources)}")
        return WrittenReport(path=relative, sources=sources, body=body)

    async def _reply(
        self,
        prompt_name: str,
        model: type[BaseModel],
        job_tools: list[str] | None = None,
        injected_job_kwargs: dict[str, Any] | None = None,
        tool_context_id: str | None = None,
        **values: Any,
    ) -> BaseModel:
        """Ask the Agent for one structured report and validate it against ``model``."""
        if self.agent_wrapper is None:
            raise RuntimeError("Auto Fin analysis requires an agent_wrapper")
        prompt = self.prompt_format(prompt_name, **values)
        started_at = perf_counter()
        self.logger.info(
            f"[{self.name}] agent input prompt={prompt_name} schema={model.__name__} "
            f"query={self._preview(prompt, AGENT_INPUT_LOG_LIMIT)}",
        )
        kwargs: dict[str, Any] = {
            "output_schema": model,
            "builtin_tools": [],
            "use_builtin_tools": False,
            "skills": [],
            "job_tools": list(self.kwargs.get("job_tools") or []) if job_tools is None else job_tools,
        }
        if injected_job_kwargs:
            kwargs["injected_job_kwargs"] = injected_job_kwargs
        if tool_context_id:
            kwargs["tool_context_id"] = tool_context_id
        result = await self.agent_wrapper.reply(prompt, **kwargs)
        if not isinstance(result, dict) or result.get("structured_output") is None:
            raise ValueError(f"Auto Fin Agent returned no structured output: {self._preview(result)}")
        value = result["structured_output"]
        output = value if isinstance(value, model) else model.model_validate(value)
        rendered = self._preview(output.model_dump(), AGENT_OUTPUT_LOG_LIMIT)
        self.logger.info(
            f"[{self.name}] agent output prompt={prompt_name} schema={model.__name__} "
            f"elapsed={perf_counter() - started_at:.2f}s output={rendered}",
        )
        return output

    @staticmethod
    def _preview(value: Any, limit: int = 1000) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return f"{text[:limit]}...<truncated>" if len(text) > limit else text
