"""Shared file and agent operations for the two proactive roles."""

from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import frontmatter

from .pa import _PaStep
from ..file_io._file_io import detect_file_encoding, get_path_lock, read_file_safe, write_file_safe
from ..file_io._path import resolve_path
from ...constants import DEFAULT_MAX_BYTES

_INTEREST_NAME = re.compile(r"interest_([0-9]{6})(?:_[0-9]+)?\.md\Z")
_STATUSES = {"unsent", "ready", "sent", "discarded"}


class _InterestStep(_PaStep):
    """Keep durable decisions in Markdown and use a fresh agent thread per call."""

    def _now(self) -> datetime:
        timezone = self.app_context.app_config.timezone
        return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()

    def _path(self, path: str) -> Path:
        target, error = resolve_path(self.workspace_path, path)
        if error:
            raise ValueError(error)
        return target

    async def _memory(self) -> str:
        self._validate()
        content, _ = await read_file_safe(
            self._path("personal/REME_MEMORY.md"),
            max_bytes=DEFAULT_MAX_BYTES,
            require_complete=True,
        )
        if not content.strip():
            raise ValueError("personal/REME_MEMORY.md is empty")
        return content

    async def _agent(self, memory: str, details: dict, tools: list[str]) -> None:
        result = await self.agent_wrapper.reply(
            json.dumps({"personal_memory": memory, **details}, ensure_ascii=False, default=str),
            system_prompt=self.get_prompt("system_prompt"),
            job_tools=list(self.kwargs.get("job_tools", tools)),
        )
        if result["status"] != "completed":
            raise RuntimeError(result.get("error") or f"Proactive agent {result['status']}")

    async def _load_note(self, path: str) -> frontmatter.Post:
        content, _ = await read_file_safe(self._path(path), max_bytes=self.max_file_bytes(), require_complete=True)
        return frontmatter.loads(content)

    async def _save_note(self, path: str, post: frontmatter.Post) -> None:
        # Callers hold the existing per-path lock throughout read/modify/write.
        post["updated_at"] = self._now().isoformat()
        target = self._path(path)
        await write_file_safe(target, frontmatter.dumps(post), encoding=await detect_file_encoding(target))

    async def _update_note(self, path: str, previous: frontmatter.Post, **metadata) -> frontmatter.Post:
        lock = await get_path_lock(self._path(path))
        async with lock:
            post = await self._load_note(path)
            if frontmatter.dumps(post) != frontmatter.dumps(previous):
                raise ValueError(f"Interest changed before update: {path}")
            post.metadata.update(metadata)
            await self._save_note(path, post)
            return post

    async def _notes(self, previous: dict[str, frontmatter.Post] | None = None) -> dict[str, frontmatter.Post]:
        """Scan every date, repairing notes left between agent writes and post-processing."""
        notes = {}
        root = self._path(str(self.config_value("daily_dir")))
        for entry in sorted(root.glob("????-??-??/interest_*.md")):
            match = _INTEREST_NAME.fullmatch(entry.name)
            if not match or entry.is_symlink() or entry.parent.is_symlink() or not entry.is_file():
                continue
            try:
                created = datetime.strptime(f"{entry.parent.name} {match[1]}", "%Y-%m-%d %H%M%S")
            except ValueError:
                continue
            path = entry.relative_to(self.workspace_path).as_posix()
            lock = await get_path_lock(self._path(path))
            async with lock:
                post = await self._load_note(path)
                if not post.content.strip():
                    continue
                original = frontmatter.dumps(post)
                post.metadata.setdefault("id", f"interest_{created:%Y%m%d}_{entry.stem[9:]}")
                post.metadata.setdefault("status", "unsent")
                post.metadata.setdefault("created_at", created.replace(tzinfo=self._now().tzinfo).isoformat())
                post.metadata.setdefault("updated_at", post["created_at"])
                if previous is not None:
                    before = previous.get(path)
                    if before is None:
                        post["status"] = "unsent"
                    elif before["status"] in {"sent", "discarded"} and frontmatter.dumps(before) != original:
                        raise ValueError(f"Proactive agent changed a terminal interest: {path}")
                if post["status"] not in _STATUSES:
                    raise ValueError(f"Invalid interest status in {path}: {post['status']!r}")
                changed = previous is not None and (
                    path not in previous or frontmatter.dumps(previous[path]) != frontmatter.dumps(post)
                )
                if changed or frontmatter.dumps(post) != original:
                    await self._save_note(path, post)
                notes[path] = post
        return notes

    @staticmethod
    def _summaries(notes: dict[str, frontmatter.Post]) -> list[dict]:
        return [{**post.metadata, "path": path} for path, post in notes.items()]
