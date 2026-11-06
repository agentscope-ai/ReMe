"""Workspace files shared by Remy's collection, decision, and delivery steps."""

from contextlib import asynccontextmanager, AsyncExitStack
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import re

import frontmatter

from .pa import _PaStep
from ..file_io._file_io import detect_file_encoding, get_path_lock, read_file_safe, write_file_safe

_INTEREST_NAME = re.compile(r"interest_([0-9]{6})(?:_[0-9]+)?\.md\Z")


class _InterestStep(_PaStep):
    """Keep interest decisions in Markdown and model instructions in YAML."""

    def _note_path(self, filename: str) -> Path:
        match = _INTEREST_NAME.fullmatch(filename) if isinstance(filename, str) else None
        if match is None:
            raise ValueError("Expected an interest_HHMMSS.md filename")
        datetime.strptime(match[1], "%H%M%S")
        raw = self.workspace_path / str(self.config_value("daily_dir")) / filename
        if raw.is_symlink():
            raise ValueError("Interest files must not be symlinks")
        return self._path(str(raw))

    async def _agent(self, tools: list[str], *, injected: dict | None = None, **details) -> None:
        prompt = self._system_prompt(
            now=self._now().isoformat(),
            interests_path=self._personal_path("INTERESTS.md"),
            interest_pattern=self._path(str(self.config_value("daily_dir"))) / "interest_*.md",
            **details,
        )
        result = await self.agent_wrapper.reply(
            self.get_prompt("user_prompt"),
            system_prompt=prompt,
            job_tools=list(self.kwargs.get("job_tools", tools)),
            injected_job_kwargs=injected or {},
            activity_source="proactive",
            activity_session_id=self._state["session_id"],
        )
        if result["status"] != "completed":
            raise RuntimeError(result.get("error") or f"Proactive agent {result['status']}")

    async def _load_note(self, filename: str) -> frontmatter.Post:
        content, _ = await read_file_safe(
            self._note_path(filename),
            max_bytes=self.max_file_bytes(),
            require_complete=True,
        )
        return frontmatter.loads(content)

    async def _save_note(self, filename: str, post: frontmatter.Post) -> None:
        # Callers hold the existing per-path lock across read/modify/write.
        post["updated_at"] = self._now().isoformat()
        target = self._note_path(filename)
        await write_file_safe(target, frontmatter.dumps(post), encoding=await detect_file_encoding(target))

    @asynccontextmanager
    async def _locked_notes(self, filenames: list[str]):
        async with AsyncExitStack() as stack:
            for filename in sorted(filenames):
                await stack.enter_async_context(await get_path_lock(self._note_path(filename)))
            yield

    async def _notes(self) -> dict[str, frontmatter.Post]:
        notes = {}
        for entry in sorted(self._path(str(self.config_value("daily_dir"))).glob("interest_*.md")):
            if entry.is_symlink() or not entry.is_file() or not _INTEREST_NAME.fullmatch(entry.name):
                continue
            self._note_path(entry.name)
            async with self._locked_notes([entry.name]):
                post = await self._load_note(entry.name)
                if not post.content.strip():
                    continue
                if "status" not in post:
                    post["status"] = "unsent"
                    post.metadata.setdefault("created_at", self._now().isoformat())
                    await self._save_note(entry.name, post)
                if post["status"] not in {"unsent", "sent"}:
                    raise ValueError(f"Invalid interest status in {entry.name}: {post['status']!r}")
                notes[entry.name] = post
        return notes

    @staticmethod
    def _version(post: frontmatter.Post) -> str:
        return sha256(frontmatter.dumps(post).encode()).hexdigest()
