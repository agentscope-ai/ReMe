"""auto_memory_codex — record a Codex session from its transcript path.

The Codex plugin's Stop hook provides ``session_id`` and ``transcript_path``.
This step validates the path, loads the JSONL transcript, deduplicates by
payload id, renders messages, and delegates to AutoMemoryStep.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .auto_memory import AutoMemoryStep
from ...components import R
from ...components.agent_wrapper import CcFileSessionStore

# Claude Code injected tags — same filter as AutoMemoryCCStep.
_INJECTED_TAGS = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<system-reminder>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
)
_TOOL_EXCERPT = 200
_SESSIONS_SUBDIR = "sessions"


@R.register("auto_memory_codex_step")
class AutoMemoryCodexStep(AutoMemoryStep):
    """Step that reads a Codex JSONL transcript and records new turns."""

    _STORE_SUBDIR = "codex"

    async def execute(self):
        assert self.context is not None
        session_id: str = self.context.get("session_id", "")
        transcript_path: str = self.context.get("transcript_path", "")

        if not session_id and not transcript_path:
            self.context.response.success = False
            self.context.response.answer = "Error: session_id or transcript_path is required"
            self.logger.warning(
                f"[{self.name}] missing both session_id and transcript_path",
            )
            return

        # Resolve and validate the transcript path.
        resolved = await self._resolve_transcript_path(transcript_path, session_id)
        if resolved is None:
            self.context.response.success = False
            self.context.response.answer = "Error: could not resolve transcript path"
            self.logger.warning(
                f"[{self.name}] unresolved transcript session_id={session_id!r} " f"given={transcript_path!r}",
            )
            return
        transcript_path = resolved

        entries = await self._load_codex_transcript(transcript_path)
        new_entries = await self._save_codex_session(session_id, entries)
        messages = self._codex_entries_to_messages(new_entries)
        self.logger.info(
            f"[{self.name}] resolved Codex session session_id={session_id!r} "
            f"transcript_path={transcript_path!r} "
            f"transcript={len(entries)} new_entries={len(new_entries)} messages={len(messages)}",
        )
        self.context["messages"] = messages
        await super().execute()

    # Codex owns the transcript; AutoMemoryStep's Msg-history dialog store does not apply.
    async def _save_session_messages(self, session_id: str, messages) -> None:  # noqa: D401
        return

    def _session_link(self, session_id: str) -> str:
        return f"[[{self._session_dir()}/{self._STORE_SUBDIR}/{session_id}.jsonl]]"

    # ----- path resolution & validation -------------------------------------

    @staticmethod
    def _codex_sessions_dir() -> Path:
        """Codex session storage root (``$CODEX_HOME/sessions``)."""
        codex_home = os.environ.get("CODEX_HOME", "~/.codex")
        return Path(codex_home).expanduser() / _SESSIONS_SUBDIR

    def _validate_transcript_path(self, path: Path) -> bool:
        """Check *path* is under ``$CODEX_HOME/sessions``."""
        try:
            path.resolve().relative_to(self._codex_sessions_dir().resolve())
            return True
        except ValueError:
            return False

    async def _resolve_transcript_path(
        self,
        transcript_path: str,
        session_id: str,
    ) -> str | None:
        """Resolve transcript path, with fallback search when the path doesn't exist."""
        if not transcript_path:
            return None
        path = Path(os.path.expanduser(transcript_path))
        if path.is_file():
            if not self._validate_transcript_path(path):
                self.logger.warning(
                    f"[{self.name}] transcript_path outside sessions dir: {path}",
                )
                return None
            return transcript_path

        self.logger.info(
            f"[{self.name}] transcript_path not found, searching for session_id={session_id!r}",
        )
        sessions_dir = self._codex_sessions_dir()
        if not sessions_dir.is_dir():
            return None

        # Codex names files rollout-<timestamp>-<session_id>.jsonl.
        pattern = f"rollout-*-{session_id}.jsonl" if session_id else "*.jsonl"
        candidates = sorted(
            sessions_dir.glob(f"**/{pattern}"),
            key=lambda p: p.stat().st_mtime if p.is_file() else 0,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.is_file():
                self.logger.info(
                    f"[{self.name}] fallback transcript found at {candidate}",
                )
                return str(candidate)

        return None

    # ----- transcript loading -----------------------------------------------

    async def _load_codex_transcript(self, transcript_path: str) -> list[dict]:
        """Read JSONL transcript entries from disk."""
        if not transcript_path:
            return []
        path = Path(os.path.expanduser(transcript_path))
        if not path.is_file():
            self.logger.warning(
                f"[{self.name}] transcript not found at {transcript_path!r}",
            )
            return []
        store = CcFileSessionStore(path.parent)
        key = {"session_id": path.name if path.suffix else path.stem}
        return await store.load(key) or []

    # ----- session dedup / store --------------------------------------------

    async def _save_codex_session(self, session_id: str, entries: list[dict]) -> list[dict]:
        """Dedup entries by ``payload.id`` and store the increment."""
        if not session_id:
            return []
        store = self._codex_store()
        key = {"session_id": session_id}
        # Only keep rows with a payload id — conversational entries.
        entries = [
            e for e in entries if isinstance(e, dict) and isinstance(e.get("payload"), dict) and e["payload"].get("id")
        ]
        existing = await store.load(key) or []
        seen = {
            e["payload"]["id"]
            for e in existing
            if isinstance(e, dict) and isinstance(e.get("payload"), dict) and e["payload"].get("id")
        }
        increment = [e for e in entries if e["payload"]["id"] not in seen]
        await store.append(key, increment)
        return increment

    def _codex_store(self) -> CcFileSessionStore:
        root = self.file_store.workspace_path / self._session_dir() / self._STORE_SUBDIR
        return CcFileSessionStore(root)

    # ----- rendering: raw Codex entries -> plain agent messages -------------

    @classmethod
    def _codex_entries_to_messages(cls, entries: list[dict]) -> list[dict[str, str]]:
        """Render ``response_item`` rows into ``{role, name, content}`` dicts."""
        messages: list[dict[str, str]] = []
        for record in entries:
            if not isinstance(record, dict):
                continue
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "message":
                continue
            msg = cls._render_codex_payload(payload)
            if msg:
                messages.append(msg)
        return messages

    @classmethod
    def _render_codex_payload(cls, payload: dict) -> dict[str, str] | None:
        """Extract role + rendered text from a message payload."""
        role = payload.get("role", "")
        if role not in ("user", "assistant"):
            return None

        content = payload.get("content", "")
        text = cls._render_codex_content(content)
        if not text or cls._is_injected_only(text):
            return None
        return {"role": role, "name": role, "content": text}

    @classmethod
    def _render_codex_content(cls, content: Any) -> str:
        """Render Codex content blocks (input_text/output_text/tool_use) to text."""
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("text", "input_text", "output_text"):
                if t := (block.get("text") or "").strip():
                    parts.append(t)
            elif btype == "tool_use":
                name = block.get("name", "?")
                try:
                    inp = json.dumps(block.get("input"), ensure_ascii=False)[:_TOOL_EXCERPT]
                except (TypeError, ValueError):
                    inp = str(block.get("input"))[:_TOOL_EXCERPT]
                parts.append(f"[tool {name}({inp})]")
            elif btype == "tool_result":
                inner = block.get("content")
                excerpt = cls._render_codex_content(inner) if isinstance(inner, list) else str(inner or "")
                excerpt = excerpt.strip()
                if len(excerpt) > _TOOL_EXCERPT:
                    excerpt = excerpt[:_TOOL_EXCERPT] + "..."
                parts.append(f"[tool_result {excerpt}]")
            # thinking blocks are private reasoning -> dropped
        return "\n".join(p for p in parts if p).strip()

    @staticmethod
    def _is_injected_only(text: str) -> bool:
        """Return True when a user turn is only Claude-Code-injected boilerplate."""
        import re

        stripped = text.strip()
        if not stripped.startswith(_INJECTED_TAGS):
            return False
        remaining = re.sub(r"<([a-z-]+)>.*?</\1>", "", stripped, flags=re.DOTALL)
        return len(remaining.strip()) < 16
