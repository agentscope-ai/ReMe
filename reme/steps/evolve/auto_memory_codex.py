"""auto_memory_codex — record a Codex session, resolved from its transcript path.

The ReMe Codex plugin's Stop hook hands the server a ``session_id`` and
``transcript_path`` (the Codex transcript JSONL file). Unlike
:class:`AutoMemoryCCStep` — which searches ``~/.claude/projects`` for Claude Code
transcripts — Codex already tells the hook exactly where the transcript lives.
So this step:

1. **resolves** the transcript path, with fallback when the hook provides a stale
   or non-existent path (known Codex bug with ``--continue`` and git worktrees).
2. **loads** the transcript entries, supporting both Codex's native format
   (``response_item`` / ``developer`` / ``input_text``) and Claude Code's format
   (``user`` / ``assistant`` / ``text``) as a compatibility fallback.
3. **saves** the *raw* entries into ReMe's own SessionStore — ``append`` dedups
   by record ``uuid``, so this both copies the conversation into ReMe and tells
   us the **increment** since the last stop.
4. renders only that increment into plain ``{role, name, content}`` messages and
   defers to :class:`AutoMemoryStep` for the daily-note write/merge.
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


@R.register("auto_memory_codex_step")
class AutoMemoryCodexStep(AutoMemoryStep):
    """Resolve a Codex transcript path to its *new* turns, then reuse AutoMemoryStep."""

    _CC_STORE_SUBDIR = "codex"

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

        # Resolve the real transcript path (with fallback for stale paths).
        resolved = await self._resolve_transcript_path(transcript_path, session_id)
        if resolved:
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
        return f"[[{self._session_dir()}/{self._CC_STORE_SUBDIR}/{session_id}.jsonl]]"

    # ----- transcript path resolution ---------------------------------------

    @staticmethod
    def _codex_sessions_dir() -> Path:
        """Return the root directory where Codex stores session transcripts."""
        codex_home = os.environ.get("CODEX_HOME", "~/.codex")
        return Path(codex_home).expanduser() / "sessions"

    async def _resolve_transcript_path(
        self,
        transcript_path: str,
        session_id: str,
    ) -> str | None:
        """Resolve the real transcript path, with fallback for stale paths.

        Known Codex bugs: ``transcript_path`` in the Stop hook payload can point
        to a stale session (after ``--continue`` / ``--resume``) or to a
        non-existent path (inside git worktrees). When the given path doesn't
        exist, search the Codex sessions directory for a JSONL file matching
        ``session_id`` (newest match wins).
        """
        if not transcript_path:
            return None
        path = Path(os.path.expanduser(transcript_path))
        if path.is_file():
            return transcript_path  # happy path — path is valid

        self.logger.info(
            f"[{self.name}] transcript_path not found, searching for session_id={session_id!r}",
        )
        sessions_dir = self._codex_sessions_dir()
        if not sessions_dir.is_dir():
            return None

        pattern = f"{session_id}.jsonl" if session_id else "*.jsonl"
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
        """Load Codex transcript entries from the provided path."""
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
        """Copy raw Codex entries into ReMe's SessionStore; return the increment.

        Only identity-bearing entries are copied: every conversational entry
        carries a ``uuid``, while uuid-less rows are control bookkeeping that
        would otherwise re-copy on every stop. Dedup against the already-stored
        uuids yields exactly the turns added since the previous stop.
        """
        if not session_id:
            return []
        store = self._codex_store()
        key = {"session_id": session_id}
        entries = [e for e in entries if isinstance(e, dict) and e.get("uuid")]
        existing = await store.load(key) or []
        seen = {e.get("uuid") for e in existing if isinstance(e, dict) and e.get("uuid")}
        increment = [e for e in entries if e.get("uuid") not in seen]
        await store.append(key, increment)
        return increment

    def _codex_store(self) -> CcFileSessionStore:
        root = self.file_store.workspace_path / self._session_dir() / self._CC_STORE_SUBDIR
        return CcFileSessionStore(root)

    # ----- rendering: raw Codex entries -> plain agent messages -------------

    @classmethod
    def _codex_entries_to_messages(cls, entries: list[dict]) -> list[dict[str, str]]:
        """Render Codex transcript entries into ``{role, name, content}`` messages.

        Codex stores entries as ``type: "response_item"`` with
        ``message.role: "developer"`` (assistant) or ``"user"``, and
        ``input_text`` content blocks. Non-conversation entries (system
        prompts, queue operations) are skipped.
        """
        messages: list[dict[str, str]] = []
        for record in entries:
            if not isinstance(record, dict):
                continue
            if record.get("type") != "response_item":
                continue
            msg = cls._render_codex_record(record)
            if msg:
                messages.append(msg)
        return messages

    @classmethod
    def _render_codex_record(cls, record: dict) -> dict[str, str] | None:
        """Render a single Codex ``response_item`` into a message dict."""
        message = record.get("message") or {}
        if not isinstance(message, dict):
            return None

        role = message.get("role", "")
        # Codex uses "developer" for assistant turns, "user" for user turns.
        if role == "developer":
            role = "assistant"
        if role not in ("user", "assistant"):
            return None

        content = message.get("content", "")
        text = cls._render_codex_content(content)
        if not text or cls._is_injected_only(text):
            return None
        return {"role": role, "name": role, "content": text}

    @classmethod
    def _render_codex_content(cls, content: Any) -> str:
        """Render Codex content blocks into plain text.

        Codex uses ``input_text`` blocks (cf. Claude Code's ``text``). Tool use
        and tool result blocks follow the same shape as CC and are handled here.
        """
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("text", "input_text"):
                if t := (block.get("text") or block.get("input_text") or "").strip():
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
