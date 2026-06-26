"""auto_memory_cc — record a Claude Code session, resolved from its session_id.

The ReMe plugin's Stop hook hands the server only a ``session_id`` (never the
messages), and it fires on *every* stop. Unlike :class:`AutoMemoryStep` — whose
callers have no session management, so it re-serializes ``Msg`` history into its
own dialog store — Claude Code already manages the session as a transcript on
disk. So this step manages everything through the :class:`CcFileSessionStore`
abstraction and avoids the ``Msg`` round-trip entirely:

1. **load** the outer Claude Code session's transcript entries (Claude Code side).
2. **save** the *raw* entries into ReMe's own CC SessionStore — ``append`` dedups
   by record ``uuid``, so this both copies the conversation into ReMe and tells
   us the **increment** since the last stop.
3. render only that increment into plain ``{role, name, content}`` messages and
   defer to :class:`AutoMemoryStep` for the daily-note write/merge.

Both the read (Claude Code side) and the copy (ReMe side) use the same
file-backed SessionStore, just rooted at different directories.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .auto_memory import AutoMemoryStep
from ...components import R
from ...components.agent_wrapper import CcFileSessionStore

# Whole-message-drop when a user turn is only Claude-Code-injected boilerplate.
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


@R.register("auto_memory_cc_step")
class AutoMemoryCCStep(AutoMemoryStep):
    """Resolve a Claude Code session_id to its *new* turns, then reuse AutoMemoryStep."""

    # Sub-directory under the session dir holding ReMe's copy of CC transcripts.
    _CC_STORE_SUBDIR = "claude_code"

    async def execute(self):
        assert self.context is not None
        session_id: str = self.context.get("session_id", "")
        cc_entries = await self._load_cc_session(session_id)
        new_entries = await self._save_cc_session(session_id, cc_entries)
        messages = self._entries_to_messages(new_entries)
        self.logger.info(
            f"[{self.name}] resolved Claude Code session session_id={session_id!r} "
            f"transcript={len(cc_entries)} new_entries={len(new_entries)} messages={len(messages)}",
        )
        self.context["messages"] = messages
        await super().execute()

    # Claude Code owns the session (transcript + the CC SessionStore copy made in
    # _save_cc_session); AutoMemoryStep's Msg-history dialog store does not apply.
    async def _save_session_messages(self, session_id: str, messages) -> None:  # noqa: D401
        return

    def _session_link(self, session_id: str) -> str:
        return f"[[{self._session_dir()}/{self._CC_STORE_SUBDIR}/{session_id}.jsonl]]"

    # ----- session: Claude Code side <-> ReMe CC SessionStore ----------------

    async def _load_cc_session(self, session_id: str) -> list[dict]:
        """Load the outer Claude Code transcript entries (Claude Code side)."""
        transcript_dir = self._resolve_transcript_dir(session_id)
        if transcript_dir is None:
            return []
        store = CcFileSessionStore(transcript_dir)
        return await store.load({"session_id": session_id}) or []

    async def _save_cc_session(self, session_id: str, cc_entries: list[dict]) -> list[dict]:
        """Copy raw CC entries into ReMe's CC SessionStore; return the increment.

        Only identity-bearing entries are copied: every conversational entry
        (user / assistant / attachment) carries a ``uuid``, while uuid-less rows
        are CC control bookkeeping (queue operations, last-prompt) that would
        otherwise re-copy on every stop. Dedup against the already-stored uuids
        yields exactly the turns added since the previous stop.
        """
        if not session_id:
            return []
        store = self._reme_cc_store()
        key = {"session_id": session_id}
        cc_entries = [e for e in cc_entries if isinstance(e, dict) and e.get("uuid")]
        existing = await store.load(key) or []
        seen = {e.get("uuid") for e in existing if isinstance(e, dict) and e.get("uuid")}
        increment = [e for e in cc_entries if e.get("uuid") not in seen]
        await store.append(key, increment)
        return increment

    def _reme_cc_store(self) -> CcFileSessionStore:
        root = self.file_store.workspace_path / self._session_dir() / self._CC_STORE_SUBDIR
        return CcFileSessionStore(root)

    @staticmethod
    def _projects_dir() -> Path:
        base = Path(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude").expanduser()
        return base if base.name == "projects" else base / "projects"

    def _resolve_transcript_dir(self, session_id: str) -> Path | None:
        """Return the project directory holding ``<session_id>.jsonl`` (newest)."""
        projects = self._projects_dir()
        if not session_id or not projects.is_dir():
            return None
        matches = list(projects.glob(f"*/{session_id}.jsonl"))
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0].parent

    # ----- rendering: raw CC entries -> plain agent messages -----------------

    @classmethod
    def _entries_to_messages(cls, entries: list[dict]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for record in entries:
            if not isinstance(record, dict) or record.get("type") not in ("user", "assistant"):
                continue
            message = record.get("message") or {}
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            text = cls._render_content(message.get("content", ""))
            if not text or cls._is_injected_only(text):
                continue
            messages.append({"role": role, "name": role, "content": text})
        return messages

    @classmethod
    def _render_content(cls, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
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
                excerpt = cls._render_content(inner) if isinstance(inner, list) else str(inner or "")
                excerpt = excerpt.strip()
                if len(excerpt) > _TOOL_EXCERPT:
                    excerpt = excerpt[:_TOOL_EXCERPT] + "..."
                parts.append(f"[tool_result {excerpt}]")
            # thinking blocks are private reasoning -> dropped
        return "\n".join(p for p in parts if p).strip()

    @staticmethod
    def _is_injected_only(text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith(_INJECTED_TAGS):
            return False
        remaining = re.sub(r"<([a-z-]+)>.*?</\1>", "", stripped, flags=re.DOTALL)
        return len(remaining.strip()) < 16
