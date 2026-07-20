"""auto_memory_codex — record a Codex session from its transcript path.

The Codex plugin's Stop hook provides ``session_id`` and ``transcript_path``.
This step validates the path, loads the JSONL transcript, deduplicates by
payload id (or call_id), renders messages and tool calls, and delegates to
AutoMemoryStep.

The persisted Codex rollout schema (per the ``toolpath-codex`` parser and
OpenAI's upstream types):

* Top-level envelope: ``{"timestamp", "type", "payload"}``
* ``response_item`` is the conversational row; ``payload.type`` discriminates:

  * ``message`` — user/assistant text with ``role`` and ``content`` blocks
    (``input_text`` / ``output_text``)
  * ``function_call`` — model-invoked tool call: ``name``, ``arguments``
    (JSON string), ``call_id``
  * ``function_call_output`` — tool result: ``call_id``, ``output``
  * ``custom_tool_call`` — custom tool: ``name``, ``input``, ``call_id``
  * ``custom_tool_call_output`` — custom tool result: ``call_id``, ``output``
  * ``reasoning`` — intentionally skipped (private model reasoning)

* Other top-level types (``session_meta``, ``event_msg``, ``turn_context``,
  ``session_state``, ``compacted``) are not conversational — they are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .auto_memory import AutoMemoryStep
from ...components import R
from ...components.agent_wrapper import CcFileSessionStore

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
        """Dedup entries by payload id (or call_id) and store the increment.

        ``message`` and ``function_call`` / ``custom_tool_call`` payloads carry
        an ``id`` field.  ``function_call_output`` and
        ``custom_tool_call_output`` use ``call_id`` instead.
        """
        if not session_id:
            return []
        store = self._codex_store()
        key = {"session_id": session_id}
        # Keep conversational entries: those with a payload id or call_id.
        entries = [
            e
            for e in entries
            if isinstance(e, dict)
            and isinstance(e.get("payload"), dict)
            and (e["payload"].get("id") or e["payload"].get("call_id"))
        ]
        existing = await store.load(key) or []
        seen = {
            e["payload"].get("id") or e["payload"].get("call_id")
            for e in existing
            if isinstance(e, dict) and isinstance(e.get("payload"), dict)
        }
        increment = [e for e in entries if (e["payload"].get("id") or e["payload"].get("call_id")) not in seen]
        await store.append(key, increment)
        return increment

    def _codex_store(self) -> CcFileSessionStore:
        root = self.file_store.workspace_path / self._session_dir() / self._STORE_SUBDIR
        return CcFileSessionStore(root)

    # ----- rendering: raw Codex entries -> plain agent messages -------------

    @classmethod
    def _codex_entries_to_messages(cls, entries: list[dict]) -> list[dict[str, str]]:
        """Render ``response_item`` rows into ``{role, name, content}`` dicts.

        Handles all conversational payload types: ``message`` (user / assistant
        text), ``function_call`` / ``custom_tool_call`` (agent tool invocations),
        and ``function_call_output`` / ``custom_tool_call_output`` (tool results).
        ``reasoning`` payloads are intentionally skipped.
        """
        messages: list[dict[str, str]] = []
        for record in entries:
            if not isinstance(record, dict):
                continue
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            msg = cls._render_codex_payload(payload)
            if msg:
                messages.append(msg)
        return messages

    @classmethod
    def _render_codex_payload(cls, payload: dict) -> dict[str, str] | None:
        """Dispatch a single ``response_item`` payload to the correct renderer."""
        ptype = payload.get("type", "")

        if ptype == "message":
            return cls._render_message_payload(payload)
        if ptype == "function_call":
            return cls._render_function_call_payload(payload)
        if ptype == "function_call_output":
            return cls._render_function_call_output_payload(payload)
        if ptype == "custom_tool_call":
            return cls._render_custom_tool_call_payload(payload)
        if ptype == "custom_tool_call_output":
            return cls._render_custom_tool_call_output_payload(payload)
        # reasoning, etc. — intentionally skipped
        return None

    # -- message -----------------------------------------------------------

    @classmethod
    def _render_message_payload(cls, payload: dict) -> dict[str, str] | None:
        """Render a ``message`` payload (user or assistant turn)."""
        role = payload.get("role", "")
        if role not in ("user", "assistant"):
            return None

        content = payload.get("content", "")
        text = cls._render_codex_content(content)
        if not text:
            return None
        return {"role": role, "name": role, "content": text}

    # -- function_call -----------------------------------------------------

    @classmethod
    def _render_function_call_payload(cls, payload: dict) -> dict[str, str] | None:
        """Render a ``function_call`` payload as a virtual assistant tool-use message.

        ``arguments`` is a JSON string (per the upstream schema), not a dict.
        We round-trip through :func:`json.loads` + :func:`json.dumps` for a
        compact single-line representation.
        """
        name = payload.get("name", "?")
        arguments = payload.get("arguments", "")
        try:
            args = json.dumps(json.loads(arguments), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, ValueError):
            args = str(arguments)
        if len(args) > _TOOL_EXCERPT:
            args = args[:_TOOL_EXCERPT] + "..."
        return {
            "role": "assistant",
            "name": "assistant",
            "content": f"[tool {name}({args})]",
        }

    # -- function_call_output ----------------------------------------------

    @classmethod
    def _render_function_call_output_payload(cls, payload: dict) -> dict[str, str] | None:
        """Render a ``function_call_output`` payload as a virtual tool-result message."""
        output = payload.get("output", "")
        excerpt = str(output).strip()
        if len(excerpt) > _TOOL_EXCERPT:
            excerpt = excerpt[:_TOOL_EXCERPT] + "..."
        if not excerpt:
            return None
        return {
            "role": "user",
            "name": "user",
            "content": f"[tool_result {excerpt}]",
        }

    # -- custom_tool_call --------------------------------------------------

    @classmethod
    def _render_custom_tool_call_payload(cls, payload: dict) -> dict[str, str] | None:
        """Render a ``custom_tool_call`` payload as a virtual assistant tool-use message.

        ``input`` is a JSON string (per the upstream schema).
        """
        name = payload.get("name", "?")
        raw_input = payload.get("input", "")
        try:
            inp = json.dumps(json.loads(raw_input), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, ValueError):
            inp = str(raw_input)
        if len(inp) > _TOOL_EXCERPT:
            inp = inp[:_TOOL_EXCERPT] + "..."
        return {
            "role": "assistant",
            "name": "assistant",
            "content": f"[tool {name}({inp})]",
        }

    # -- custom_tool_call_output -------------------------------------------

    @classmethod
    def _render_custom_tool_call_output_payload(cls, payload: dict) -> dict[str, str] | None:
        """Render a ``custom_tool_call_output`` payload as a virtual tool-result message."""
        output = payload.get("output", "")
        excerpt = str(output).strip()
        if len(excerpt) > _TOOL_EXCERPT:
            excerpt = excerpt[:_TOOL_EXCERPT] + "..."
        if not excerpt:
            return None
        return {
            "role": "user",
            "name": "user",
            "content": f"[tool_result {excerpt}]",
        }

    # -- content blocks ----------------------------------------------------

    @classmethod
    def _render_codex_content(cls, content: Any) -> str:
        """Render Codex message content blocks to plain text.

        Codex ``ContentPart`` only has ``input_text`` and ``output_text``
        (both carry a ``text`` field).  Everything else is non-text and
        intentionally skipped.
        """
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("input_text", "output_text"):
                if t := (block.get("text") or "").strip():
                    parts.append(t)
        return "\n".join(p for p in parts if p).strip()
