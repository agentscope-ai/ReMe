"""Parse Claude Code transcript JSONL into a plain message slice.

Both routes (``reme-driver`` external, ``reme-service`` internal) accept
``transcript_path`` as the canonical ``sync`` input — Claude Code hooks
hand us the path, not the message list. Centralising the parse here
keeps the two routes producing identical slices for the same transcript.

The JSONL format is one record per line. The records we care about:

    {"type": "user",      "message": {"role": "user",      "content": str | list[block]}, ...}
    {"type": "assistant", "message": {"role": "assistant", "content": list[block]}, ...}

Everything else (``ai-title`` / ``mode`` / ``permission-mode`` /
``file-history-snapshot`` / ``attachment`` / ``last-prompt`` /
``queue-operation`` / ``system``) is metadata or harness chatter and is
ignored.

Content blocks we recognise (shape from the Anthropic message format):

* ``{"type": "text", "text": str}``                    — appended verbatim
* ``{"type": "tool_use", "name": str, "input": ...}``  — rendered as ``[tool <name>(<json excerpt>)]``
* ``{"type": "tool_result", "content": ...}``          — rendered as ``[tool_result <excerpt>]``
* ``{"type": "thinking", "thinking": str}``            — dropped (private reasoning)

User content frequently contains Claude-Code-injected boilerplate that
isn't part of the real conversation:

* ``<local-command-caveat>...``  — `bash` command warnings prepended to first user turn
* ``<local-command-stdout>...``  — output of slash commands
* ``<command-name>...``          — slash command label
* ``<command-message>...``       — slash command description
* ``<system-reminder>...``       — periodic harness reminders

These are filtered out (whole-message drop if the text is only injected
markers, partial strip otherwise) so the synchronizer sees the actual
user/assistant dialogue.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Whole-message-drop when the user content is only one of these wrappers.
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

# Heuristic: if the message starts with one of these AND is short / mostly markup,
# drop it. We keep the regex permissive — false negatives (a real message that
# looks like markup) are recoverable downstream; false positives (dropping real
# user text) are silent and worse.
_DROP_IF_STARTS_WITH = tuple(_INJECTED_TAGS)


def load_messages_from_transcript(
    transcript_path: str | Path,
    *,
    include_thinking: bool = False,
    tool_input_excerpt: int = 200,
) -> list[dict[str, str]]:
    """Read a Claude Code transcript JSONL → list of role/content dicts.

    Returns ``[{role, name, content}, ...]`` in source order, where
    ``role`` is ``"user"`` or ``"assistant"`` and ``name`` mirrors role
    (so the dicts are directly consumable by AgentScope's ``Msg``, which
    requires a ``name`` field). Empty list when the file is missing,
    empty, or contains no user/assistant turns.

    Parameters
    ----------
    transcript_path:
        Absolute or relative path to the transcript JSONL file.
    include_thinking:
        When True, include ``thinking`` blocks (assistant private reasoning).
        Default False — the synchronizer wants observable dialogue.
    tool_input_excerpt:
        Max chars of a ``tool_use`` input JSON to render inline. Default 200.
    """
    path = Path(transcript_path)
    if not path.is_file():
        return []

    messages: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        record_type = record.get("type")
        if record_type not in ("user", "assistant"):
            continue

        message = record.get("message") or {}
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        text = _render_content(
            message.get("content", ""),
            include_thinking=include_thinking,
            tool_input_excerpt=tool_input_excerpt,
        )
        if not text:
            continue
        if _is_injected_only(text):
            continue

        messages.append({"role": role, "name": role, "content": text})

    return messages


def _render_content(
    content: Any,
    *,
    include_thinking: bool,
    tool_input_excerpt: int,
) -> str:
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
            t = (block.get("text") or "").strip()
            if t:
                parts.append(t)

        elif btype == "thinking":
            if include_thinking:
                t = (block.get("thinking") or "").strip()
                if t:
                    parts.append(f"[thinking]\n{t}")

        elif btype == "tool_use":
            name = block.get("name", "?")
            try:
                inp = json.dumps(block.get("input"), ensure_ascii=False)[:tool_input_excerpt]
            except (TypeError, ValueError):
                inp = str(block.get("input"))[:tool_input_excerpt]
            parts.append(f"[tool {name}({inp})]")

        elif btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                excerpt = _render_content(
                    inner,
                    include_thinking=False,
                    tool_input_excerpt=tool_input_excerpt,
                )
            else:
                excerpt = str(inner or "")
            excerpt = excerpt.strip()
            if len(excerpt) > tool_input_excerpt:
                excerpt = excerpt[:tool_input_excerpt] + "..."
            parts.append(f"[tool_result {excerpt}]")

    return "\n".join(p for p in parts if p).strip()


def _is_injected_only(text: str) -> bool:
    """True if the text is composed entirely of Claude-Code-injected markers
    (no genuine user/assistant dialogue around them).
    """
    stripped = text.strip()
    if not stripped.startswith(_DROP_IF_STARTS_WITH):
        return False
    # If it starts with an injected tag, peek whether anything substantive
    # follows the closing tag. Cheap heuristic: strip all wrapped <tag>...</tag>
    # blocks and see what's left.
    remaining = re.sub(r"<([a-z-]+)>.*?</\1>", "", stripped, flags=re.DOTALL)
    return len(remaining.strip()) < 16  # arbitrary "essentially empty" threshold
