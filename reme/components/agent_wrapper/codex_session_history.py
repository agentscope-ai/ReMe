"""Read append provenance and tool calls from Codex's native rollout."""

import json
from pathlib import Path
import re

from ...constants import PROACTIVE_MESSAGE_ID_PREFIX
from ...schema import SessionMessage, SessionSnapshot

_PREFIX = "reme-proactive:"


def item_id(session_id: str, message_id: str, turn_id: str | None, delivery: str) -> str:
    """Keep append identity and presentation metadata in the native message ID."""
    return f"{_PREFIX}{session_id}:{delivery}:{turn_id or ''}:{message_id}"


def _history_path(home: Path, identifier: str) -> Path:
    """Find an immutable rollout ID, including a reverted or archived base file."""
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
        raise ValueError("Invalid Codex history base ID")
    matches = [
        path
        for directory in ("sessions", "archived_sessions")
        for path in (home / directory).rglob(f"rollout-*{identifier}.jsonl")
        if path.name.endswith((f"-{identifier}.jsonl", f"_{identifier}.jsonl"))
    ]
    if not matches:
        raise FileNotFoundError(f"Codex history base not found: {identifier}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous Codex history base: {identifier}")
    return matches[0]


def _entries(path: Path, home: Path, seen: set[Path], *, byte_limit: int | None = None):
    """Follow native history_base prefixes without reading later writes in a parent."""
    resolved = path.resolve()
    if not resolved.is_relative_to(home):
        raise ValueError("Codex session path is outside its configured home")
    if resolved in seen:
        raise ValueError("Cyclic Codex history base")
    seen.add(resolved)
    with resolved.open("rb") as stream:
        while byte_limit is None or stream.tell() < byte_limit:
            line = stream.readline()
            if not line:
                if byte_limit is not None and stream.tell() < byte_limit:
                    raise ValueError("Codex history base is shorter than its recorded prefix")
                break
            if byte_limit is not None and stream.tell() > byte_limit:
                raise ValueError("Codex history base prefix ends inside a record")
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if byte_limit is None and not line.endswith(b"\n"):
                    break
                raise
            if entry.get("type") == "session_meta" and (base := entry.get("payload", {}).get("history_base")):
                limit = base["end_byte_offset"]
                if not isinstance(limit, int) or limit < 0:
                    raise ValueError("Invalid Codex history base prefix")
                yield from _entries(_history_path(home, base["thread_id"]), home, seen, byte_limit=limit)
            yield entry


def read_appends(path: str | None, home: Path) -> list[SessionMessage]:
    """Recover accepted appends from native files, including immutable fork prefixes."""
    if path is None:
        return []
    resolved, home = Path(path).resolve(), home.resolve()
    if not resolved.is_relative_to(home):
        raise ValueError("Codex session path is outside its configured home")
    if not resolved.exists():
        return []
    messages: dict[str, SessionMessage] = {}
    for entry in _entries(resolved, home, set()):
        if entry.get("type") != "response_item":
            continue
        payload = entry.get("payload", {})
        identifier = payload.get("id")
        if payload.get("type") != "message" or not isinstance(identifier, str) or not identifier.startswith(_PREFIX):
            continue
        _, origin, delivery, turn_id, message_id = identifier.split(":", 4)
        message = SessionMessage(
            id=f"{PROACTIVE_MESSAGE_ID_PREFIX}{message_id}",
            role=payload["role"],
            content="".join(
                part["text"] for part in payload["content"] if part.get("type") in {"input_text", "output_text"}
            ),
            source="proactive",
            created_at=entry.get("timestamp"),
            turn_id=turn_id or None,
            delivery=delivery,
            origin_session_id=origin,
        )
        previous = messages.get(message.id)
        if previous is not None and (previous.content, previous.role) != (message.content, message.role):
            raise ValueError("Codex append ID belongs to different content or role")
        messages.setdefault(message.id, message)
    return list(messages.values())


def read_tool_calls(path: Path | None, home: Path, turn_id: str | None = None) -> list[dict] | None:
    """Include native calls rejected before the SDK emits a tool item."""
    if path is None:
        return None
    home = home.resolve()
    path = path.resolve()
    if not path.is_relative_to(home):
        raise ValueError("Codex session path is outside its configured home")
    if not path.exists():
        return None
    current_turn, calls, details = None, {}, {}
    for entry in _entries(path, home, set()):
        payload = entry.get("payload", {})
        if entry.get("type") == "turn_context":
            current_turn = payload.get("turn_id")
        if turn_id is not None and current_turn != turn_id:
            continue
        kind = payload.get("type", "")
        if entry.get("type") == "response_item" and kind in {
            "function_call",
            "custom_tool_call",
            "web_search_call",
            "image_generation_call",
        }:
            identifier = payload.get("call_id") or payload.get("id")
            if not identifier:
                continue
            arguments = payload.get("arguments", payload.get("input", payload.get("action", {})))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls[(current_turn, identifier)] = {
                "id": identifier,
                "turn_id": current_turn,
                "name": payload.get("name") or kind.removesuffix("_call"),
                "arguments": arguments,
            }
        elif entry.get("type") == "event_msg" and kind == "item_completed":
            item = payload.get("item", {})
            details[item.get("id")] = item
            native_kind = item.get("type", "")
            if native_kind.lower() in {
                "commandexecution",
                "filechange",
                "mcptoolcall",
                "dynamictoolcall",
                "collabagenttoolcall",
                "websearch",
                "imageview",
                "imagegeneration",
                "clock",
                "sleep",
            }:
                identifier = item["id"]
                calls.setdefault(
                    (current_turn, identifier),
                    {
                        "id": identifier,
                        "turn_id": current_turn,
                        "name": item.get("tool")
                        or (
                            "webSearch"
                            if native_kind.lower() == "websearch"
                            else native_kind[:1].lower() + native_kind[1:]
                        ),
                        "arguments": item,
                    },
                )
    for call in calls.values():
        item = details.get(call["id"], {})
        call["actions"] = item.get("parsed_cmd", [])
        call["cwd"] = item.get("cwd")
    return list(calls.values())


def merge_appends(snapshot: SessionSnapshot, messages: list[SessionMessage]) -> SessionSnapshot:
    """Insert messages after their conversation anchors, preserving append order."""
    pending: dict[str | None, list[SessionMessage]] = {}
    existing = {message.id for message in snapshot.messages}
    for message in messages:
        if message.id not in existing:
            pending.setdefault(message.turn_id, []).append(message)
    merged = pending.pop(None, [])
    for index, message in enumerate(snapshot.messages):
        merged.append(message)
        following = snapshot.messages[index + 1] if index + 1 < len(snapshot.messages) else None
        if following is None or following.turn_id != message.turn_id:
            merged.extend(pending.pop(message.turn_id, []))
    for remaining in pending.values():
        merged.extend(remaining)
    snapshot.messages = merged
    return snapshot
