"""Codex display-history projection and file-native proactive message records."""

import datetime
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from ...enumeration import ChunkEnum
from ...schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk, TokenUsage

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def validate_identifier(value: str) -> str:
    """Keep session and caller-provided message identifiers out of path syntax."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("Session/message IDs must contain 1-128 letters, digits, underscores or hyphens")
    return value


def session_status(thread) -> SessionStatus:
    """Normalize native status without mistaking unloaded/error sessions for idle."""
    status = thread.status.root
    return SessionStatus(
        session_id=thread.id,
        status={"notLoaded": "not_loaded", "systemError": "error"}.get(status.type, status.type),
        active_flags=[getattr(flag, "value", flag) for flag in getattr(status, "active_flags", [])],
    )


def read_records(path: Path) -> list[dict]:
    """Read the application-owned provenance file; corruption fails explicitly."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_records(path: Path, records: list[dict]) -> None:
    """Atomically publish records before/after an injection; never rewrite a Codex rollout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def injected_ids(thread, home: Path) -> set[str]:
    """Recover accepted injections from the native rollout, including an interrupted publication.

    Codex 0.163 persists injected response items but omits them from its public
    display-history RPCs. Only our reserved IDs are read from this internal format.
    """
    raw_path = getattr(thread, "path", None)
    if not raw_path:
        raise ValueError("Appending requires a persisted Codex session")
    path = Path(raw_path).resolve()
    if not path.is_relative_to(home.resolve()):
        raise ValueError("Codex session path is outside its configured home")
    if not path.exists():
        return set()
    found = set()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break  # An unrelated native append may still be writing its final line.
            raise
        if entry.get("type") != "response_item":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message" or payload.get("role") != "assistant":
            continue
        item_id = payload.get("id", "")
        if isinstance(item_id, str) and item_id.startswith("reme-proactive-"):
            found.add(item_id)
    return found


def read_token_usage(thread, home: Path) -> TokenUsage:
    """Recover reported cumulative spending before starting a restored native session's next turn."""
    raw_path = getattr(thread, "path", None)
    if not raw_path:
        return TokenUsage()
    path = Path(raw_path).resolve()
    if not path.is_relative_to(home.resolve()):
        raise ValueError("Codex session path is outside its configured home")
    usage = TokenUsage()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    break
                raise
            payload = record.get("payload", {})
            if record.get("type") == "event_msg" and payload.get("type") == "token_count":
                if info := payload.get("info"):
                    usage = TokenUsage.from_provider(info["total_token_usage"])
    return usage


def new_record(message_id: str, content: str, thread) -> dict:
    """Anchor proactive messages after the last completed native turn."""
    return {
        "message": SessionMessage(
            id=f"reme-proactive-{message_id}",
            role="assistant",
            content=content,
            source="proactive",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            turn_id=thread.turns[-1].id if thread.turns else None,
        ).model_dump(mode="json"),
        "injected": False,
    }


def session_snapshot(thread, records: list[dict], recovered_ids: set[str]) -> SessionSnapshot:
    """Combine native turn groups and durable proactive messages in insertion order."""
    proactive: dict[str | None, list[SessionMessage]] = {}
    for record in records:
        message = SessionMessage.model_validate(record["message"])
        if record["injected"] or message.id in recovered_ids:
            proactive.setdefault(message.turn_id, []).append(message)
    messages = list(proactive.pop(None, []))
    for turn in thread.turns:
        users: list[str] = []
        chunks: list[StreamChunk] = []
        for wrapped in turn.items:
            item = wrapped.root
            if item.type == "userMessage":
                users.extend(part.root.text for part in item.content if part.root.type == "text")
            elif item.type == "agentMessage":
                chunks.append(StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk=item.text, block_id=item.id))
            elif item.type in {"reasoning", "plan"}:
                text = item.text if item.type == "plan" else "\n".join(item.summary or item.content or [])
                if text:
                    chunks.append(StreamChunk(chunk_type=ChunkEnum.THINK, chunk=text, block_id=item.id))
            elif item.type in {
                "commandExecution",
                "fileChange",
                "mcpToolCall",
                "dynamicToolCall",
                "collabAgentToolCall",
                "webSearch",
            }:
                chunks.append(
                    StreamChunk(
                        chunk_type=ChunkEnum.TOOL_RESULT,
                        chunk=item.model_dump(mode="json", by_alias=True),
                        tool_call_id=item.id,
                        tool_call_name=getattr(item, "tool", None) or item.type,
                        metadata={"state": getattr(getattr(item, "status", None), "value", "completed")},
                    ),
                )
        if users:
            messages.append(
                SessionMessage(id=f"{turn.id}:user", role="user", content="\n".join(users), turn_id=turn.id),
            )
        if getattr(turn, "error", None):
            chunks.append(StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=turn.error.message))
        if chunks:
            messages.append(
                SessionMessage(
                    id=f"{turn.id}:assistant",
                    role="assistant",
                    turn_id=turn.id,
                    chunks=chunks,
                    content="".join(str(chunk.chunk) for chunk in chunks if chunk.chunk_type == ChunkEnum.CONTENT),
                    completed=getattr(getattr(turn, "status", None), "value", "completed") != "inProgress",
                ),
            )
        messages.extend(proactive.pop(turn.id, []))
    # A native history migration/compaction can omit an old anchor. Retain its
    # user-owned proactive records rather than silently dropping messages.
    for remaining in proactive.values():
        messages.extend(remaining)
    return SessionSnapshot(session_id=thread.id, state=session_status(thread), messages=messages)
