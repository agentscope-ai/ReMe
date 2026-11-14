"""Codex native display-history projection."""

import json
from pathlib import Path
import re

from ...enumeration import ChunkEnum
from ...schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk, TokenUsage
from ...schema.stream_chunk import stream_status

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


def item_chunks(item, *, event="end", session_id=None, turn_id=None) -> list[StreamChunk]:
    """Project the same native item snapshots for live streams and history."""
    payload = item.model_dump(mode="json", by_alias=True)
    common = {"event": event, "session_id": session_id, "turn_id": turn_id, "message_id": item.id}
    if item.type == "userMessage":
        return []
    if item.type in {"agentMessage", "plan", "reasoning"}:
        metadata = {
            "native": {key: value for key, value in payload.items() if key not in {"text", "summary", "content"}},
        }
        if item.type == "reasoning":
            return [
                StreamChunk(
                    **common,
                    chunk_type=ChunkEnum.THINK,
                    block_id=f"{item.id}:{kind}:{index}",
                    chunk=text,
                    metadata={**metadata, "kind": f"reasoning_{kind}", "index": index},
                )
                for kind in ("summary", "content")
                for index, text in enumerate(getattr(item, kind, None) or [])
            ]
        return [
            StreamChunk(
                **common,
                chunk_type=ChunkEnum.CONTENT if item.type == "agentMessage" else ChunkEnum.THINK,
                block_id=item.id,
                chunk=item.text,
                metadata={**metadata, "kind": item.type, "phase": payload.get("phase")},
            ),
        ]
    if item.type in {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
        "webSearch",
    }:
        native_status = getattr(item, "status", None)
        status = (
            stream_status(native_status)
            if native_status is not None
            else "running" if event == "start" else "completed"
        )
        if event == "end":
            if getattr(item, "error", None) or getattr(item, "success", None) is False:
                status = "failed"
            if getattr(item, "exit_code", None) not in {None, 0}:
                status = "failed"
        return [
            StreamChunk(
                **common,
                chunk_type=ChunkEnum.TOOL_CALL if event == "start" else ChunkEnum.TOOL_RESULT,
                block_id=item.id,
                tool_call_id=item.id,
                tool_call_name=getattr(item, "tool", None) or item.type,
                chunk=payload,
                status=status,
            ),
        ]
    return [StreamChunk(**common, chunk_type=ChunkEnum.DATA, block_id=item.id, chunk=payload)]


def session_snapshot(thread) -> SessionSnapshot:
    """Project native turn groups into display history."""
    messages = []
    for turn in thread.turns:
        users: list[str] = []
        chunks: list[StreamChunk] = []
        for wrapped in turn.items:
            item = wrapped.root
            if item.type == "userMessage":
                users.extend(part.root.text for part in item.content if part.root.type == "text")
            else:
                chunks.extend(item_chunks(item, session_id=thread.id, turn_id=turn.id))
        if users:
            messages.append(
                SessionMessage(id=f"{turn.id}:user", role="user", content="\n".join(users), turn_id=turn.id),
            )
        if getattr(turn, "error", None):
            chunks.append(StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=turn.error.message))
        status = stream_status(getattr(turn, "status", None))
        if chunks and status != "running":
            chunks.append(StreamChunk(chunk_type=ChunkEnum.REPLY_END, event="end", turn_id=turn.id, status=status))
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
    return SessionSnapshot(
        session_id=thread.id,
        state=session_status(thread),
        messages=messages,
        last_turn_id=thread.turns[-1].id if thread.turns else None,
    )
