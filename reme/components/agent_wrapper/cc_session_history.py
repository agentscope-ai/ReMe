"""Display projection of Claude's full transcript, including pre-compaction history."""

from typing import Callable

from ...enumeration import ChunkEnum
from ...schema import SessionMessage, StreamChunk
from .cc_sessions import AppendRecord, SessionData, SessionRuntime, native_id


def text_content(content) -> str:
    """Keep text independent from thinking, tool results and native metadata."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content or [] if block.get("type") == "text")


def accepted_message(entry: dict, record: AppendRecord) -> SessionMessage:
    """Native history must agree with the stored intent before it can be published."""
    content = text_content((entry.get("message") or {}).get("content"))
    expected = record.message.content
    prefixed = f"[MESSAGE FROM NON-USER SOURCE - NOT USER INPUT]\n{expected}"
    if entry.get("type") != record.message.role or content not in (expected, prefixed):
        raise ValueError("Claude native message differs from its append record")
    return record.message.model_copy(deep=True)


def last_turn_id(entries: list[dict], data: SessionData) -> str | None:
    """Recover the accepted prompt, excluding injected observations and hidden inputs."""
    hidden = {*data.hidden_ids, *(record.native_id for record in data.appends)}
    return next(
        (
            entry.get("uuid")
            for entry in reversed(entries)
            if entry.get("type") == "user"
            and entry.get("uuid") not in hidden
            and not (entry.get("isMeta") or entry.get("isSidechain") or entry.get("isCompactSummary"))
            and text_content((entry.get("message") or {}).get("content"))
        ),
        None,
    )


def messages(session_id: str, entries: list[dict], data: SessionData, convert: Callable) -> list[SessionMessage]:
    """Group model blocks by API message ID, associating tool results with their calls."""
    from claude_agent_sdk._internal.message_parser import parse_message

    records = {record.native_id: record for record in data.appends}
    hidden = {
        *data.hidden_ids,
        native_id(session_id, "seed"),
        *(native_id(session_id, f"flush:{key}") for key in records),
    }
    result: list[SessionMessage] = []
    grouped: dict[str, SessionMessage] = {}
    tools: dict[str, SessionMessage] = {}
    offsets: dict[str, int] = {}
    seen = set()
    turn_id = None
    for entry in entries:
        role, identifier = entry.get("type"), entry.get("uuid")
        if role not in {"user", "assistant"} or entry.get("isSidechain"):
            continue
        if not identifier or identifier in hidden or identifier in seen:
            continue
        seen.add(identifier)
        if identifier in records:
            result.append(accepted_message(entry, records[identifier]))
            continue
        raw = entry.get("message") or {}
        if raw.get("model") == "<synthetic>" or entry.get("isCompactSummary") or entry.get("isMeta"):
            continue
        model_message = parse_message({**entry, "session_id": session_id})
        message_id = raw.get("id") or identifier
        chunks = convert(
            model_message,
            session_id=session_id,
            include_text=role == "assistant",
            message_id=message_id,
            block_offset=offsets.get(message_id, 0),
        )
        if chunks:
            chunks[0].metadata["native"] = entry
        elif role == "assistant" and raw.get("content"):
            chunks = [StreamChunk(chunk_type=ChunkEnum.DATA, event="end", chunk=raw, metadata={"native": entry})]
        content = text_content(raw.get("content"))
        if role == "user":
            for chunk in chunks:
                owner = tools.get(chunk.tool_call_id)
                if owner is not None:
                    chunk.turn_id = owner.turn_id
                    owner.chunks.append(chunk)
            if not content:
                continue
            turn_id = identifier
            result.append(
                SessionMessage(
                    id=identifier,
                    role="user",
                    content=content,
                    created_at=entry.get("timestamp"),
                    turn_id=turn_id,
                ),
            )
            continue
        message = grouped.get(message_id)
        if message is None:
            message = SessionMessage(
                id=message_id,
                role="assistant",
                content="",
                created_at=entry.get("timestamp"),
                turn_id=turn_id,
            )
            grouped[message_id] = message
            result.append(message)
        message.content += content
        message.chunks.extend(chunks)
        offsets[message_id] = offsets.get(message_id, 0) + len(raw.get("content", []))
        for chunk in chunks:
            chunk.turn_id = turn_id
            if chunk.tool_call_id:
                tools[chunk.tool_call_id] = message
    return result


def overlay_live(history: list[SessionMessage], runtime: SessionRuntime) -> None:
    """Expose detached partial output while the native writer is still generating."""
    if runtime.input is not None and not any(message.id == runtime.input.id for message in history):
        history.append(runtime.input.model_copy(deep=True))
    if not runtime.chunks:
        return
    identifier = runtime.chunks[0].message_id
    message = next((item for item in history if item.id == identifier and item.role == "assistant"), None)
    if message is None:
        message = SessionMessage(id=identifier, role="assistant", content="", completed=False)
        history.append(message)
    message.turn_id = runtime.input.turn_id if runtime.input is not None else None
    message.completed = False
    message.chunks = [chunk.model_copy(deep=True) for chunk in runtime.chunks]
    blocks: dict[str | None, str] = {}
    for chunk in message.chunks:
        if chunk.chunk_type == ChunkEnum.CONTENT and isinstance(chunk.chunk, str):
            key = chunk.block_id
            blocks[key] = blocks.get(key, "") + chunk.chunk if chunk.event == "delta" else chunk.chunk
    message.content = "".join(blocks.values())
