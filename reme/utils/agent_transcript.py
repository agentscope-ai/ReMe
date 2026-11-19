"""One projection of provider events for live replies and restored client history."""

import json
import re

from ..schema.agent_message import AgentMessage
from ..schema.stream_chunk import StreamChunk

_SETTLED = {"completed", "failed", "interrupted", "unknown"}
_ACTIONS = (
    (r"\b(write|edit|patch)\b|file change", "edit"),
    (r"\b(view image|image view)\b", "image"),
    (r"web extract|fetch|browse|open page", "browse"),
    (r"web search|search web", "webSearch"),
    (r"\b(search|find|grep)\b", "search"),
    (r"\blist\b", "list"),
    (r"\b(read|load)(?:$| (?:files?|directories|directory|dir|paths?)\b)", "read"),
    (r"\b(command|exec|bash|shell|terminal)\b", "command"),
)


def _record(value):
    return value if isinstance(value, dict) else {}


def _text(value):
    return value if isinstance(value, str) and value.strip() else None


def _printable(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def tool_actions(name, value):
    """Interpret native tool semantics once, before exposing provider-independent blocks."""
    payload = _record(value)
    if payload.get("commandActions"):
        result = []
        for action in payload["commandActions"]:
            action = _record(action)
            kind = {"read": "read", "listFiles": "list", "search": "search"}.get(action.get("type"), "command")
            result.append(
                {
                    "kind": kind,
                    "target": _text(action.get("path"))
                    or _text(action.get("command" if kind == "command" else "name")),
                    "query": _text(action.get("query")),
                },
            )
        return result
    if payload.get("type") == "fileChange" and payload.get("changes"):
        return [{"kind": "edit", "target": _text(_record(change).get("path"))} for change in payload["changes"]]
    if payload.get("type") == "webSearch":
        action = _record(payload.get("action"))
        if action.get("type") == "search":
            return [
                {"kind": "webSearch", "query": _text(query)} for query in action.get("queries") or [action.get("query")]
            ]
        if action.get("type") == "openPage":
            return [{"kind": "browse", "target": _text(action.get("url"))}]
        if action.get("type") == "findInPage":
            return [{"kind": "search", "target": _text(action.get("url")), "query": _text(action.get("pattern"))}]
    words = re.sub(r"[_./:]+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", name)).lower()
    kind = next((kind for pattern, kind in _ACTIONS if re.search(pattern, words)), "tool")
    arguments = payload.get("arguments", payload.get("input", payload))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    arguments = _record(arguments)
    target = (
        _text(arguments.get("command")) or _text(arguments.get("cmd")) or _text(value)
        if kind == "command"
        else _text(arguments.get("path")) or _text(arguments.get("file_path")) or _text(arguments.get("url"))
    )
    return [
        {
            "kind": kind,
            "target": name if kind == "tool" else target,
            "query": _text(arguments.get("query")) or _text(arguments.get("pattern")),
        },
    ]


class TranscriptProjector:
    """Ordered complete blocks; token deltas never cross the client boundary."""

    def __init__(self, session_id="", turn_id=None):
        self.session_id = session_id
        self.turn_id = turn_id
        self.items = {}
        self.outcome = None
        self.error = None

    def update(self, chunk: StreamChunk):
        """Merge one portable wrapper event, preserving null ends and settled tools."""
        self.session_id = chunk.session_id or self.session_id
        self.turn_id = chunk.turn_id or self.turn_id
        kind, payload, metadata = chunk.chunk_type, _record(chunk.chunk), chunk.metadata
        identifier = chunk.tool_call_id or chunk.block_id or chunk.message_id or kind
        current = self.items.get(identifier)
        if current and current["status"] in _SETTLED and chunk.event != "end":
            return
        status = chunk.status or ("completed" if chunk.event == "end" else "running")
        item = None
        if kind in {"content", "think"}:
            if metadata.get("visibility") == "diagnostic" or (
                kind == "think" and metadata.get("kind") not in {"reasoning_summary", "plan"}
            ):
                return
            previous = current.get("text", "") if current else ""
            text = (
                previous
                if chunk.chunk is None
                else (previous + _printable(chunk.chunk) if chunk.event == "delta" else _printable(chunk.chunk))
            )
            item = {"id": identifier, "status": status, "text": text}
            if kind == "content":
                phase = metadata.get("phase", _record(metadata.get("native")).get("phase"))
                item.update(
                    type="text",
                    phase=phase if phase in {"commentary", "final_answer"} else (current or {}).get("phase"),
                )
            else:
                item.update(type="summary", kind="plan" if metadata.get("kind") == "plan" else "reasoning")
        elif kind in {"tool_call", "tool_result"}:
            item = (
                dict(current)
                if current and current["type"] == "tool"
                else {
                    "id": identifier,
                    "type": "tool",
                    "name": "tool",
                    "input": "",
                    "output": "",
                    "actions": [],
                    "status": "running",
                }
            )
            result = kind == "tool_result"
            native_type = payload.get("type")
            name = chunk.tool_call_name or _text(payload.get("tool")) or _text(payload.get("name")) or item["name"]
            inputs = (
                "\n\n".join(
                    "\n".join(str(change[key]) for key in ("path", "diff") if key in change)
                    for change in payload["changes"]
                )
                if native_type == "fileChange" and isinstance(payload.get("changes"), list)
                else payload.get("arguments", payload.get("input", payload.get("command")))
            )
            if native_type in {"commandExecution", "fileChange"}:
                output = payload.get("aggregatedOutput", payload.get("error", ""))
            elif native_type == "mcpToolCall":
                output = payload.get("error") or payload.get("result", "")
            else:
                output = payload.get("result", payload.get("content", chunk.chunk))
            content = output if result else inputs if inputs is not None else chunk.chunk
            field = "output" if result else "input"
            if inputs is not None:
                item["input"] = _printable(inputs)
            if (
                not result
                and chunk.event != "delta"
                or not item["actions"]
                or native_type in {"commandExecution", "fileChange"}
                or "commandActions" in payload
            ):
                item["actions"] = tool_actions(name, chunk.chunk)
            exit_code = payload.get("exitCode", payload.get("exit_code"))
            failed = (
                payload.get("is_error") is True
                or payload.get("success") is False
                or bool(payload.get("error"))
                or (type(exit_code) in (int, float) and exit_code != 0)
            )
            item["status"] = (
                "failed"
                if failed
                else (
                    item["status"]
                    if item["status"] in _SETTLED and (not result or chunk.event == "delta")
                    else status if result else "running"
                )
            )
            item["name"] = name
            if chunk.chunk is not None:
                item[field] = (
                    item[field] + content
                    if chunk.event == "delta" and isinstance(content, str)
                    else _printable(content)
                )
        elif (
            kind == "data"
            and metadata.get("visibility") != "diagnostic"
            and (chunk.media_type or payload.get("url") or payload.get("type") == "imageView")
        ):
            url = payload.get("url") or payload.get("path") or (chunk.chunk if isinstance(chunk.chunk, str) else "")
            if (
                url
                and chunk.media_type
                and chunk.media_type.startswith("image/")
                and not re.match(r"^(data:|https?:|file:|/)", url)
            ):
                url = f"data:{chunk.media_type};base64,{url}"
            if url:
                item = {
                    "id": identifier,
                    "type": "artifact",
                    "status": status,
                    "url": url,
                    "media_type": chunk.media_type or "application/octet-stream",
                    "label": payload.get("name") or ("Image" if url.startswith("data:") else url),
                }
        elif kind == "reply_end":
            self.outcome = chunk.status or payload.get("status")
            self.error = _text(payload.get("error")) or self.error
            final = next(
                (
                    value
                    for value in reversed(self.items.values())
                    if value["type"] == "text" and value.get("phase") == "final_answer"
                ),
                None,
            )
            final = final or next((value for value in reversed(self.items.values()) if value["type"] == "text"), None)
            if isinstance(payload.get("result"), str) and (payload["result"] or not final):
                key = final["id"] if final else f"{self.turn_id or 'reply'}:final"
                self.items[key] = {
                    "id": key,
                    "type": "text",
                    "text": payload["result"],
                    "phase": "final_answer",
                    "status": "completed",
                }
        elif kind == "error" and metadata.get("will_retry") is not True:
            self.error = chunk.chunk if isinstance(chunk.chunk, str) else "Reply failed"
        if item is not None:
            self.items[identifier] = item

    def message(self, *, completed=None, **fields) -> AgentMessage:
        """Snapshot without modifying the projector's mutable live state."""
        completed = self.outcome is not None if completed is None else completed
        items = []
        for value in self.items.values():
            value = dict(value)
            if completed and value["status"] not in _SETTLED:
                outcome = self.outcome or "completed"
                value["status"] = "unknown" if outcome == "completed" and value["type"] == "tool" else outcome
            items.append(value)
        content = "\n\n".join(item["text"] for item in items if item["type"] == "text")
        return AgentMessage(
            **{
                "id": f"{self.turn_id or 'reply'}:assistant",
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "role": "assistant",
                "content": content,
                "completed": completed,
                "items": items,
                "outcome": self.outcome,
                "error": self.error,
                **fields,
            },
        )


def project_messages(session_id, messages, *, notifications=False):
    """Group one assistant turn independently of provider-specific model-message boundaries."""
    result, turns = [], {}
    for message in messages:
        value = message.model_dump() if hasattr(message, "model_dump") else message
        fields = {key: value[key] for key in AgentMessage.model_fields if key in value}
        turn = value.get("turn_id")
        grouped = value["role"] == "assistant" and value.get("source", "chat") == "chat" and turn
        if grouped and turn in turns:
            projector, index = turns[turn]
            previous = result[index]
        else:
            projector, index, previous = TranscriptProjector(session_id, turn), len(result), None
            if grouped:
                turns[turn] = projector, index
        for chunk in value.get("chunks", []):
            projector.update(chunk if isinstance(chunk, StreamChunk) else StreamChunk.model_validate(chunk))
        content = "\n\n".join(filter(None, [previous.content if previous else "", value["content"]]))
        fields.pop("content", None)
        fields.update(
            id=f"{turn}:assistant" if grouped else value["id"],
            completed=value.get("completed", True) and (previous.completed if previous else True),
            notification_eligible=bool(
                notifications
                and value.get("source") == "proactive"
                and value["role"] == "assistant"
                and value.get("completed", True)
                and value["content"].strip(),
            ),
        )
        projected = projector.message(**fields)
        if not any(item.type == "text" for item in projected.items):
            projected.content = content
        if previous is not None:
            projected.created_at = previous.created_at
            result[index] = projected
        else:
            result.append(projected)
    return result
