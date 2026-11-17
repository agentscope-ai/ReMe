"""Bounded, reconstructible live output until the native transcript catches up."""

import asyncio
from contextlib import contextmanager

from ....enumeration import ChunkEnum
from ....schema import StreamChunk


class LiveTurn:
    """Keep one current block snapshot rather than an ever-growing token event log."""

    def __init__(self, session_id, query):
        self.session_id = session_id
        self.query = query
        self.turn_id = None
        self.blocks = {}
        self.completed = False
        self.version = 0

    def update(self, chunk):
        """Snapshots replace their own block; deltas append without changing its position."""
        self.turn_id = chunk.turn_id or self.turn_id
        if chunk.chunk_type in {ChunkEnum.DONE, ChunkEnum.USAGE, ChunkEnum.REPLY_START}:
            return
        if chunk.metadata.get("visibility") == "diagnostic" or (
            chunk.chunk_type == ChunkEnum.THINK and chunk.metadata.get("kind") not in {"reasoning_summary", "plan"}
        ):
            return
        key = (
            chunk.chunk_type,
            chunk.tool_call_id or chunk.block_id or chunk.message_id,
        )
        previous = self.blocks.get(key)
        completed_tool = self.blocks.get((ChunkEnum.TOOL_RESULT, key[1]))
        terminal = previous
        if chunk.chunk_type == ChunkEnum.TOOL_CALL and completed_tool is not None:
            terminal = completed_tool
        if chunk.event != "end" and (self.completed or (terminal is not None and terminal.event == "end")):
            return
        self.version += 1
        value = chunk.model_copy(deep=True)
        if previous is not None:
            value.metadata = {**previous.metadata, **value.metadata}
            if value.chunk is None:
                value.chunk = previous.chunk
            elif value.event == "delta" and isinstance(value.chunk, str) and isinstance(previous.chunk, str):
                value.chunk = previous.chunk + value.chunk
            value.tool_call_name = value.tool_call_name or previous.tool_call_name
            value.media_type = value.media_type or previous.media_type
        if chunk.chunk_type == ChunkEnum.REPLY_END:
            self.completed = True
            outcome = chunk.chunk if isinstance(chunk.chunk, dict) else {}
            value.chunk = {key: outcome[key] for key in ("result", "status", "error") if key in outcome}
        value.event = "end" if chunk.event == "end" else "start"
        self.blocks[key] = value

    def messages(self):
        """Use stable turn identities shared with native-history projection."""
        chunks = [chunk.model_dump(mode="json") for chunk in self.blocks.values()]
        return [
            {
                "id": f"{self.turn_id}:user",
                "turn_id": self.turn_id,
                "role": "user",
                "content": self.query,
                "source": "chat",
                "completed": True,
                "delivery": "idle",
                "notification_eligible": False,
                "chunks": [],
            },
            {
                "id": f"{self.turn_id}:assistant",
                "turn_id": self.turn_id,
                "role": "assistant",
                "content": "\n\n".join(
                    chunk.chunk
                    for chunk in self.blocks.values()
                    if chunk.chunk_type == ChunkEnum.CONTENT and isinstance(chunk.chunk, str)
                ),
                "source": "chat",
                "completed": self.completed,
                "delivery": "idle",
                "notification_eligible": False,
                "chunks": chunks,
            },
        ]


class LiveTurns:
    """Runtime-owned transient state; subscribers never own or prolong generation."""

    def __init__(self):
        self.turns = {}
        self.listeners = set()
        self.epoch = 0

    def notify(self):
        """Wake every subscriber, coalescing pressure into one replaceable snapshot."""
        for listener in self.listeners:
            listener.set()

    @contextmanager
    def observe(self, agent_id, session_id, query):
        """A later formal turn replaces the previous cache; native files remain authoritative."""
        turn = LiveTurn(session_id, query)
        self.turns[agent_id] = turn

        def publish(chunk):
            turn.update(chunk)
            self.notify()

        try:
            yield publish
        except BaseException as exc:
            publish(
                StreamChunk(
                    chunk_type=ChunkEnum.REPLY_END,
                    event="end",
                    turn_id=turn.turn_id,
                    status=("interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"),
                    chunk={"error": str(exc)},
                ),
            )
            raise
        finally:
            self.notify()

    def overlay(self, agent_id, value):
        """Recover unfinished output on reconnect without replaying token deltas."""
        turn = self.turns.get(agent_id)
        if not turn or not turn.turn_id or turn.session_id != value["session_id"]:
            return value
        snapshot = value["snapshot"]
        messages = (snapshot or {}).get("messages", [])
        native = [
            message
            for message in messages
            if message.get("turn_id") == turn.turn_id and message.get("role") == "assistant"
        ]
        displayed_types = {
            ChunkEnum.CONTENT,
            ChunkEnum.THINK,
            ChunkEnum.DATA,
            ChunkEnum.TOOL_CALL,
            ChunkEnum.TOOL_RESULT,
        }
        expected = {
            (chunk.chunk_type, chunk.tool_call_id or chunk.block_id or chunk.message_id)
            for chunk in turn.blocks.values()
            if chunk.chunk_type in displayed_types and chunk.chunk is not None
        }
        observed = {
            (chunk.get("chunk_type"), chunk.get("tool_call_id") or chunk.get("block_id") or chunk.get("message_id"))
            for message in native
            for chunk in message.get("chunks", [])
        }
        # Providers may restore a tool as one completed result instead of a separate call/result pair.
        observed |= {
            (ChunkEnum.TOOL_CALL, identifier) for kind, identifier in observed if kind == ChunkEnum.TOOL_RESULT
        }
        result = next(
            (chunk.chunk.get("result") for chunk in turn.blocks.values() if chunk.chunk_type == ChunkEnum.REPLY_END),
            None,
        )
        final_saved = not result or any(message.get("content", "").endswith(result) for message in native)
        if (
            turn.completed
            and native
            and all(message["completed"] for message in native)
            and final_saved
            and expected <= observed
        ):
            self.turns.pop(agent_id, None)
            self.epoch += 1
            self.notify()
            return value
        retained = [message for message in messages if message.get("turn_id") != turn.turn_id]
        if agent_id != "chat":
            retained = []
        return {
            **value,
            "snapshot": {
                **(snapshot or {}),
                "session_id": turn.session_id,
                "state": {
                    "session_id": turn.session_id,
                    "status": value["agent"]["status"],
                    "active_flags": [],
                },
                "messages": [*retained, *turn.messages()],
                "last_turn_id": turn.turn_id,
                "revision": f"{(snapshot or {}).get('revision', '')}:{turn.turn_id}:{turn.version}",
            },
        }

    def clear(self):
        """Release caches and wake request-owned subscribers during application shutdown."""
        self.turns.clear()
        self.epoch += 1
        self.notify()
