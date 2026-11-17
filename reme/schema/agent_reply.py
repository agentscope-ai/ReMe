"""Agent outcomes and lossless collection/replay of their event stream."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from copy import deepcopy
from typing import Any, Literal, TypedDict

from .stream_chunk import StreamChunk
from ..enumeration import ChunkEnum

AgentReplyStatus = Literal["completed", "failed", "interrupted", "waiting"]


class AgentReply(TypedDict):
    """One invocation's outcome and complete, ordered JSON event trace.

    ``result`` is final reply text; ``usage`` covers this invocation only.
    ``raw`` contains JSON-serializable native results, including provider
    usage details and termination reasons omitted from the portable fields.
    """

    session_id: str
    result: str
    structured_output: Any
    usage: dict[str, int] | None
    status: AgentReplyStatus
    error: str | None
    raw: dict[str, Any]
    events: list[dict[str, Any]]


async def collect_reply(
    stream: AsyncGenerator[StreamChunk, None],
    on_chunk: Callable[[StreamChunk], Awaitable[Any]] | None = None,
) -> AgentReply:
    """Collect an invocation without deriving its outcome from display text.

    The terminal reply_end carries the same outcome fields as AgentReply.
    Missing/duplicate outcomes fail explicitly; DONE is only a transport marker.
    """
    events = []
    outcome = None
    async with aclosing(stream):
        async for chunk in stream:
            events.append(chunk.model_dump(mode="json"))
            if on_chunk is not None:
                await on_chunk(chunk)
            if chunk.chunk_type == ChunkEnum.REPLY_END:
                required = AgentReply.__required_keys__ - {"events"}
                if outcome is not None or not isinstance(chunk.chunk, dict) or not required <= chunk.chunk.keys():
                    raise ValueError("Expected exactly one complete reply_end outcome")
                outcome = chunk.chunk
    if outcome is None:
        raise ValueError("Agent stream ended without a reply_end outcome")
    return AgentReply(**{**outcome, "events": events})


async def replay_reply(reply: AgentReply) -> AsyncGenerator[StreamChunk, None]:
    """Replay fresh chunks in their original order, including native metadata."""
    for event in reply["events"]:
        yield StreamChunk.model_validate(deepcopy(event))
