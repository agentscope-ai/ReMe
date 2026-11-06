"""Reduce portable stream events to small, deduplicated execution phases."""

from ..schema.agent_activity import AgentActivity
from ..schema.stream_chunk import StreamChunk


def reduce_activity(previous: AgentActivity, chunk: StreamChunk) -> AgentActivity:
    """Keep the latest real execution phase; transport DONE never completes a turn."""
    phase, tool = previous.phase, None
    kind = chunk.chunk_type
    if kind in {"reply_start", "think"}:
        phase = "thinking"
    elif kind == "content" and isinstance(chunk.chunk, str) and chunk.chunk:
        phase = "answering"
    elif kind == "tool_call" or (kind == "tool_result" and chunk.status == "running"):
        tool = chunk.tool_call_name or previous.tool
        phase = (
            "remembering"
            if tool and any(word in tool.lower() for word in ("memory", "conversation", "recall"))
            else "working"
        )
    elif kind == "tool_result":
        phase = "thinking"
    elif kind == "approval" and chunk.status == "waiting":
        phase = "waiting"
    elif kind == "reply_end":
        phase = {"completed": "completed", "failed": "failed", "interrupted": "interrupted", "waiting": "waiting"}.get(
            chunk.status,
            previous.phase,
        )
    elif kind == "error" and chunk.metadata.get("will_retry") is not True:
        phase = "failed"
    else:
        return previous
    return previous.model_copy(update={"phase": phase, "tool": tool})
