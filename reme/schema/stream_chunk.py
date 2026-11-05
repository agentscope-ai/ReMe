"""Stream chunk schema for incremental responses (e.g. LLM streaming)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ..enumeration import ChunkEnum

StreamStatus = Literal["running", "completed", "failed", "interrupted", "waiting"]


def stream_status(value: Any) -> StreamStatus | None:
    """Normalize provider execution states without treating unknown states as success."""
    return {
        "inProgress": "running",
        "running": "running",
        "started": "running",
        "completed": "completed",
        "success": "completed",
        "succeeded": "completed",
        "error": "failed",
        "failed": "failed",
        "declined": "failed",
        "denied": "failed",
        "exceed_max_iters": "failed",
        "interrupted": "interrupted",
        "cancelled": "interrupted",
        "waiting": "waiting",
    }.get(getattr(value, "value", value))


class StreamChunk(BaseModel):
    """Backend-independent display event, with provider extras in metadata.

    ``start`` creates a block, ``delta`` appends content, and ``end`` settles it.
    A non-null start/end payload is a complete snapshot replacing accumulated
    content; null means no content update (an empty string is a valid snapshot).
    Simple text producers use the default ``delta`` operation.

    Reply markers delimit one agent invocation, including its tool/model loop.
    DONE closes the transport and does not imply success. IDs correlate a
    session, invocation (turn), model message, content block, and tool call.
    ``reply_end.chunk`` carries the complete outcome (excluding its event trace).
    ``metadata.native`` preserves complete provider events; ``metadata.kind``
    distinguishes plans, reasoning summaries, and other display subtypes.
    """

    model_config = ConfigDict(extra="allow")

    chunk_type: ChunkEnum = Field(default=ChunkEnum.CONTENT, description="Type of chunk content")
    event: Literal["start", "delta", "end"] = Field(default="delta", description="Block update operation")
    chunk: JsonValue = Field(default="", description="Delta or complete start/end snapshot")
    done: bool = Field(default=False, description="Whether this is the final chunk")
    status: StreamStatus | None = Field(default=None, description="Normalized execution state")

    session_id: str | None = Field(default=None, description="Session identifier")
    turn_id: str | None = Field(default=None, description="Agent invocation identifier")
    message_id: str | None = Field(default=None, description="Model message identifier")
    block_id: str | None = Field(default=None, description="Content block identifier")
    tool_call_id: str | None = Field(default=None, description="Tool call identifier")
    tool_call_name: str | None = Field(default=None, description="Tool call name")
    media_type: str | None = Field(default=None, description="MIME type for data blocks")
    input_tokens: int | None = Field(default=None, description="Prompt tokens consumed")
    output_tokens: int | None = Field(default=None, description="Completion tokens generated")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
