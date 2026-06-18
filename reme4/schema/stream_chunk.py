"""Stream chunk schema for incremental responses (e.g. LLM streaming)."""

from typing import Any

from pydantic import BaseModel, Field

from ..enumeration import ChunkEnum


class StreamChunk(BaseModel):
    """A single chunk in a unified streaming response sequence.

    Carries full information from both AgentScope and Claude Code SDK
    backends.  Optional fields are ``None`` by default so that simple
    text-only streams (e.g. plain CONTENT deltas) stay lightweight.

    Fields:
        chunk_type:   Category of this chunk (see ChunkEnum).
        chunk:        Payload, typically a text delta, but can be
                      dict/list for structured data (tool-call JSON,
                      usage stats, etc.).
        done:         Terminal marker. True only for the final DONE chunk.
        session_id:   Session identifier (AgentScope: agent.state.session_id;
                      Claude Code: ResultMessage.session_id).
        block_id:     Content-block identifier for matching start/delta/end
                      sequences (both backends assign block IDs).
        tool_call_id: Tool-call identifier for correlating call deltas
                      with their start event and result.
        tool_call_name: Name of the tool being invoked.
        media_type:   MIME type for DATA blocks (e.g. ``"image/png"``).
        input_tokens:  Prompt tokens consumed (populated on USAGE chunks).
        output_tokens: Completion tokens generated (populated on USAGE chunks).
        metadata:     Backend-specific extras that don't warrant a dedicated
                      field.
    """

    chunk_type: ChunkEnum = Field(default=ChunkEnum.CONTENT, description="Type of chunk content")
    chunk: str | dict | list = Field(default="", description="Chunk payload")
    done: bool = Field(default=False, description="Whether this is the final chunk")

    session_id: str | None = Field(default=None, description="Session identifier")
    block_id: str | None = Field(default=None, description="Content block identifier")
    tool_call_id: str | None = Field(default=None, description="Tool call identifier")
    tool_call_name: str | None = Field(default=None, description="Tool call name")
    media_type: str | None = Field(default=None, description="MIME type for data blocks")
    input_tokens: int | None = Field(default=None, description="Prompt tokens consumed")
    output_tokens: int | None = Field(default=None, description="Completion tokens generated")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
