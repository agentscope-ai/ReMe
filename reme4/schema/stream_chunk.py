"""Stream chunk schema for incremental responses (e.g. LLM streaming)."""

from pydantic import BaseModel, Field

from ..enumeration import ChunkEnum


class StreamChunk(BaseModel):
    """A single chunk in a streaming response sequence."""

    chunk_type: ChunkEnum = Field(default=ChunkEnum.CONTENT, description="Type of chunk content")
    chunk: str | dict | list = Field(default="", description="Chunk payload")
    done: bool = Field(default=False, description="Whether this is the final chunk")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata")
