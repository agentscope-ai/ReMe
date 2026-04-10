"""Defines the data structure for individual data packets in a streaming response."""

from pydantic import Field, BaseModel

from ..enumeration import ChunkEnum


class StreamChunk(BaseModel):
    """Represents a single chunk of streamed data including its type, content, and completion status."""

    chunk_type: ChunkEnum = Field(default=ChunkEnum.ANSWER)
    chunk: str | dict | list = Field(default="")
    done: bool = Field(default=False)
    metadata: dict = Field(default_factory=dict)
