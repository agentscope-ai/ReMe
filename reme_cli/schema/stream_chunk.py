"""Stream chunk schema module.

This module defines the StreamChunk model for handling streaming
responses in the application, particularly for LLM outputs.
"""

from pydantic import BaseModel, Field

from ..enumeration import ChunkEnum


class StreamChunk(BaseModel):
    """A chunk of streaming response data.

    Represents a single chunk in a streaming response sequence,
    commonly used for LLM outputs that are delivered incrementally.

    Attributes:
        chunk_type: Type identifier for the chunk content.
        chunk: The actual chunk data (string, dict, or list).
        done: Whether this is the final chunk in the stream.
        metadata: Additional metadata about this chunk.
    """

    chunk_type: ChunkEnum = Field(default=ChunkEnum.CONTENT, description="Type of chunk content")
    chunk: str | dict | list = Field(default="", description="Chunk payload data")
    done: bool = Field(default=False, description="Whether stream is complete")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata")
