from pydantic import Field, BaseModel

from ..enumeration import ChunkEnum


class StreamChunk(BaseModel):
    chunk_type: ChunkEnum = Field(default=ChunkEnum.ANSWER)
    chunk: str | dict | list = Field(default="")
    done: bool = Field(default=False)
    metadata: dict = Field(default_factory=dict)
