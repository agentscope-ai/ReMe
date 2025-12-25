from pydantic import Field, BaseModel

from ..enumeration import ChunkEnum


class FlowStreamChunk(BaseModel):
    flow_id: str = Field(default="")
    chunk_type: ChunkEnum = Field(default=ChunkEnum.ANSWER)
    chunk: str | dict | list = Field(default="")
    done: bool = Field(default=False)
    metadata: dict = Field(default_factory=dict)
