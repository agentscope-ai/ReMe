from typing import List

from pydantic import Field, BaseModel, ConfigDict

from .message import Message


class Request(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = Field(default="")
    messages: List[Message] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
