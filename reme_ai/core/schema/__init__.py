from typing import List

from flowllm.core.schema import Message, ToolCall
from flowllm.core.enumeration import Role
from pydantic import BaseModel, Field


class Trajectory(BaseModel):

    task_id: str = Field(default="")
    messages: List[Message] = Field(default_factory=list)
    score: float = Field(default=0.0)
    metadata: dict = Field(default_factory=dict)
