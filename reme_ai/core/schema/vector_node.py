from typing import List
from uuid import uuid4

from pydantic import BaseModel, Field


class VectorNode(BaseModel):
    unique_id: str = Field(default_factory=lambda: uuid4().hex)
    content: str = Field(default="")
    vector: List[float] | None = Field(default=None)
    metadata: dict = Field(default_factory=dict)
