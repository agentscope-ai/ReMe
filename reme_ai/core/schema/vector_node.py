from typing import List, Dict
from uuid import uuid4

from pydantic import BaseModel, Field


class VectorNode(BaseModel):
    vector_id: str = Field(default_factory=lambda: uuid4().hex)
    content: str = Field(default="")
    vector: List[float] | None = Field(default=None)
    metadata: Dict[str, str | bool | int | float] = Field(default_factory=dict)
