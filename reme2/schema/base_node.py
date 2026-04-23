from uuid import uuid4

from pydantic import BaseModel, Field


class BaseNode(BaseModel):
    """Base node model for graph and document structures."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    text: str = Field(default="")
    embedding: list[float] | None = Field(default=None)
    metadata: dict = Field(default_factory=dict)
