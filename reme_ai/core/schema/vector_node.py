"""Vector node schema for representing nodes with embeddings."""

from typing import List
from uuid import uuid4

from pydantic import BaseModel, Field


class VectorNode(BaseModel):
    """Represents a node with content, vector embeddings, and metadata."""

    unique_id: str = Field(default_factory=lambda: uuid4().hex)
    workspace_id: str = Field(default="")
    content: str = Field(default="")
    vector: List[float] | None = Field(default=None)
    metadata: dict = Field(default_factory=dict)
