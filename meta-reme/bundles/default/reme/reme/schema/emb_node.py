"""Embedding node — base record carrying text and its vector."""

from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class EmbNode(BaseModel):
    """A text record with an optional embedding vector and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: uuid4().hex, description="Unique node id")
    text: str = Field(default="", description="Text content")
    embedding: np.ndarray | None = Field(default=None, description="Embedding vector (float16)")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata")

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, v):
        """Coerce list/tuple to float16 ndarray."""
        # Coerce list/tuple inputs into a float16 ndarray for compact storage.
        if v is None:
            return v
        return np.array(v, dtype=np.float16)

    @field_serializer("embedding")
    def serialize_embedding(self, v: np.ndarray | None, _info):
        """Serialize ndarray to a JSON-friendly list."""
        # ndarray is not JSON-serializable; emit a plain list.
        if v is None:
            return None
        return v.tolist()
