from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_serializer, field_validator, ConfigDict


class EmbNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    text: str = Field(default="")
    embedding: np.ndarray | None = Field(default=None)
    metadata: dict = Field(default_factory=dict)

    @field_validator('embedding', mode='before')
    @classmethod
    def validate_embedding(cls, v):
        if v is None:
            return v
        return np.array(v, dtype=np.float16)

    @field_serializer('embedding')
    def serialize_embedding(self, v: np.ndarray | None, _info):
        if v is None:
            return None
        return v.tolist()
