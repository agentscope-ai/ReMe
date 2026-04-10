"""Defines the standardized data structure for model output responses."""

from typing import Any

from pydantic import Field, BaseModel


class Response(BaseModel):
    """Represents a structured response containing the execution result, status, and metadata."""

    answer: str | Any = Field(default="")
    success: bool = Field(default=True)
    metadata: dict = Field(default_factory=dict)
