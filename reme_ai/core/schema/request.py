"""Defines the data structure for processing incoming user requests and message history."""

from pydantic import Field, BaseModel, ConfigDict


class Request(BaseModel):
    """Represents a structured request payload containing a query, message list, and metadata."""

    model_config = ConfigDict(extra="allow")

    metadata: dict = Field(default_factory=dict)
