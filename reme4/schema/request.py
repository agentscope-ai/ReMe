"""Request schema for service endpoints."""

from pydantic import BaseModel, ConfigDict, Field


class Request(BaseModel):
    """Incoming service request; extra fields are allowed for endpoint-specific payloads."""

    model_config = ConfigDict(extra="allow")

    metadata: dict = Field(default_factory=dict, description="Request metadata for context")
