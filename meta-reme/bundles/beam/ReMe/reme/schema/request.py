"""Request schema for service endpoints."""

from pydantic import BaseModel, ConfigDict, Field


class Request(BaseModel):
    """Incoming service request; extra fields are allowed for endpoint-specific payloads."""

    model_config = ConfigDict(extra="allow")

    metadata: dict | None = Field(default=None, description="Request metadata for context")
