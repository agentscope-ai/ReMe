"""Response schema for service endpoints and LLM calls."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Response(BaseModel):
    """Standard response envelope; extra fields allowed for endpoint-specific output.

    ``answer`` is the primary tool result and must be sufficient for the next LLM action.
    ``metadata`` is auxiliary request context for programmatic clients and diagnostics.
    """

    model_config = ConfigDict(extra="allow")

    answer: str | Any = Field(default="", description="Primary response content or result data exposed to tool callers")
    success: bool = Field(default=True, description="Whether the operation succeeded")
    metadata: dict = Field(default_factory=dict, description="Auxiliary request context and diagnostics")
