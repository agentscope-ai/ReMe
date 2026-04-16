"""Request schema module.

This module defines the Request model for handling incoming requests
in the application service layer.
"""

from pydantic import BaseModel, ConfigDict, Field


class Request(BaseModel):
    """Request model for service endpoints.

    Represents an incoming request with optional metadata and
    extensible fields for various request types.

    The model uses ConfigDict with extra="allow" to support
    dynamically added fields while maintaining type safety.

    Attributes:
        metadata: Request-level metadata for tracking and context.
    """

    model_config = ConfigDict(extra="allow")

    metadata: dict = Field(default_factory=dict, description="Request metadata for context")
