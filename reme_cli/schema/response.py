"""Response schema module.

This module defines the standardized data structure for model output
responses used throughout the application.
"""

from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class Response(BaseModel):
    """Represents a structured response with result, status, and metadata.

    This model provides a consistent interface for returning results
    from operations, LLM calls, and service endpoints.

    Attributes:
        answer: The main response content, typically a string or structured data.
        success: Whether the operation completed successfully.
        metadata: Additional context and diagnostic information.
    """

    model_config = ConfigDict(extra="allow")

    answer: str | Any = Field(default="", description="Response content or result data")
    success: bool = Field(default=True, description="Operation success status")
    metadata: dict = Field(default_factory=dict, description="Additional context and diagnostics")
