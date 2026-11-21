"""Request schema for service endpoints."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..constants import TRUSTED_SESSION_ID_KWARG


class Request(BaseModel):
    """Incoming service request; extra fields are allowed for endpoint-specific payloads."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def reject_internal_session_provenance(cls, value):
        """Prevent external HTTP callers from forging the internal session binding."""
        if isinstance(value, dict) and TRUSTED_SESSION_ID_KWARG in value:
            raise ValueError(f"{TRUSTED_SESSION_ID_KWARG} is reserved for internal job calls")
        return value

    metadata: dict | None = Field(default=None, description="Request metadata for context")
