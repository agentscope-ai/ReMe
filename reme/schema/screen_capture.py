"""Screenshot provenance retained with images and successful task summaries."""

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from .connector_result import ConnectorResult


class CapturedDisplay(BaseModel):
    """Identity and image-local exclusions; display ordering is never an identity."""

    model_config = ConfigDict(extra="forbid")

    display_id: str = Field(min_length=1, max_length=1024)
    stable_id: bool
    bounds: tuple[int, int, int, int]
    pixel_size: tuple[int, int]
    rotation: float = 0
    ignored_regions: list[tuple[int, int, int, int]] = Field(default_factory=list, max_length=64)
    path: str
    scope: str = "manual"
    captured_at: AwareDatetime

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Apply the connector's workspace-relative path contract."""
        ConnectorResult(context="", paths=[value])
        return value

    @field_validator("pixel_size")
    @classmethod
    def validate_size(cls, value):
        """Keep decoded images and comparison work bounded."""
        if any(size <= 0 or size > 32768 for size in value) or value[0] * value[1] > 40_000_000:
            raise ValueError("Invalid screenshot dimensions")
        return value
