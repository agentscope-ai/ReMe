"""Text and progressively readable local resources returned by connectors."""

from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConnectorResult(BaseModel):
    """Result context and reading instructions, plus workspace-relative resource paths."""

    model_config = ConfigDict(extra="forbid")

    context: str
    paths: list[str] = Field(default_factory=list)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: list[str]) -> list[str]:
        """Reject ambiguous, duplicate, or escaping resource references."""
        for value in values:
            path = PurePosixPath(value)
            if any(
                (
                    not value,
                    value.startswith("~"),
                    "\\" in value,
                    PureWindowsPath(value).anchor,
                    path.is_absolute(),
                    ".." in path.parts,
                    path.as_posix() != value,
                    value == ".",
                ),
            ):
                raise ValueError("paths must contain workspace-relative file or directory paths")
        if len(set(values)) != len(values):
            raise ValueError("paths must be distinct")
        return values
