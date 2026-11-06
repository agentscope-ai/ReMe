"""Portable manifests for connector-produced resource packages."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class ResourceFile(BaseModel):
    """A package-relative payload path, with optional connector-specific fields."""

    model_config = ConfigDict(extra="allow")

    path: str
    description: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Keep payload references inside the package and reserve its manifest name."""
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
                path.name.casefold() == "metadata.json",
            ),
        ):
            raise ValueError("path must be a package-relative file path other than metadata.json")
        return value


class ResourceMetadata(BaseModel):
    """Describe one complete resource package; processing state belongs to the catalog."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataclass: str = Field(min_length=1)
    source: str = Field(min_length=1)
    created_at: AwareDatetime
    modified_at: AwareDatetime
    description: str
    files: list[ResourceFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_package(self):
        """Require distinct payloads and chronological timestamps."""
        paths = [item.path.casefold() for item in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("files must contain distinct paths")
        if self.modified_at < self.created_at:
            raise ValueError("modified_at must not precede created_at")
        return self
