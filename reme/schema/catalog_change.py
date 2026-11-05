"""A catalog change carries both source and committed versions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class CatalogChange(BaseModel):
    """Round-trip this record from catalog_diff to catalog_commit after processing it."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    path: str
    change: Literal["added", "modified", "deleted"]
    st_mtime: float | None
    previous_mtime: float | None

    @model_validator(mode="after")
    def validate_versions(self):
        """Require the versions needed to confirm this change safely."""
        if (self.st_mtime is None) != (self.change == "deleted"):
            raise ValueError("st_mtime must be null only for deleted changes")
        if (self.previous_mtime is None) != (self.change == "added"):
            raise ValueError("previous_mtime must be null only for added changes")
        return self
