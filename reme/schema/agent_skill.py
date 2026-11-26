"""Backend-independent file references and generated agent skills."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillSpec(BaseModel):
    """Reference a named skill, or provide its description and Markdown body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        """Generated skills require both a description and a nonblank body."""
        if self.content is None:
            if self.description is not None:
                raise ValueError("description requires skill content")
        elif not self.content.strip() or not self.description or not self.description.strip():
            raise ValueError("generated skills require nonblank description and content")
        return self
