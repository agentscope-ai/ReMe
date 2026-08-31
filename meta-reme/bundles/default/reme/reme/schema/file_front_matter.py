"""FileFrontMatter — parsed Markdown front matter."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FileFrontMatter(BaseModel):
    """Markdown front matter; unknown keys are preserved as extras."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="", description="Document name")
    description: str = Field(default="", description="Document description")

    @property
    def model_extra(self) -> dict[str, Any] | None:
        """Get extra fields set during validation.

        Returns:
            A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.
        """
        return self.__pydantic_extra__
