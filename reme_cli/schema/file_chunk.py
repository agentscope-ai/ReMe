"""File chunk schema."""

from pydantic import Field
from .base_node import BaseNode


class FileChunk(BaseNode):
    """A chunk of file content with metadata."""

    path: str = Field(..., description="File path relative to workspace")
    start_line: int = Field(..., description="Starting line number in the source file")
    end_line: int = Field(..., description="Ending line number in the source file")
    hash: str = Field(..., description="Hash of the chunk content")
    scores: dict[str, float] = Field(default_factory=dict, description="Search scores by type")

    @property
    def score(self) -> float:
        """Final score for search result."""
        return self.scores.get("score", 0.0)

    @property
    def merge_key(self) -> str:
        """Key for merging search results."""
        return f"{self.path}:{self.start_line}:{self.end_line}"