"""File chunk schema module.

This module defines the FileChunk model for representing chunks of file content
in the document processing and retrieval pipeline.
"""

from pydantic import Field

from .base_node import BaseNode


class FileChunk(BaseNode):
    """A chunk of file content with positional and scoring metadata.

    Represents a contiguous section of a file that has been extracted
    for processing, embedding, or retrieval. Inherits text and embedding
    capabilities from BaseNode.

    Attributes:
        path: File path relative to workspace root.
        start_line: Starting line number (1-indexed) in the source file.
        end_line: Ending line number (1-indexed) in the source file.
        hash: Hash of the chunk content for deduplication.
        scores: Search relevance scores indexed by score type.

    Properties:
        score: Final combined score for search result ranking.
        merge_key: Unique key for merging duplicate search results.
    """

    path: str = Field(..., description="File path relative to workspace")
    start_line: int = Field(..., description="Starting line number (1-indexed)")
    end_line: int = Field(..., description="Ending line number (1-indexed)")
    hash: str = Field(..., description="Hash of chunk content for deduplication")
    scores: dict[str, float] = Field(default_factory=dict, description="Search scores by type")

    @property
    def score(self) -> float:
        """Get the final score for search result ranking.

        Returns:
            The combined score, or 0.0 if not set.
        """
        return self.scores.get("score", 0.0)

    @property
    def merge_key(self) -> str:
        """Generate a unique key for merging search results.

        Returns:
            A string key in format "path:start_line:end_line".
        """
        return f"{self.path}:{self.start_line}:{self.end_line}"
