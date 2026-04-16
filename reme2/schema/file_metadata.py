"""File metadata schema module.

This module defines the FileMetadata model for tracking file state and
content information in the document processing pipeline.
"""

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """File metadata with optional extended fields.

    Stores essential file information for tracking changes and
    managing the document processing pipeline. Optional fields allow
    for different usage patterns (e.g., just tracking vs. full content).

    Attributes:
        hash: Hash of the file content for change detection.
        mtime_ms: Last modification time in milliseconds since epoch.
        size: File size in bytes.
        path: Relative path to the file within the workspace.
        content: Parsed content from the file (optional, memory-intensive).
        chunk_count: Number of chunks extracted from this file.
        metadata: Additional file-specific metadata.
    """

    hash: str = Field(..., description="Hash of file content for change detection")
    mtime_ms: float = Field(..., description="Last modification time in milliseconds")
    size: int = Field(..., description="File size in bytes")
    path: str | None = Field(default=None, description="Relative path within workspace")
    content: str | None = Field(default=None, description="Parsed content (optional)")
    chunk_count: int | None = Field(default=None, description="Number of extracted chunks")
    metadata: dict = Field(default_factory=dict, description="Additional file metadata")
