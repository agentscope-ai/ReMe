"""Abstract base for file chunkers."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileChunker(BaseComponent):
    """Abstract base for file chunkers. Subclasses implement `parse`."""

    component_type = ComponentEnum.FILE_CHUNKER

    def __init__(self, supported_extensions: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions: list[str] = supported_extensions or []

    @abstractmethod
    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Parse a file into (node, chunks)."""
