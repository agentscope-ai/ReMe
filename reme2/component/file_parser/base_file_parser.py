"""Abstract base class for file parsers."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileMetadata


class BaseFileParser(BaseComponent):
    """Abstract base class for file format parsers.

    Each parser declares which file suffixes it handles and implements
    the parse method to produce FileMetadata and FileChunks.
    """

    component_type = ComponentEnum.FILE_PARSER

    suffixes: list[str] = []

    def __init__(self, chunk_tokens: int = 400, chunk_overlap: int = 80, **kwargs):
        super().__init__(**kwargs)
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap

    async def _start(self, app_context=None):
        pass

    async def _close(self):
        pass

    @abstractmethod
    async def parse(self, path: str) -> tuple[FileMetadata, list[FileChunk]]:
        """Parse a file into metadata and chunks.

        Args:
            path: Absolute path to the file.

        Returns:
            Tuple of (FileMetadata, list of FileChunks).
        """
