"""Abstract base class for file parsers."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum, FileSuffixEnum
from ...schema import FileChunk, FileMetadata


class BaseFileParser(BaseComponent):
    """Parser that declares handled suffixes and produces FileChunks."""

    component_type = ComponentEnum.FILE_PARSER
    suffixes: list[FileSuffixEnum] = []

    @abstractmethod
    async def parse(self, path: str) -> tuple[FileMetadata, list[FileChunk]]:
        """Parse a file into metadata and chunks."""
