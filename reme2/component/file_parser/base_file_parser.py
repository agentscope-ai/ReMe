"""Base file parser interface."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileParser(BaseComponent):
    """Abstract base class for file parsers.

    Subclasses must implement `_parse` to extract file content into chunks.
    """

    component_type = ComponentEnum.FILE_PARSER

    async def parse(self, path: str, cache_chunks: list[FileChunk] | None = None) -> tuple[FileNode, list[FileChunk]]:
        """Parse a file and optionally reuse cached chunks by hash.

        Args:
            path: Path to the file to parse.
            cache_chunks: Previously parsed chunks to reuse when hashes match.

        Returns:
            Tuple of file metadata and parsed chunks.
        """
        file_node, chunks = await self._parse(path)
        if cache_chunks:
            cache_chunk_dict = {chunk.hash: chunk for chunk in cache_chunks}
            for i in range(len(chunks)):
                chunk = chunks[i]
                if chunk.hash in cache_chunk_dict:
                    chunks[i] = cache_chunk_dict[chunk.hash]
        return file_node, chunks

    @abstractmethod
    async def _parse(self, path: str) -> tuple[FileNode, list[FileChunk]]:
        """Parse a file into metadata and chunks. Subclasses must implement this.

        Args:
            path: Path to the file to parse.

        Returns:
            Tuple of file metadata and parsed chunks.
        """
