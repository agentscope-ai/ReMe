"""Base file parser interface."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileParser(BaseComponent):
    """Abstract base class for file parsers.

    Subclasses must implement `_parse` to extract file content into chunks.
    """

    component_type = ComponentEnum.FILE_PARSER

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.app_context is not None:
            self.working_dir = self.app_context.app_config.working_dir
        else:
            self.working_dir = str(Path.cwd())

    def _get_relative_path(self, path: str | Path) -> str:
        """Get path relative to working_dir.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            Path relative to working_dir.
        """
        file_path = Path(path).absolute()
        working_path = Path(self.working_dir).absolute()
        try:
            return str(file_path.relative_to(working_path))
        except ValueError:
            return str(file_path)

    async def parse(self, path: str | Path, cache_chunks: list[FileChunk] | None = None) -> tuple[
        FileNode, list[FileChunk]]:
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
    async def _parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Parse a file into metadata and chunks. Subclasses must implement this.

        Args:
            path: Path to the file to parse.

        Returns:
            Tuple of file metadata and parsed chunks.
        """
