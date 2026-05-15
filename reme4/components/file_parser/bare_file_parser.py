from pathlib import Path

from .base_file_parser import BaseFileParser
from ..component_registry import R
from ...schema import FileChunk, FileNode


@R.register("bare")
class BareFileParser(BaseFileParser):
    """Stat-only parser for attachment/binary files.

    No content read, no chunking, no link extraction. The resulting FileNode
    has empty links and chunk_ids; front_matter carries mime and size so
    retrieval can filter by file type without reopening the file.
    """

    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        file_path = Path(path)
        stat = file_path.stat()
        return FileNode(path=self._get_relative_path(path), st_mtime=stat.st_mtime, links=[], chunk_ids=[]), []
