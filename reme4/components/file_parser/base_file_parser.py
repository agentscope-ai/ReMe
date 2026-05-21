"""Abstract base for file parsers."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileParser(BaseComponent):
    """Abstract base for file parsers. Subclasses implement `parse`."""

    component_type = ComponentEnum.FILE_PARSER

    def __init__(self, supported_extensions: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.working_dir = self.app_context.app_config.working_dir if self.app_context else ""
        self.supported_extensions: list[str] = supported_extensions or []

    def _get_relative_path(self, path: str | Path) -> str:
        """Return path relative to working_dir, or absolute path if outside."""
        file_path = Path(path).absolute()
        try:
            return str(file_path.relative_to(Path(self.working_dir).absolute()))
        except ValueError:
            return str(file_path)

    @abstractmethod
    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Parse a file into (node, chunks)."""
