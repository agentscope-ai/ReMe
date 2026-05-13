"""Abstract base class for file parsers."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileParser(BaseComponent):
    """Abstract base class for file parsers."""

    component_type = ComponentEnum.FILE_PARSER

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.working_dir = self.app_context.app_config.working_dir if self.app_context is not None else ""


    def _get_relative_path(self, path: str | Path) -> str:
        """Get path relative to working_dir."""
        file_path = Path(path).absolute()
        working_path = Path(self.working_dir).absolute()
        try:
            return str(file_path.relative_to(working_path))
        except ValueError:
            return str(file_path)

    @abstractmethod
    async def parse(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Parse a file into metadata and chunks. Subclasses must implement this."""
