"""Abstract base for file chunkers."""

from abc import abstractmethod
import hashlib
import json
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class BaseFileChunker(BaseComponent):
    """Abstract base for file chunkers. Subclasses implement `chunk`."""

    component_type = ComponentEnum.FILE_CHUNKER

    def __init__(self, supported_extensions: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions: list[str] = supported_extensions or []

    def config_fingerprint(self) -> str:
        """Return a stable fingerprint for the chunker's derived output shape."""
        payload = self._config_fingerprint_payload()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def _config_fingerprint_payload(self) -> dict:
        """Return the structured state that should invalidate persisted chunks."""
        return {
            "type": type(self).__name__,
            "backend": self.backend,
            "supported_extensions": list(self.supported_extensions),
        }

    @abstractmethod
    async def chunk(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Chunk a file into (node, chunks)."""
