"""Local file catalog backend: in-memory dict persisted as compressed JSONL."""

import asyncio

from .base_file_catalog import BaseFileCatalog
from ..component_registry import R
from ...schema import FileNode
from ...utils.jsonl_zst import read_jsonl_zst, write_jsonl_zst


@R.register("local")
class LocalFileCatalog(BaseFileCatalog):
    """Dict-backed catalog persisted as JSONL."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self._nodes: dict[str, FileNode] = {}
        self._io_lock = asyncio.Lock()
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        self._catalog_file = self.component_metadata_path / f"{self.name}.jsonl.zst"

    async def load(self) -> None:
        async with self._io_lock:
            if not self._catalog_file.exists():
                return
            await self._read_jsonl()
            self.logger.debug(f"Loaded {len(self._nodes)} nodes from {self._catalog_file}")

    async def dump(self) -> None:
        async with self._io_lock:
            await self._write_jsonl()
            self.logger.info(f"Saved {len(self._nodes)} nodes to {self._catalog_file}")

    async def upsert(self, nodes: list[FileNode]) -> None:
        async with self._io_lock:
            for node in nodes:
                self._nodes[node.path] = node

    async def delete(self, path: str | list[str]) -> None:
        paths = [path] if isinstance(path, str) else path
        async with self._io_lock:
            for p in paths:
                self._nodes.pop(p, None)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        async with self._io_lock:
            if paths is None:
                return list(self._nodes.values())
            return [self._nodes[p] for p in paths if p in self._nodes]

    async def _read_jsonl(self) -> None:
        for line in read_jsonl_zst(self._catalog_file, self.encoding):
            if stripped := line.strip():
                node = FileNode.model_validate_json(stripped)
                self._nodes[node.path] = node

    async def _write_jsonl(self) -> None:
        write_jsonl_zst(self._catalog_file, (n.model_dump_json() for n in self._nodes.values()), self.encoding)
