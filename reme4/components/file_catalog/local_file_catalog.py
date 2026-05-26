"""In-memory file catalog with JSONL persistence."""

import aiofiles

from .base_file_catalog import BaseFileCatalog
from ..component_registry import R
from ...schema import FileNode


@R.register("local")
class LocalFileCatalog(BaseFileCatalog):
    """Dict-backed file catalog persisted as JSONL on close."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self._nodes: dict[str, FileNode] = {}
        self._catalog_file = self.catalog_path / f"{self.name}.jsonl"

    async def load(self) -> None:
        if not self._catalog_file.exists():
            return
        try:
            async with aiofiles.open(self._catalog_file, encoding=self.encoding) as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        node = FileNode.model_validate_json(line)
                        self._nodes[node.path] = node
            self.logger.info(f"Loaded {len(self._nodes)} nodes from {self._catalog_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load {self._catalog_file}: {e}")

    async def dump(self) -> None:
        try:
            tmp = self._catalog_file.with_suffix(".tmp")
            async with aiofiles.open(tmp, "w", encoding=self.encoding) as f:
                await f.write("\n".join(n.model_dump_json() for n in self._nodes.values()))
            tmp.replace(self._catalog_file)
            self.logger.info(f"Saved {len(self._nodes)} nodes to {self._catalog_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self._catalog_file}: {e}")

    async def upsert(self, nodes: list[FileNode]) -> None:
        for node in nodes:
            self._nodes[node.path] = node

    async def delete(self, path: str | list[str]) -> None:
        paths = [path] if isinstance(path, str) else path
        for p in paths:
            self._nodes.pop(p, None)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        if paths is None:
            return list(self._nodes.values())
        return [self._nodes[p] for p in paths if p in self._nodes]
