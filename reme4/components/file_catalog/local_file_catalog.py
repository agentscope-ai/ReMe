"""Local file catalog backend: in-memory dict persisted as JSONL."""

import aiofiles

from .base_file_catalog import BaseFileCatalog
from ..component_registry import R
from ...schema import FileNode


@R.register("local")
class LocalFileCatalog(BaseFileCatalog):
    """Dict-backed catalog persisted as JSONL."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self._nodes: dict[str, FileNode] = {}
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        self._catalog_file = self.component_metadata_path / f"{self.name}.jsonl"

    async def load(self) -> None:
        if not self._catalog_file.exists():
            return
        try:
            await self._read_jsonl()
            self.logger.info(f"Loaded {len(self._nodes)} nodes from {self._catalog_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load {self._catalog_file}: {e}")

    async def dump(self) -> None:
        try:
            await self._write_jsonl()
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

    async def _read_jsonl(self) -> None:
        async with aiofiles.open(self._catalog_file, encoding=self.encoding) as f:
            async for line in f:
                if stripped := line.strip():
                    node = FileNode.model_validate_json(stripped)
                    self._nodes[node.path] = node

    async def _write_jsonl(self) -> None:
        tmp = self._catalog_file.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w", encoding=self.encoding) as f:
            await f.write("\n".join(n.model_dump_json() for n in self._nodes.values()))
        tmp.replace(self._catalog_file)
