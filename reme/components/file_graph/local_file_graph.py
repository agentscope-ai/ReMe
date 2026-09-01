"""Pure-Python file-graph backend (no external deps)."""

import asyncio
from pathlib import Path

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...enumeration import LinkScopeEnum
from ...schema import FileLink, FileNode
from ...utils.async_utils import complete_in_thread
from ...utils.jsonl_zst import read_jsonl_zst, write_jsonl_zst


@R.register("local")
class LocalFileGraph(BaseFileGraph):
    """Dict-backed file graph; uses FileLink.target_path for adjacency."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nodes: dict[str, FileNode] = {}
        self._inverse: dict[str, set[str]] = {}  # real target → sources
        self._pending: dict[str, set[str]] = {}  # virtual target → sources
        self._io_lock = asyncio.Lock()
        self._graph_file: Path = self.component_metadata_path / f"{self.name}.jsonl.zst"

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        await super()._start()  # base calls load()

    async def load(self) -> None:
        async with self._io_lock:
            if not self._graph_file.exists():
                return
            try:
                nodes, inverse, pending = await complete_in_thread(
                    self._load_sync,
                    self._graph_file,
                    self._nodes,
                )
                self._nodes = nodes
                self._inverse = inverse
                self._pending = pending
                self.logger.debug(f"Loaded {len(self._nodes)} nodes from {self._graph_file}")
            except Exception as e:
                self.logger.exception(f"Failed to load {self._graph_file}: {e}")

    async def dump(self) -> None:
        async with self._io_lock:
            try:
                await complete_in_thread(self._dump_sync)
                self.logger.info(f"Saved {len(self._nodes)} nodes to {self._graph_file}")
            except Exception as e:
                self.logger.exception(f"Failed to write {self._graph_file}: {e}")

    @classmethod
    def _load_sync(
        cls,
        path: Path,
        existing: dict[str, FileNode],
    ) -> tuple[dict[str, FileNode], dict[str, set[str]], dict[str, set[str]]]:
        """Restore nodes and rebuild adjacency entirely outside the event loop."""
        nodes = dict(existing)
        for line in read_jsonl_zst(path):
            if stripped := line.strip():
                node = FileNode.model_validate_json(stripped)
                nodes[node.path] = node

        inverse: dict[str, set[str]] = {}
        pending: dict[str, set[str]] = {}
        for source, node in nodes.items():
            for target in cls._targets(node):
                bucket = inverse if target in nodes else pending
                bucket.setdefault(target, set()).add(source)
        return nodes, inverse, pending

    def _dump_sync(self) -> None:
        """Serialize and atomically publish the graph while its state is locked."""
        write_jsonl_zst(self._graph_file, (n.model_dump_json() for n in self._nodes.values()))

    # -- Internals ---------------------------------------------------------

    @staticmethod
    def _targets(node: FileNode) -> list[str]:
        return [lnk.target_path for lnk in node.links if lnk.target_path]

    def _add_edge(self, src: str, target: str) -> None:
        bucket = self._inverse if target in self._nodes else self._pending
        bucket.setdefault(target, set()).add(src)

    def _remove_edge(self, src: str, target: str) -> None:
        for bucket in (self._inverse, self._pending):
            srcs = bucket.get(target)
            if srcs and src in srcs:
                srcs.discard(src)
                if not srcs:
                    del bucket[target]

    @staticmethod
    def _normalize_scope(scope: LinkScopeEnum | str) -> LinkScopeEnum:
        return scope if isinstance(scope, LinkScopeEnum) else LinkScopeEnum(scope)

    def _scope_match(self, target: str, scope: LinkScopeEnum | str) -> bool:
        scope = self._normalize_scope(scope)
        if scope is LinkScopeEnum.ALL:
            return True
        is_real = target in self._nodes
        return is_real if scope is LinkScopeEnum.REAL else not is_real

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        async with self._io_lock:
            for node in nodes:
                path = node.path
                old = self._nodes.get(path)
                if old is not None:
                    for target in self._targets(old):
                        self._remove_edge(path, target)
                self._nodes[path] = node
                for target in self._targets(node):
                    self._add_edge(path, target)
                promoted = self._pending.pop(path, None)
                if promoted:
                    self._inverse.setdefault(path, set()).update(promoted)

    async def delete_nodes(self, paths: list[str]) -> None:
        async with self._io_lock:
            for path in paths:
                node = self._nodes.pop(path, None)
                if node is None:
                    continue
                for target in self._targets(node):
                    self._remove_edge(path, target)
                demoted = self._inverse.pop(path, None)
                if demoted:
                    self._pending.setdefault(path, set()).update(demoted)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        async with self._io_lock:
            if paths is None:
                return list(self._nodes.values())
            return [self._nodes[p] for p in paths if p in self._nodes]

    async def rebuild_links(self) -> None:
        async with self._io_lock:
            nodes, inverse, pending = await complete_in_thread(self._rebuild_links_sync, self._nodes)
            self._nodes = nodes
            self._inverse = inverse
            self._pending = pending

    @classmethod
    def _rebuild_links_sync(
        cls,
        existing: dict[str, FileNode],
    ) -> tuple[dict[str, FileNode], dict[str, set[str]], dict[str, set[str]]]:
        """Rebuild adjacency from an in-memory node generation off-loop."""
        nodes = dict(existing)
        inverse: dict[str, set[str]] = {}
        pending: dict[str, set[str]] = {}
        for source, node in nodes.items():
            for target in cls._targets(node):
                bucket = inverse if target in nodes else pending
                bucket.setdefault(target, set()).add(source)
        return nodes, inverse, pending

    async def clear(self):
        async with self._io_lock:
            self._nodes.clear()
            self._inverse.clear()
            self._pending.clear()
            self._graph_file.unlink(missing_ok=True)

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str, scope: LinkScopeEnum | str = LinkScopeEnum.REAL) -> list[FileLink]:
        async with self._io_lock:
            scope = self._normalize_scope(scope)
            node = self._nodes.get(path)
            if node is None:
                return []
            return [lnk for lnk in node.links if lnk.target_path and self._scope_match(lnk.target_path, scope)]

    async def get_inlinks(self, path: str, scope: LinkScopeEnum | str = LinkScopeEnum.REAL) -> list[FileLink]:
        async with self._io_lock:
            scope = self._normalize_scope(scope)
            sources: set[str] = set()
            if scope in (LinkScopeEnum.REAL, LinkScopeEnum.ALL):
                sources |= self._inverse.get(path, set())
            if scope in (LinkScopeEnum.VIRTUAL, LinkScopeEnum.ALL):
                sources |= self._pending.get(path, set())
            return [
                link
                for src in sorted(sources)
                if src in self._nodes
                for link in self._nodes[src].links
                if link.target_path == path
            ]
