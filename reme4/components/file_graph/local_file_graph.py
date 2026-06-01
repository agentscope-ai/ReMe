"""Pure-Python file-graph backend (no external deps)."""

from pathlib import Path

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...enumeration import LinkScopeEnum
from ...schema import FileLink, FileNode


@R.register("local")
class LocalFileGraph(BaseFileGraph):
    """Dict-backed file graph; uses FileLink.target_path for adjacency."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nodes: dict[str, FileNode] = {}
        self._inverse: dict[str, set[str]] = {}  # real target → sources
        self._pending: dict[str, set[str]] = {}  # virtual target → sources
        self._graph_file: Path = self.component_metadata_path / f"{self.name}.jsonl"

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        await super()._start()  # base calls load()
        await self.rebuild_links()

    async def load(self) -> None:
        if not self._graph_file.exists():
            return
        try:
            with open(self._graph_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        node = FileNode.model_validate_json(line)
                        self._nodes[node.path] = node
            self.logger.info(f"Loaded {len(self._nodes)} nodes from {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load {self._graph_file}: {e}")

    async def dump(self) -> None:
        try:
            tmp = self._graph_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(f"{n.model_dump_json()}\n" for n in self._nodes.values())
            tmp.replace(self._graph_file)
            self.logger.info(f"Saved {len(self._nodes)} nodes to {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self._graph_file}: {e}")

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

    def _scope_match(self, target: str, scope: LinkScopeEnum) -> bool:
        if scope is LinkScopeEnum.ALL:
            return True
        is_real = target in self._nodes
        return is_real if scope is LinkScopeEnum.REAL else not is_real

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
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
        if paths is None:
            return list(self._nodes.values())
        return [self._nodes[p] for p in paths if p in self._nodes]

    async def rebuild_links(self) -> None:
        self._inverse.clear()
        self._pending.clear()
        for src, node in self._nodes.items():
            for target in self._targets(node):
                self._add_edge(src, target)

    async def clear(self):
        self._nodes.clear()
        self._inverse.clear()
        self._pending.clear()
        self._graph_file.unlink(missing_ok=True)

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        node = self._nodes.get(path)
        if node is None:
            return []
        return [lnk for lnk in node.links if lnk.target_path and self._scope_match(lnk.target_path, scope)]

    async def get_inlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        sources: set[str] = set()
        if scope in (LinkScopeEnum.REAL, LinkScopeEnum.ALL):
            sources |= self._inverse.get(path, set())
        if scope in (LinkScopeEnum.VIRTUAL, LinkScopeEnum.ALL):
            sources |= self._pending.get(path, set())
        return [
            link for src in sources if src in self._nodes for link in self._nodes[src].links if link.target_path == path
        ]
