"""Pure-Python file-graph backend (no external deps)."""

from pathlib import Path

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...schema import FileLink, FileNode


@R.register("local")
class LocalFileGraph(BaseFileGraph):
    """Dict-backed file graph; trusts ``FileLink.path`` for adjacency."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nodes: dict[str, FileNode] = {}
        self._inverse: dict[str, set[str]] = {}
        # Virtual edges: src→target where target isn't in ``_nodes`` yet.
        self._pending: dict[str, set[str]] = {}
        self._graph_file: Path = self.graph_path / f"{self.graph_name}.jsonl"

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        self._load()
        await self.rebuild_links()
        edges = sum(len(s) for s in self._inverse.values())
        pending = sum(len(s) for s in self._pending.values())
        self.logger.info(
            f"LocalFileGraph '{self.graph_name}' ready: "
            f"{len(self._nodes)} nodes, {edges} edges, {pending} pending",
        )

    async def _close(self) -> None:
        self._dump()
        await super()._close()

    def _load(self) -> None:
        if not self._graph_file.exists():
            return
        with open(self._graph_file, "r", encoding="utf-8") as f:
            self._nodes.update(
                (n.path, n)
                for n in (FileNode.model_validate_json(line) for line in f if line.strip())
            )

    def _dump(self) -> None:
        tmp = self._graph_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(f"{n.model_dump_json()}\n" for n in self._nodes.values())
        tmp.replace(self._graph_file)

    # -- Edge bookkeeping --------------------------------------------------

    def _add_edge(self, src: str, target: str) -> None:
        bucket = self._inverse if target in self._nodes else self._pending
        bucket.setdefault(target, set()).add(src)

    def _remove_edge(self, src: str, target: str) -> None:
        for bucket in (self._inverse, self._pending):
            srcs = bucket.get(target)
            if srcs is None or src not in srcs:
                continue
            srcs.discard(src)
            if not srcs:
                del bucket[target]

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        for node in nodes:
            path = node.path
            old = self._nodes.get(path)
            if old is not None:
                for link in old.links:
                    if link.path:
                        self._remove_edge(path, link.path)
            self._nodes[path] = node
            for link in node.links:
                if link.path:
                    self._add_edge(path, link.path)
            # Promote virtual edges aimed at this newly-arrived target.
            promoted = self._pending.pop(path, None)
            if promoted:
                self._inverse.setdefault(path, set()).update(promoted)

    async def delete_nodes(self, paths: list[str]) -> None:
        for path in paths:
            node = self._nodes.pop(path, None)
            if node is None:
                continue
            for link in node.links:
                if link.path:
                    self._remove_edge(path, link.path)
            # Demote inbound edges to virtual; sources still link here.
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
            for link in node.links:
                if link.path:
                    self._add_edge(src, link.path)

    async def clear(self):
        self._nodes.clear()
        self._inverse.clear()
        self._pending.clear()

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str) -> list[FileLink]:
        node = self._nodes.get(path)
        if node is None:
            return []
        return [link for link in node.links if link.path and link.path in self._nodes]

    async def get_inlinks(self, path: str) -> list[FileLink]:
        if path not in self._nodes:
            return []
        return [
            link
            for src in self._inverse.get(path, ())
            for link in self._nodes[src].links
            if link.path == path
        ]