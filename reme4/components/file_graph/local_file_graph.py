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
        self._inverse: dict[str, set[str]] = {}  # target → {sources}
        self._pending: dict[str, set[str]] = {}  # virtual target → {sources}
        self._graph_file: Path = self.graph_path / f"{self.graph_name}_{self.graph_version}.jsonl"

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()  # base calls load()
        await self.rebuild_links()

    async def load(self) -> None:
        """Load nodes from JSONL file into memory; keep current state on failure."""
        if not self._graph_file.exists():
            return
        try:
            with open(self._graph_file, "r", encoding="utf-8") as f:
                self._nodes.update(
                    (n.path, n) for line in f if line.strip() for n in [FileNode.model_validate_json(line)]
                )
            self.logger.info(f"Loaded {len(self._nodes)} nodes from {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load {self._graph_file}: {e}")

    async def dump(self) -> None:
        """Persist all nodes to JSONL via atomic rename."""
        try:
            tmp = self._graph_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(f"{n.model_dump_json()}\n" for n in self._nodes.values())
            tmp.replace(self._graph_file)
            self.logger.info(f"Saved {len(self._nodes)} nodes to {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self._graph_file}: {e}")

    # -- Edge bookkeeping --------------------------------------------------

    def _add_edge(self, src: str, target: str) -> None:
        """Register src→target; route to pending if target is virtual."""
        bucket = self._inverse if target in self._nodes else self._pending
        bucket.setdefault(target, set()).add(src)

    def _remove_edge(self, src: str, target: str) -> None:
        """Remove src→target from both inverse and pending buckets."""
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
                    if link.target_path:
                        self._remove_edge(path, link.target_path)
            self._nodes[path] = node
            for link in node.links:
                if link.target_path:
                    self._add_edge(path, link.target_path)
            # Promote pending edges that now target a real node.
            promoted = self._pending.pop(path, None)
            if promoted:
                self._inverse.setdefault(path, set()).update(promoted)

    async def delete_nodes(self, paths: list[str]) -> None:
        for path in paths:
            node = self._nodes.pop(path, None)
            if node is None:
                continue
            for link in node.links:
                if link.target_path:
                    self._remove_edge(path, link.target_path)
            # Demote inbound edges to pending (sources still reference this path).
            demoted = self._inverse.pop(path, None)
            if demoted:
                self._pending.setdefault(path, set()).update(demoted)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        if paths is None:
            return list(self._nodes.values())
        return [self._nodes[p] for p in paths if p in self._nodes]

    async def rebuild_links(self) -> None:
        """Rebuild inverse/pending indexes from all node link payloads."""
        self._inverse.clear()
        self._pending.clear()
        for src, node in self._nodes.items():
            for link in node.links:
                if link.target_path:
                    self._add_edge(src, link.target_path)

    async def clear(self):
        self._nodes.clear()
        self._inverse.clear()
        self._pending.clear()
        self._graph_file.unlink(missing_ok=True)

    # -- Link access -------------------------------------------------------

    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        # Source must be real (only real nodes carry a ``links`` payload).
        # Targets may be real or virtual; ``scope`` selects which to surface.
        node = self._nodes.get(path)
        if node is None:
            return []
        return [lnk for lnk in node.links if lnk.target_path and _match_target(lnk.target_path, self._nodes, scope)]

    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        # ``_inverse`` keys real targets; ``_pending`` keys virtual ones.
        # The queried ``path`` lives in at most one bucket, so ``scope``
        # is satisfied by selecting which bucket to read.
        sources: set[str] = set()
        if scope in (LinkScopeEnum.REAL, LinkScopeEnum.ALL):
            sources |= self._inverse.get(path, set())
        if scope in (LinkScopeEnum.VIRTUAL, LinkScopeEnum.ALL):
            sources |= self._pending.get(path, set())
        return [
            link for src in sources if src in self._nodes for link in self._nodes[src].links if link.target_path == path
        ]


def _match_target(target_path: str, nodes: dict, scope: LinkScopeEnum) -> bool:
    """Whether an edge into ``target_path`` should be surfaced under ``scope``."""
    if scope is LinkScopeEnum.ALL:
        return True
    is_real = target_path in nodes
    return is_real if scope is LinkScopeEnum.REAL else not is_real
