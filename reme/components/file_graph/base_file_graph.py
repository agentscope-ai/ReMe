"""Abstract base for file-graph backends."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum, LinkScopeEnum
from ...schema import FileLink, FileNode
from ...utils import canonical_index_path


class BaseFileGraph(BaseComponent):
    """Abstract base for file-graph backends.

    Link scope (``get_outlinks`` / ``get_inlinks``):
        REAL    edges touching an indexed node
        VIRTUAL edges touching a dangling placeholder
                (referenced but never upserted, or already deleted)
        ALL     both
    """

    component_type = ComponentEnum.FILE_GRAPH

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dropped_chunk_ids: set[str] = set()

    # -- Path identity -----------------------------------------------------

    @staticmethod
    def canonical_path(path: str) -> str:
        """Return the workspace-relative POSIX identity of ``path``."""
        return canonical_index_path(path)

    @classmethod
    def canonical_node(cls, node: FileNode) -> FileNode:
        """Return ``node`` with its own path and link paths canonicalized."""
        links = [
            link.model_copy(
                update={
                    "source_path": canonical_index_path(link.source_path),
                    "target_path": canonical_index_path(link.target_path),
                },
            )
            for link in node.links
        ]
        path = canonical_index_path(node.path)
        if path == node.path and links == node.links:
            return node
        return node.model_copy(update={"path": path, "links": links})

    def select_winning_record(self, keep: FileNode, extra: FileNode) -> FileNode:
        """Return ``keep`` and remember the handles only ``extra`` referenced.

        Handles owned by the losing spelling are reported through
        :meth:`take_dropped_chunk_ids`, so the owning store can remove the
        superseded text instead of leaving it searchable under the winner.
        """
        self._dropped_chunk_ids.update(
            [chunk_id for chunk_id in extra.chunk_ids if chunk_id not in keep.chunk_ids],
        )
        return keep

    async def take_dropped_chunk_ids(self) -> list[str]:
        """Return and clear handles that no surviving record references any more.

        A later merge can hand a previously discarded handle to the winning
        record, so recorded handles are filtered against the live graph before
        the store deletes anything.
        """
        if not self._dropped_chunk_ids:
            return []
        nodes = await self.get_nodes()
        referenced = {chunk_id for node in nodes for chunk_id in node.chunk_ids}
        dropped = sorted(self._dropped_chunk_ids - referenced)
        self._dropped_chunk_ids.clear()
        return dropped

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        await self.load()

    async def _close(self) -> None:
        if self.is_started:
            await self.dump()
        await super()._close()

    async def load(self) -> None:
        """Restore persisted state. No-op for backends without local files."""

    async def dump(self) -> None:
        """Persist state. No-op for backends without local files."""

    # -- Node CRUD ---------------------------------------------------------

    @abstractmethod
    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        """Insert or update nodes."""

    @abstractmethod
    async def delete_nodes(self, paths: list[str]) -> None:
        """Remove nodes by path."""

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return nodes by paths; ``None`` = all real nodes."""

    @abstractmethod
    async def rebuild_links(self) -> None:
        """Rebuild all edges from each node's link payload."""

    @abstractmethod
    async def clear(self):
        """Remove all nodes and edges."""

    # -- Link access -------------------------------------------------------

    @abstractmethod
    async def get_outlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        """Outgoing links from *path*."""

    @abstractmethod
    async def get_inlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        """Inbound links to *path*."""
