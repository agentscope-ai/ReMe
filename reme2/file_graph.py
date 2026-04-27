import json
from collections import defaultdict
from pathlib import Path

from reme2.schema.file_metadata import FileMetadata


class FileGraph:

    def __init__(self) -> None:
        self._nodes: dict[str, FileMetadata] = {}
        self._backlinks: dict[str, set[str]] = defaultdict(set)

    # -- CRUD ----------------------------------------------------------------

    def add(self, *metadatas: FileMetadata) -> None:
        for metadata in metadatas:
            path = metadata.path
            if path in self._nodes:
                self._remove_forward(self._nodes[path])
            self._nodes[path] = metadata
        for metadata in metadatas:
            self._add_forward(metadata)

    def update(self, path: str, **fields) -> FileMetadata | None:
        metadata = self._nodes.get(path)
        if metadata is None:
            return None
        self._remove_forward(metadata)
        updated = metadata.model_copy(update=fields)
        self._nodes[path] = updated
        self._add_forward(updated)
        return updated

    def remove(self, path: str) -> FileMetadata | None:
        metadata = self._nodes.pop(path, None)
        if metadata is None:
            return None
        self._remove_forward(metadata)
        self._backlinks.pop(path, None)
        return metadata

    def get(self, path: str) -> FileMetadata | None:
        return self._nodes.get(path)

    # -- Link queries --------------------------------------------------------

    def get_links(self, path: str) -> list[FileMetadata]:
        metadata = self._nodes.get(path)
        if metadata is None:
            return []
        return [self._nodes[link] for link in metadata.link if link in self._nodes]

    def get_backlinks(self, path: str) -> list[FileMetadata]:
        return [
            self._nodes[src]
            for src in self._backlinks.get(path, set())
            if src in self._nodes
        ]

    # -- Index helpers -------------------------------------------------------

    def _add_forward(self, metadata: FileMetadata) -> None:
        for link in metadata.link:
            self._backlinks[link].add(metadata.path)

    def _remove_forward(self, metadata: FileMetadata) -> None:
        for link in metadata.link:
            self._backlinks[link].discard(metadata.path)

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {p: meta.model_dump(mode="json") for p, meta in self._nodes.items()}
        content = json.dumps(raw, ensure_ascii=False)
        temp = path.with_suffix(".tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "FileGraph":
        path = Path(path)
        graph = cls()
        if not path.exists():
            return graph
        raw: dict = json.loads(path.read_text(encoding="utf-8"))
        nodes = [FileMetadata(**meta) for meta in raw.values()]
        graph.add(*nodes)
        return graph

    # -- Dunder --------------------------------------------------------------

    @property
    def nodes(self) -> dict[str, FileMetadata]:
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, path: str) -> bool:
        return path in self._nodes
