"""In-memory tag index derived from ``FileNode.front_matter``."""

import asyncio
from pathlib import PurePosixPath

from .base_tag_index import BaseTagIndex, TagListItem, TagListResult, TagOrder, TagOrderBy
from ..component_registry import R
from ...schema import FileFrontMatter, FileNode


@R.register("local")
class LocalTagIndex(BaseTagIndex):
    """Maintain bidirectional path/tag relationships without separate source I/O."""

    reserved_tag_keys = frozenset(FileFrontMatter.model_fields) | {
        "kind",
        "session_id",
        "source_conversation",
        "source_resource",
        "status",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path_to_tags: dict[str, tuple[str, ...]] = {}
        self.tag_to_paths: dict[str, set[str]] = {}
        self._maintenance_lock = asyncio.Lock()

    @property
    def n_files(self) -> int:
        return len(self.path_to_tags)

    def _normalize_tags(self, value: object, *, limit: int | None) -> list[str]:
        """Normalize a strict tag list, optionally limiting the result count."""
        if not isinstance(value, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                continue
            raw = "_".join(str(item).split())
            if not raw or len(raw) > self.max_tag_length:
                continue
            if not any(char.isalnum() for char in raw):
                continue
            canonical = raw.casefold()
            if canonical in seen:
                continue
            seen.add(canonical)
            result.append(canonical)
            if limit is not None and len(result) >= limit:
                break
        return result

    def normalize_tags(self, value: object) -> list[str]:
        """Normalize frontmatter tags according to the per-file count limit."""
        return self._normalize_tags(value, limit=self.max_tags_per_file)

    def normalize_query_tags(self, value: object) -> list[str]:
        """Normalize lookup tags without truncating the query expression."""
        return self._normalize_tags(value, limit=None)

    @staticmethod
    def _validate_path(path: str) -> str:
        if not isinstance(path, str) or not path or "\\" in path:
            raise ValueError(f"Invalid workspace-relative tag-index path: {path!r}")
        pure = PurePosixPath(path)
        if pure.is_absolute() or path != pure.as_posix() or any(part in ("", ".", "..") for part in pure.parts):
            raise ValueError(f"Invalid workspace-relative tag-index path: {path!r}")
        return path

    def _prepare_nodes(self, nodes: list[FileNode]) -> list[tuple[str, tuple[str, ...]]]:
        prepared: list[tuple[str, tuple[str, ...]]] = []
        for node in nodes:
            path = self._validate_path(node.path)
            frontmatter = node.front_matter.model_extra or {}
            tags = self.normalize_tags(frontmatter.get(self.tag_key))
            prepared.append((path, tuple(tags)))
        return prepared

    @staticmethod
    def _replace(
        path_to_tags: dict[str, tuple[str, ...]],
        tag_to_paths: dict[str, set[str]],
        path: str,
        tags: tuple[str, ...],
    ) -> None:
        old_tags = path_to_tags.pop(path, ())
        for tag in old_tags:
            paths = tag_to_paths[tag]
            paths.discard(path)
            if not paths:
                del tag_to_paths[tag]
        if not tags:
            return
        path_to_tags[path] = tags
        for tag in tags:
            tag_to_paths.setdefault(tag, set()).add(path)

    async def rebuild(self, nodes: list[FileNode]) -> None:
        prepared = self._prepare_nodes(nodes)
        path_to_tags: dict[str, tuple[str, ...]] = {}
        tag_to_paths: dict[str, set[str]] = {}
        for path, tags in prepared:
            self._replace(path_to_tags, tag_to_paths, path, tags)
        async with self._maintenance_lock:
            self.path_to_tags = path_to_tags
            self.tag_to_paths = tag_to_paths
            self.is_healthy = True
        self.logger.info(f"Rebuilt tag index: files={len(path_to_tags)}, tags={len(tag_to_paths)}")

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        prepared = self._prepare_nodes(nodes)
        if not prepared:
            return
        async with self._maintenance_lock:
            for path, tags in prepared:
                self._replace(self.path_to_tags, self.tag_to_paths, path, tags)

    async def delete(self, paths: list[str]) -> None:
        validated = [self._validate_path(path) for path in paths]
        if not validated:
            return
        async with self._maintenance_lock:
            for path in validated:
                self._replace(self.path_to_tags, self.tag_to_paths, path, ())

    async def paths_for_tags(self, tags: object, *, match_all: bool = True) -> list[str]:
        if not self.is_healthy:
            return []
        # ``max_tags_per_file`` constrains indexed documents, not lookup
        # expressions. Truncating here would silently weaken AND queries and
        # omit valid matches from OR queries.
        normalized = self.normalize_query_tags(tags)
        if not normalized:
            return []
        async with self._maintenance_lock:
            postings = [self.tag_to_paths.get(tag, set()) for tag in normalized]
            matches = set.intersection(*postings) if match_all else set.union(*postings)
            return sorted(matches)

    async def tags_for_path(self, path: str) -> list[str]:
        if not self.is_healthy:
            return []
        path = self._validate_path(path)
        async with self._maintenance_lock:
            return list(self.path_to_tags.get(path, ()))

    async def list_tags(
        self,
        *,
        page: int = 1,
        order_by: TagOrderBy = "tag",
        order: TagOrder | None = None,
        page_size: int = 100,
    ) -> TagListResult:
        """Return a deterministic page of active tags, counts, and its 1-based range."""
        page = self._positive_int("page", page)
        page_size = self._positive_int("page_size", page_size)
        if page_size > 1000:
            raise ValueError("page_size must be less than or equal to 1000")

        if not isinstance(order_by, str):
            raise ValueError("order_by must be one of ['file_count', 'tag']")
        order_by = order_by.lower()
        if order_by not in {"tag", "file_count"}:
            raise ValueError("order_by must be one of ['file_count', 'tag']")
        if order is None:
            order = "asc" if order_by == "tag" else "desc"
        if not isinstance(order, str):
            raise ValueError("order must be one of ['asc', 'desc']")
        order = order.lower()
        if order not in {"asc", "desc"}:
            raise ValueError("order must be one of ['asc', 'desc']")

        async with self._maintenance_lock:
            if not self.is_healthy:
                raise RuntimeError("tag index is unavailable")
            items: list[TagListItem] = [(tag, len(paths)) for tag, paths in self.tag_to_paths.items()]

        if order_by == "tag":
            items.sort(key=lambda item: item[0], reverse=order == "desc")
        else:
            direction = 1 if order == "asc" else -1
            items.sort(key=lambda item: (direction * item[1], item[0]))

        total_tags = len(items)
        total_pages = (total_tags + page_size - 1) // page_size
        start = (page - 1) * page_size
        page_items = items[start : start + page_size]
        item_range = (start + 1, start + len(page_items)) if page_items else (0, 0)
        return {
            "total_tags": total_tags,
            "total_pages": total_pages,
            "page": page,
            "range": item_range,
            "items": page_items,
        }

    async def clear(self) -> None:
        async with self._maintenance_lock:
            self.path_to_tags = {}
            self.tag_to_paths = {}
            self.is_healthy = True

    async def _close(self) -> None:
        await self.clear()
