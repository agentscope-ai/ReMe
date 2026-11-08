"""Browse indexed tag matches without requiring a text-search query."""

from pathlib import Path

from ..base_step import BaseStep
from ..file_io._path import _check_path_permission, resolve_path
from ...components import R


@R.register("list_tag_files_step")
class ListTagFilesStep(BaseStep):
    """Return a bounded, workspace-relative page of existing tagged files."""

    async def execute(self):
        assert self.context is not None
        page = self.context.get("page", 1)
        page_size = self.context.get("page_size", 100)
        if type(page) is not int or page < 1:
            raise ValueError("page must be a positive integer")
        if type(page_size) is not int or not 1 <= page_size <= 1000:
            raise ValueError("page_size must be an integer from 1 to 1000")
        index = self.file_store.require_tag_index()
        if not index.is_healthy:
            raise RuntimeError("tag index is unavailable")
        tags = index.normalize_query_tags(self.context.get("tags"))
        if not tags:
            raise ValueError("tags must contain at least one valid tag")
        workspace = Path(self.file_store.workspace_path).resolve()
        directory, error = resolve_path(workspace, self.context.get("path", ""), allow_empty=True)
        if error or directory is None:
            raise ValueError(error or "invalid path")
        allowed = self.context.get("_allowed_paths")
        paths = []
        for path in await index.paths_for_tags(tags, match_all=False):
            target, error = resolve_path(workspace, path)
            if error or target is None or not target.is_relative_to(directory):
                continue
            if target.is_file() and _check_path_permission(workspace, target, allowed):
                paths.append(path)
        if not index.is_healthy:
            raise RuntimeError("tag index is unavailable")
        start = (page - 1) * page_size
        items = [{"path": path, "tags": await index.tags_for_path(path)} for path in paths[start : start + page_size]]
        if not index.is_healthy:
            raise RuntimeError("tag index is unavailable")
        self.context.response.metadata["tag_filter"] = {
            "requested": True,
            "applied": True,
            "tags": tags,
            "matched_paths": len(paths),
        }
        self.context.response.answer = {
            "total_files": len(paths),
            "total_pages": (len(paths) + page_size - 1) // page_size,
            "page": page,
            "range": (start + 1, start + len(items)) if items else (0, 0),
            "items": items,
        }
        return self.context.response
