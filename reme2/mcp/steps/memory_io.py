"""MCP step shells over the Memory File System engine API.

Per `structure.md`, .md files are the SSOT and the engine surface lives
in `reme2.memory.memory_io` (CRUD writes + MFS reads + Projections).
This module only hosts the `@R.register("memory_*")` Step shells —
each one translates RuntimeContext ↔ JSON payload and delegates to the
matching engine API function.

Search ops live in `memory_retriever.py` (Retriever composes V/K/graph
projections with policy).
"""

from __future__ import annotations

from pathlib import Path

from ...component import R
from ...component.base_step import BaseStep
from ...component.runtime_response import _set_answer
from ...enumeration import ComponentEnum
from ...memory import memory_io


@R.register("memory_get")
class MemoryGet(BaseStep):
    """Read a single memory file (frontmatter + body, optional chunks)."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "")
        include_chunks: bool = bool(self.context.get("include_chunks", False))
        assert path, "path is required"
        result = await memory_io.read_file(self.file_store, path, include_chunks=include_chunks)
        _set_answer(self.context, result)


@R.register("memory_list")
class MemoryList(BaseStep):
    """List indexed files filtered by frontmatter fields, tags, or path prefix."""

    async def execute(self):
        assert self.context is not None
        result = memory_io.list_files(
            self.file_store,
            path_prefix=self.context.get("path_prefix"),
            tags=self.context.get("tags") or [],
            metadata=self.context.get("metadata") or {},
            limit=int(self.context.get("limit", 100)),
        )
        _set_answer(self.context, result)


@R.register("memory_backlinks")
class MemoryBacklinks(BaseStep):
    """Files linking to a given path. Each entry carries the typed-edge predicate."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "")
        assert path, "path is required"
        _set_answer(self.context, memory_io.backlinks_of(self.file_store, path))


@R.register("memory_links")
class MemoryLinks(BaseStep):
    """Files a given path links to (resolved). Each entry carries the typed-edge predicate."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "")
        assert path, "path is required"
        _set_answer(self.context, memory_io.links_of(self.file_store, path))


@R.register("memory_resolve_wikilink")
class MemoryResolveWikilink(BaseStep):
    """Resolve a `[[target]]` wikilink with full ambiguity context."""

    async def execute(self):
        assert self.context is not None
        wikilink: str = self.context.get("wikilink", "") or ""
        assert wikilink, "wikilink is required"
        payload = memory_io.wikilink_lookup(self.file_store, wikilink)
        self.context.response.success = bool(payload.get("exists"))
        _set_answer(self.context, payload)


@R.register("memory_create")
class MemoryCreate(BaseStep):
    """Create a markdown file. Wikilink-uniqueness gate runs unless force=True."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "")
        metadata: dict = dict(self.context.get("metadata", {}) or {})
        content: str = self.context.get("content", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", False))
        force: bool = bool(self.context.get("force", False))

        assert path, "path is required"
        target = Path(path)

        ok, payload = memory_io.write_create(
            self.file_store, target,
            metadata=metadata, content=content,
            overwrite=overwrite, force=force,
        )
        self.context.response.success = ok
        if ok:
            payload = {**payload, "path": str(target.resolve())}
        _set_answer(self.context, payload)


@R.register("memory_delete")
class MemoryDelete(BaseStep):
    """Delete a file. Watcher removes from store + graph."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "")
        assert path, "path is required"
        ok, payload = memory_io.write_delete(path)
        self.context.response.success = ok
        _set_answer(self.context, payload)


@R.register("memory_rename")
class MemoryRename(BaseStep):
    """Rename a file and rewrite incoming wikilinks across the vault."""

    async def execute(self):
        assert self.context is not None
        old_path: str = self.context.get("old_path", "")
        new_path: str = self.context.get("new_path", "")
        assert old_path and new_path, "old_path and new_path are required"

        watcher = self.app_context.components["file_watcher"]["default"]  # type: ignore[index,union-attr]
        vault_root = Path(watcher.watch_path).resolve()  # type: ignore[union-attr]

        ok, payload = memory_io.write_rename(self.file_store, vault_root, old_path, new_path)
        self.context.response.success = ok
        _set_answer(self.context, payload)


@R.register("memory_property_update")
class MemoryPropertyUpdate(BaseStep):
    """Update a single YAML frontmatter key. value=null deletes the key."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "")
        key: str = self.context.get("key", "")
        value = self.context.get("value")
        assert path and key, "path and key are required"
        ok, payload = memory_io.write_property_update(path, key, value)
        self.context.response.success = ok
        _set_answer(self.context, payload)


@R.register("memory_update")
class MemoryUpdate(BaseStep):
    """Edit-style content update: replace `old_string` with `new_string`."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "")
        old_string: str = self.context.get("old_string", "")
        new_string: str = self.context.get("new_string", "")
        replace_all: bool = bool(self.context.get("replace_all", False))
        assert path, "path is required"
        ok, payload = memory_io.write_update(path, old_string, new_string, replace_all=replace_all)
        self.context.response.success = ok
        _set_answer(self.context, payload)


@R.register("memory_archive")
class MemoryArchive(BaseStep):
    """Archive a file: flip `status: archived` then move to `<vault>/Archive/`."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        archive_dir_name: str = self.context.get("archive_dir", "Archive") or "Archive"
        assert path, "path is required"

        watcher = self._get_component_optional(ComponentEnum.FILE_WATCHER, "default")
        vault_root = Path(getattr(watcher, "watch_path", ".")).resolve() if watcher else Path.cwd()

        ok, payload = memory_io.write_archive(vault_root, path, archive_dir_name)
        self.context.response.success = ok
        _set_answer(self.context, payload)


@R.register("memory_count_tokens")
class MemoryCountTokens(BaseStep):
    """Estimate tokens for a file body or raw text. One of `path`/`text` required."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        text: str = self.context.get("text", "") or ""
        result = await memory_io.count_tokens(
            self.as_token_counter,
            path=path or None,
            text=text or None,
        )
        self.context.response.success = "error" not in result
        _set_answer(self.context, result)
