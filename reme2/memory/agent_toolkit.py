"""Agent toolkit — the 11 tools an agent uses to operate on the vault.

Three categories. Each tool is a single-purpose ``BaseStep`` exposing
two surfaces:

  * ``execute()`` — the MCP transport surface (reads
    ``RuntimeContext`` parameters, writes via ``_set_answer``).
  * a method named after the tool (e.g. ``memory_get``) — the
    agentscope toolkit surface; agentscope introspects the signature
    directly, no separate JSON schema.

Categories:

    Memory (5)  schema-bound markdown management
        memory_get / memory_create / memory_update_body /
        memory_update_meta / memory_search

    File (5)    type-agnostic vault transport + directory operations
        file_download / file_upload / file_delete / file_list /
        file_move

    Graph (1)   relationship exploration via BFS
        graph_traverse

`memory_graph_search` is also defined here as an MCP-only tool (no
agent toolkit method); it stays out of the 11-tool agent surface but
is registered for MCP/HTTP callers that want graph-aware retrieval.

Atomic maintenance/check tools live in ``lint_toolkit.py`` —
separate category, separate factory, NOT bound to the agent toolkit
by default.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import frontmatter
from agentscope.tool import Toolkit, ToolResponse

from . import memory_io
from ..component import R
from ..component.base_step import BaseStep
from .retriever import BaseRetriever, HybridRetriever
from .runtime_response import _set_answer, _tool_response, _to_jsonable
from ..enumeration import ComponentEnum


# ===========================================================================
# Section 1 — Schema policy (status state machine + path templates)
# ===========================================================================
#
# Used by memory_create (path template) and memory_update_meta (status
# state machine). Pure helpers; the gates fire only when force=False.


_STATUS_STATES = ("active", "distilled", "archived")
_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "active": {"active", "distilled"},
    "distilled": {"distilled", "archived"},
    "archived": {"archived"},
}


def validate_status_transition(prior, requested) -> str | None:
    """Return error string if the requested status transition is invalid."""
    if requested is None:
        return None
    if requested not in _STATUS_STATES:
        return f"invalid status {requested!r}; must be one of {list(_STATUS_STATES)}"
    if prior in _STATUS_STATES and requested not in _STATUS_TRANSITIONS[prior]:
        return (
            f"status transition {prior!r} → {requested!r} not allowed; "
            f"state machine is single-direction "
            f"active → distilled → archived"
        )
    return None


def validate_path_template(path: Path, working_dir: Path | None) -> str | None:
    """Return error string if `path` doesn't match an agent-facing template.

    Allowed templates (relative to working_dir):
        topics/{folder}/{name}.md         — topic file
        events/{date}/{name}/{filename}   — event index OR sibling material
        Archive/...                       — archive moves can land anywhere
    """
    if working_dir is None:
        return None
    try:
        rel = path.resolve().relative_to(working_dir)
    except ValueError:
        return f"path {path} is outside working_dir {working_dir}"
    parts = rel.parts
    if not parts:
        return "path has no components relative to working_dir"
    head = parts[0]
    if head == "Archive":
        return None
    if head == "topics" and len(parts) >= 3:
        return None
    if head == "events" and len(parts) >= 4:
        return None
    return (
        f"path {rel} doesn't match a known template — expected one of: "
        f"topics/{{folder}}/{{name}}.md, "
        f"events/{{date}}/{{name}}/{{filename}}, or Archive/..."
    )


def _update_status(path: Path | str, *, value, force: bool = False) -> tuple[bool, dict]:
    """Schema-aware status flip. Reads current status, validates the
    transition, then delegates to ``memory_io.update_meta``."""
    target = Path(path)
    if not force:
        prior = None
        if target.is_file():
            try:
                prior = frontmatter.loads(
                    target.read_text(encoding="utf-8"),
                ).metadata.get("status")
            except Exception:
                prior = None
        err = validate_status_transition(prior, value)
        if err is not None:
            return False, {
                "path": str(target),
                "key": "status",
                "error": err,
                "prior": prior,
                "requested": value,
            }
    return memory_io.update_meta(target, key="status", value=value)


def _create_with_schema(
    file_store,
    path: Path,
    *,
    metadata: dict,
    content: str,
    overwrite: bool = False,
    force: bool = False,
) -> tuple[bool, dict]:
    """Schema-aware file create — path template gate then engine."""
    if not force:
        working_dir = getattr(file_store, "working_dir", None)
        template_err = validate_path_template(path, working_dir)
        if template_err is not None:
            return False, {
                "path": str(path),
                "error": template_err,
                "hint": (
                    "place topics under topics/{folder}/{name}.md and "
                    "events under events/{date}/{name}/...; pass "
                    "force=true only if you intentionally need a "
                    "non-template path"
                ),
            }
    return memory_io.create_file(
        file_store, path,
        metadata=metadata, content=content,
        overwrite=overwrite, force=force,
    )


# ===========================================================================
# Section 2 — File-IO support (session temp dir + path resolution)
# ===========================================================================


_TEMP_ROOT: Path | None = None


def _get_temp_root() -> Path:
    """Lazy session-scoped temp dir. Auto-cleaned on process exit."""
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp(prefix="reme2-files-"))
    return _TEMP_ROOT


def _resolve_vault_path(file_store, vault_path: str) -> Path:
    """Compose the absolute on-disk path for a vault-relative entry."""
    working_dir = getattr(file_store, "working_dir", None) or "."
    p = Path(vault_path)
    if p.is_absolute():
        return p.resolve()
    return (Path(working_dir) / p).resolve()


# ===========================================================================
# Section 3 — Memory category (5 tools)
# ===========================================================================


@R.register("memory_get")
class MemoryGet(BaseStep):
    """Read a single memory file (frontmatter + body, optional chunks)."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        include_chunks: bool = bool(self.context.get("include_chunks", False))
        assert path, "path is required"
        result = await memory_io.get_file(self.file_store, path, include_chunks=include_chunks)
        _set_answer(self.context, result)

    async def memory_get(self, path: str, include_chunks: bool = False) -> ToolResponse:
        """Read a single memory file (frontmatter + body, optional chunks)."""
        result = await memory_io.get_file(self.file_store, path, include_chunks=include_chunks)
        return _tool_response("memory_get", True, result, audit=self.audit)


@R.register("memory_create")
class MemoryCreate(BaseStep):
    """Create a markdown file. Path-template gate + wikilink-uniqueness
    gate fire unless ``force=True``."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        metadata: dict = dict(self.context.get("metadata") or {})
        content: str = self.context.get("content", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", False))
        force: bool = bool(self.context.get("force", False))
        assert path, "path is required"
        target = Path(path)
        ok, payload = _create_with_schema(
            self.file_store, target,
            metadata=metadata, content=content,
            overwrite=overwrite, force=force,
        )
        self.context.response.success = ok
        if ok:
            payload = {**payload, "path": str(target.resolve())}
        _set_answer(self.context, payload)

    async def memory_create(
        self,
        path: str,
        metadata: dict | None = None,
        content: str = "",
        overwrite: bool = False,
        force: bool = False,
    ) -> ToolResponse:
        """Create a markdown file. Path template + wikilink uniqueness
        gates fire unless ``force=True``."""
        target = Path(path)
        ok, payload = _create_with_schema(
            self.file_store, target,
            metadata=dict(metadata or {}), content=content,
            overwrite=overwrite, force=force,
        )
        if ok:
            payload = {**payload, "path": str(target.resolve())}
        return _tool_response("memory_create", ok, payload, audit=self.audit)


@R.register("memory_update_body")
class MemoryUpdateBody(BaseStep):
    """Edit-style body update: replace ``old_string`` with ``new_string``.
    Frontmatter is preserved verbatim."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        old_string: str = self.context.get("old_string", "") or ""
        new_string: str = self.context.get("new_string", "") or ""
        replace_all: bool = bool(self.context.get("replace_all", False))
        assert path, "path is required"
        ok, payload = memory_io.update_body(
            path, old_string=old_string, new_string=new_string, replace_all=replace_all,
        )
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def memory_update_body(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResponse:
        """Edit-style body update: replace ``old_string`` with ``new_string``."""
        ok, payload = memory_io.update_body(
            path, old_string=old_string, new_string=new_string, replace_all=replace_all,
        )
        return _tool_response("memory_update_body", ok, payload, audit=self.audit)


@R.register("memory_update_meta")
class MemoryUpdateMeta(BaseStep):
    """Frontmatter patch (merge). value=None deletes the key.
    ``status`` transitions go through the state-machine validator
    unless ``force=True``."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        patch: dict = dict(self.context.get("patch") or {})
        force: bool = bool(self.context.get("force", False))
        assert path, "path is required"
        ok, payload = await self._apply(path, patch, force)
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def memory_update_meta(
        self,
        path: str,
        patch: dict,
        force: bool = False,
    ) -> ToolResponse:
        """Frontmatter patch (merge). value=None deletes the key."""
        ok, payload = await self._apply(path, dict(patch or {}), force)
        return _tool_response("memory_update_meta", ok, payload, audit=self.audit)

    async def _apply(self, path: str, patch: dict, force: bool) -> tuple[bool, dict]:
        results: dict[str, dict] = {}
        all_ok = True
        for key, value in patch.items():
            if key == "status":
                ok, payload = _update_status(path, value=value, force=force)
            else:
                ok, payload = memory_io.update_meta(path, key=key, value=value)
            results[key] = payload
            if not ok:
                all_ok = False
                break  # stop on first failure; partial state already on disk
        return all_ok, {"path": path, "applied": results}


# ----- memory_search (retrieval) ------------------------------------------


_RETRIEVER_CACHE: dict[int, BaseRetriever] = {}


def _resolve_retriever(step: BaseStep) -> BaseRetriever:
    """Get (or build) the retriever instance for this step."""
    cached = _RETRIEVER_CACHE.get(id(step))
    if cached is not None:
        return cached
    retriever = R.get(ComponentEnum.RETRIEVER, "hybrid")
    if retriever is None:
        retriever = HybridRetriever(app_context=step.app_context)
    elif isinstance(retriever, type):
        retriever = retriever(app_context=step.app_context)
    _RETRIEVER_CACHE[id(step)] = retriever
    return retriever


def _serialize_chunk(chunk, file_store, extras: dict | None = None) -> dict:
    """Flatten a FileChunk into a dict, joining file metadata."""
    item = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
    node = file_store.file_nodes.get(item.get("path"))
    if node is not None:
        meta = node.front_matter.model_dump()
        item["file_metadata"] = meta
        item["file_st_mtime"] = node.st_mtime
    else:
        item["file_metadata"] = {}
        item["file_st_mtime"] = None
    if extras:
        item.update(extras)
    return item


@R.register("memory_search")
class MemorySearch(BaseStep):
    """Pure-relevance retrieval (V + K hybrid). Delegates to the Retriever."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "").strip()
        min_score: float = self.context.get("min_score", 0.1)
        max_results: int = self.context.get("max_results", 5)
        assert query, "Query cannot be empty"
        assert 0.0 <= min_score <= 1.0, f"min_score must be in [0,1], got {min_score}"
        assert max_results > 0, f"max_results must be positive, got {max_results}"
        chunk_filter = memory_io.make_filter(
            self.file_store,
            paths=self.context.get("paths") or None,
            tags=self.context.get("tags") or None,
            exclude_paths=self.context.get("exclude_paths") or None,
        )
        retriever = _resolve_retriever(self)
        results = await retriever.search(
            query=query, max_results=max_results, min_score=min_score, chunk_filter=chunk_filter,
        )
        payload = [_serialize_chunk(r, self.file_store) for r in results]
        _set_answer(self.context, payload)

    async def memory_search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.1,
        paths: list[str] | None = None,
        tags: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> ToolResponse:
        """Pure-relevance retrieval (V + K hybrid)."""
        chunk_filter = memory_io.make_filter(
            self.file_store, paths=paths, tags=tags, exclude_paths=exclude_paths,
        )
        retriever = _resolve_retriever(self)
        results = await retriever.search(
            query=query, max_results=max_results, min_score=min_score, chunk_filter=chunk_filter,
        )
        payload = [_serialize_chunk(r, self.file_store) for r in results]
        return _tool_response("memory_search", True, payload, audit=self.audit)


# ===========================================================================
# Section 4 — File category (5 tools)
# ===========================================================================


@R.register("file_download")
class FileDownload(BaseStep):
    """Copy a vault file to a session temp dir; return the local path."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        vault_path: str = self.context.get("vault_path", "") or ""
        assert vault_path, "vault_path is required"
        payload = self._download(vault_path)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_download(self, vault_path: str) -> ToolResponse:
        """Copy a vault file to session temp dir; return the local path."""
        payload = self._download(vault_path)
        ok = "error" not in payload
        return _tool_response("file_download", ok, payload, audit=self.audit)

    def _download(self, vault_path: str) -> dict:
        src = _resolve_vault_path(self.file_store, vault_path)
        if not src.is_file():
            return {"vault_path": vault_path, "error": "not found"}
        dst_dir = Path(tempfile.mkdtemp(prefix="dl-", dir=_get_temp_root()))
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return {
            "vault_path": vault_path,
            "local_path": str(dst),
            "size": dst.stat().st_size,
        }


@R.register("file_upload")
class FileUpload(BaseStep):
    """Copy a local file into the vault. Watcher / parser register the
    FileNode asynchronously."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        local_path: str = self.context.get("local_path", "") or ""
        vault_path: str = self.context.get("vault_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", True))
        assert local_path and vault_path, "local_path and vault_path are required"
        payload = self._upload(local_path, vault_path, overwrite)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_upload(
        self, local_path: str, vault_path: str, overwrite: bool = True,
    ) -> ToolResponse:
        """Copy local_path into the vault at vault_path."""
        payload = self._upload(local_path, vault_path, overwrite)
        ok = "error" not in payload
        return _tool_response("file_upload", ok, payload, audit=self.audit)

    def _upload(self, local_path: str, vault_path: str, overwrite: bool) -> dict:
        src = Path(local_path)
        if not src.is_file():
            return {"local_path": local_path, "error": "source not found"}
        dst = _resolve_vault_path(self.file_store, vault_path)
        if dst.exists() and not overwrite:
            return {"vault_path": vault_path, "error": "destination exists; pass overwrite=True"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {
            "vault_path": vault_path,
            "size": dst.stat().st_size,
            "mime": mimetypes.guess_type(dst.name)[0] or "application/octet-stream",
        }


@R.register("file_delete")
class FileDelete(BaseStep):
    """Delete a vault file. Universal entry point for any file type."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        vault_path: str = self.context.get("vault_path", "") or ""
        assert vault_path, "vault_path is required"
        target = _resolve_vault_path(self.file_store, vault_path)
        ok, payload = memory_io.delete_file(target)
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def file_delete(self, vault_path: str) -> ToolResponse:
        """Delete a vault file."""
        target = _resolve_vault_path(self.file_store, vault_path)
        ok, payload = memory_io.delete_file(target)
        return _tool_response("file_delete", ok, payload, audit=self.audit)


@R.register("file_list")
class FileList(BaseStep):
    """Enumerate vault files with optional frontmatter filters."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        result = memory_io.list_files(
            self.file_store,
            path_prefix=self.context.get("prefix") or self.context.get("path_prefix"),
            tags=self.context.get("tags") or [],
            metadata=self.context.get("metadata") or {},
            limit=int(self.context.get("limit") or 100),
        )
        _set_answer(self.context, result)

    async def file_list(
        self,
        prefix: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        limit: int = 100,
    ) -> ToolResponse:
        """List vault files. Filters: path prefix, frontmatter tags / fields."""
        result = memory_io.list_files(
            self.file_store,
            path_prefix=prefix,
            tags=tags or [],
            metadata=metadata or {},
            limit=limit,
        )
        return _tool_response("file_list", True, result, audit=self.audit)


@R.register("file_move")
class FileMove(BaseStep):
    """Rename / relocate. Default leaves inbound wikilinks untouched
    (maintainer cleans dangling refs); pass ``update_refs=True`` to
    rewrite ``[[old]] → [[new]]`` in every referencing file."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        src: str = self.context.get("src") or self.context.get("old_path") or ""
        dst: str = self.context.get("dst") or self.context.get("new_path") or ""
        update_refs: bool = bool(self.context.get("update_refs", False))
        assert src and dst, "src and dst are required"
        payload = self._move(src, dst, update_refs)
        self.context.response.success = payload.get("ok", False)
        _set_answer(self.context, payload)

    async def file_move(
        self, src: str, dst: str, update_refs: bool = False,
    ) -> ToolResponse:
        """Rename / relocate. update_refs=True rewrites [[old]] → [[new]]."""
        payload = self._move(src, dst, update_refs)
        ok = payload.get("ok", False)
        return _tool_response("file_move", ok, payload, audit=self.audit)

    def _move(self, src: str, dst: str, update_refs: bool) -> dict:
        src_abs = _resolve_vault_path(self.file_store, src)
        dst_abs = _resolve_vault_path(self.file_store, dst)
        if not src_abs.is_file():
            return {"ok": False, "src": src, "error": "source not found"}
        if update_refs:
            working_dir = Path(getattr(self.file_store, "working_dir", None) or ".").resolve()
            ok, payload = memory_io.rename_file(
                self.file_store, working_dir,
                old_path=src_abs, new_path=dst_abs,
            )
            payload["ok"] = ok
            return payload
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_abs), str(dst_abs))
        return {"ok": True, "src": str(src_abs), "dst": str(dst_abs), "refs_updated": 0}


# ===========================================================================
# Section 5 — Graph category (1 tool)
# ===========================================================================


def _outlinks(file_store, path: str) -> list[tuple[str, str | None, str | None]]:
    """Outgoing edges from ``path`` — [(target_path, predicate, anchor)]."""
    node = file_store.file_nodes.get(path)
    if node is None:
        return []
    return [(link.path, link.predicate, link.anchor) for link in node.links if link.path]


def _inlinks(file_store, path: str) -> list[tuple[str, str | None, str | None]]:
    """Incoming edges to ``path`` — linear scan over all nodes' links.

    Cheap for vault sizes; if it ever becomes hot, swap for a precomputed
    reverse index on the file_graph component.
    """
    out: list[tuple[str, str | None, str | None]] = []
    for src_path, src_node in file_store.file_nodes.items():
        if src_path == path:
            continue
        for link in src_node.links:
            if link.path == path:
                out.append((src_path, link.predicate, link.anchor))
    return out


def _bfs_traverse(
    file_store,
    seeds: list[str],
    max_depth: int,
    direction: str,
    predicate: str | None,
) -> list[dict]:
    """BFS from each seed. One record per edge traversed."""
    visited_edges: set[tuple[str, str, str | None]] = set()
    results: list[dict] = []
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        edges: list[tuple[str, str | None, str | None]] = []
        if direction in ("out", "both"):
            for tgt, pred, anchor in _outlinks(file_store, current):
                if predicate is not None and pred != predicate:
                    continue
                edges.append((tgt, pred, anchor))
        if direction in ("in", "both"):
            for src, pred, anchor in _inlinks(file_store, current):
                if predicate is not None and pred != predicate:
                    continue
                edges.append((src, pred, anchor))
        for next_path, pred, anchor in edges:
            edge_key = (current, next_path, pred)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            results.append({
                "path": next_path,
                "depth": depth + 1,
                "via": current,
                "predicate": pred,
                "anchor": anchor,
            })
            if depth + 1 < max_depth:
                queue.append((next_path, depth + 1))
    return results


@R.register("graph_traverse")
class GraphTraverse(BaseStep):
    """BFS from seed(s) to explore relationships in the memory graph.

    Output: one record per edge traversed (same node may appear
    multiple times if reached via different predicates or paths).
    """

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        seeds_raw = self.context.get("seeds") or []
        if isinstance(seeds_raw, str):
            seeds = [seeds_raw]
        else:
            seeds = list(seeds_raw)
        max_depth: int = int(self.context.get("max_depth") or 1)
        direction: str = self.context.get("direction", "out") or "out"
        predicate = self.context.get("predicate")
        assert seeds, "seeds is required (single path or list of paths)"
        assert direction in ("out", "in", "both"), \
            f"direction must be 'out' | 'in' | 'both', got {direction!r}"
        results = _bfs_traverse(self.file_store, seeds, max_depth, direction, predicate)
        _set_answer(self.context, results)

    async def graph_traverse(
        self,
        seeds: str | list[str],
        max_depth: int = 1,
        direction: str = "out",
        predicate: str | None = None,
    ) -> ToolResponse:
        """BFS from seed(s). Args:
            seeds: single path or list to start from.
            max_depth: hops to expand (default 1 = immediate neighbors).
            direction: "out" / "in" / "both".
            predicate: filter edges by predicate (None = no filter).
        """
        if isinstance(seeds, str):
            seeds_list = [seeds]
        else:
            seeds_list = list(seeds)
        assert direction in ("out", "in", "both"), \
            f"direction must be 'out' | 'in' | 'both', got {direction!r}"
        results = _bfs_traverse(self.file_store, seeds_list, max_depth, direction, predicate)
        return _tool_response("graph_traverse", True, results, audit=self.audit)


# ===========================================================================
# Section 6 — Toolkit factory
# ===========================================================================


# The 11 tools the agent gets bound to. memory_graph_search stays
# registered for MCP/HTTP but is intentionally NOT in the agent surface
# (per-call retrieval-knob tuning is internal).
AGENT_TOOL_NAMES: tuple[str, ...] = (
    # memory (5)
    "memory_get",
    "memory_create",
    "memory_update_body",
    "memory_update_meta",
    "memory_search",
    # file (5)
    "file_download",
    "file_upload",
    "file_delete",
    "file_list",
    "file_move",
    # graph (1)
    "graph_traverse",
)


def build_agent_toolkit(
    app_context,
    audit: list[dict] | None = None,
    toolkit: Toolkit | None = None,
) -> Toolkit:
    """Bind every agent tool's method to an agentscope ``Toolkit``.

    For each name in ``AGENT_TOOL_NAMES``, instantiates the registered
    BaseStep against ``app_context``, attaches the shared ``audit``
    list, and registers the same-named class method as a tool function.
    agentscope introspects the method signature directly — no separate
    JSON schema layer.
    """
    toolkit = toolkit or Toolkit()
    for name in AGENT_TOOL_NAMES:
        step_cls = R.get(ComponentEnum.STEP, name)
        if step_cls is None:
            continue
        instance = step_cls(app_context=app_context)
        instance.audit = audit  # type: ignore[attr-defined]
        toolkit.register_tool_function(
            getattr(instance, name),
            namesake_strategy="override",
        )
    return toolkit
