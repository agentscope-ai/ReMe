"""Memory category — schema-bound markdown management + hybrid retrieval.

Five tools:

    memory_get          read structured (frontmatter + body + links)
    memory_create       schema-validated create
    memory_update_body  Edit-style body replacement
    memory_update_meta  frontmatter patch (status state machine enforced)
    memory_search       V+K hybrid retrieval

Plus ``memory_graph_search`` (MCP-only — no agent toolkit method, but
registered so HTTP/MCP callers can still tune graph fusion knobs).

Schema policy lives here — ``validate_status_transition``,
``validate_path_template``, ``_create_with_schema``, ``_update_status``
— and is reused by ``event_toolkit`` so event index files go through
the same gates as memory_create.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
from agentscope.tool import ToolResponse

from . import memory_io
from ..component import R
from ..component.base_step import BaseStep
from ..enumeration import ComponentEnum
from .retriever import BaseRetriever, HybridRetriever
from ..steps.runtime_response import _set_answer, _tool_response


# ===========================================================================
# Section 1 — Schema policy (status state machine + path templates)
# ===========================================================================
#
# Used by memory_create (path template), memory_update_meta (status
# state machine), and event_open (both, since the event index file
# follows the same memory schema). Pure helpers; the gates fire only
# when force=False.


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


def validate_path_template(path: Path, working_dir: Path | str | None) -> str | None:
    """Return error string if `path` doesn't match an agent-facing template.

    Allowed templates (relative to working_dir):
        topics/{folder}/{name}.md         — topic file
        events/{date}/{name}/{filename}   — event index OR sibling material
        Archive/...                       — archive moves can land anywhere
    """
    if working_dir is None:
        return None
    vault = Path(working_dir).resolve()
    try:
        rel = path.resolve().relative_to(vault)
    except ValueError:
        return f"path {path} is outside working_dir {vault}"
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


def update_status(path: Path | str, *, value, force: bool = False) -> tuple[bool, dict]:
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


def create_with_schema(
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
# Section 2 — Memory tools
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
        ok, payload = create_with_schema(
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
        ok, payload = create_with_schema(
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
                ok, payload = update_status(path, value=value, force=force)
            else:
                ok, payload = memory_io.update_meta(path, key=key, value=value)
            results[key] = payload
            if not ok:
                all_ok = False
                break
        return all_ok, {"path": path, "applied": results}


# ===========================================================================
# Section 3 — Retrieval (memory_search + memory_graph_search)
# ===========================================================================


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


@R.register("memory_graph_search")
class MemoryGraphSearch(BaseStep):
    """V + K + graph BFS fusion. MCP-only — not bound to agent toolkit.

    Per-call overrides for fusion knobs are forwarded if present in
    the RuntimeContext."""

    _OVERRIDE_KEYS = (
        "vector_weight", "graph_weight", "graph_depth", "graph_decay",
        "graph_direction", "graph_mode", "graph_per_path_cap", "anchor_expand",
    )

    async def execute(self):
        assert self.context is not None
        ctx = self.context
        query: str = ctx.get("query", "").strip()
        max_results: int = int(ctx.get("max_results", 5))
        min_score: float = float(ctx.get("min_score", 0.0))
        explicit_seeds: list[str] = list(ctx.get("seeds") or [])
        assert query or explicit_seeds, "query or seeds must be provided"
        assert max_results > 0
        chunk_filter = memory_io.make_filter(
            self.file_store,
            paths=ctx.get("paths") or None,
            tags=ctx.get("tags") or None,
            exclude_paths=ctx.get("exclude_paths") or None,
        )
        overrides = {k: ctx.get(k) for k in self._OVERRIDE_KEYS if ctx.get(k) is not None}
        retriever = _resolve_retriever(self)
        results = await retriever.search(
            query=query, max_results=max_results, min_score=min_score,
            chunk_filter=chunk_filter, seeds=explicit_seeds or None, **overrides,
        )
        payload = [_serialize_chunk(r, self.file_store) for r in results]
        _set_answer(self.context, payload)


MEMORY_TOOL_NAMES: tuple[str, ...] = (
    "memory_get",
    "memory_create",
    "memory_update_body",
    "memory_update_meta",
    "memory_search",
)
