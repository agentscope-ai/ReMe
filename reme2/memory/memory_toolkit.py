"""Memory toolkit — schema-bound projection of the Memory File System.

Layered on top of `reme2.memory.memory_io` (the pure core engine). One
`BaseStep` subclass per memory_* tool, each exposing TWO class methods:

  * `execute()` — the MCP surface. Reads parameters from
    `RuntimeContext` and writes the result through `_set_answer`.
  * a method named after the tool (e.g. `memory_get`) — the agent
    toolkit surface. Takes explicit parameters, returns a
    `ToolResponse`. agentscope's `Toolkit.register_tool_function`
    introspects the signature directly — no hand-authored JSON schema.

`build_memory_toolkit(app_context, audit, toolkit)` instantiates every
registered memory_* step and binds its tool method to a `Toolkit`.
Each instance carries an `audit` list so the host can surface what the
agent actually called.

Schema policy (status state machine, path templates) lives at the top
of this file as pure helpers; the relevant write tools (`memory_create`
/ `memory_property_update`) call them. `force=True` is the single
escape hatch — bypasses BOTH the policy gates here and the
wikilink-uniqueness gate downstream in `memory_io.create_file`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import frontmatter
from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from . import memory_io
from ..component import R
from ..component.base_step import BaseStep
from ..component.runtime_response import _set_answer, _to_jsonable
from ..enumeration import ComponentEnum


# ===========================================================================
# Section 1 — Schema policy
# ===========================================================================


_STATUS_STATES = ("active", "distilled", "archived")
_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "active": {"active", "distilled"},
    "distilled": {"distilled", "archived"},
    "archived": {"archived"},
}


def validate_status_transition(prior, requested) -> str | None:
    """Return error string if the requested status transition is invalid,
    else None. Files without a prior status accept any initial value
    (so first-write doesn't get blocked)."""
    if requested is None:
        return None  # delete operation
    if requested not in _STATUS_STATES:
        return f"invalid status {requested!r}; must be one of " f"{list(_STATUS_STATES)}"
    if prior in _STATUS_STATES and requested not in _STATUS_TRANSITIONS[prior]:
        return (
            f"status transition {prior!r} → {requested!r} not allowed; "
            f"state machine is single-direction "
            f"active → distilled → archived"
        )
    return None


def validate_path_template(path: Path, working_dir: Path | None) -> str | None:
    """Return error string if `path` doesn't match a known agent-facing
    template, else None. When `working_dir` is unknown, skip the check.

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


def update_status(
    path: Path | str,
    *,
    value,
    force: bool = False,
) -> tuple[bool, dict]:
    """Schema-aware status flip. Reads current status from disk, validates
    the transition, then delegates to `memory_io.update_meta`."""
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


def create_file_with_schema(
    file_store,
    path: Path,
    *,
    metadata: dict,
    content: str,
    overwrite: bool = False,
    force: bool = False,
) -> tuple[bool, dict]:
    """Schema-aware file create. Validates the path template (unless
    `force=True`), then delegates to `memory_io.create_file` (which
    still enforces the wikilink-uniqueness graph invariant — that gate
    lives in the engine because it's structural, not business policy).

    `force=True` bypasses BOTH the template gate here and the wikilink
    gate downstream — it's the single escape hatch for any caller that
    intentionally needs to step outside conventions.
    """
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
        file_store,
        path,
        metadata=metadata,
        content=content,
        overwrite=overwrite,
        force=force,
    )


# ===========================================================================
# Section 2 — Tool response helper
# ===========================================================================


def _tool_response(
    op: str,
    ok: bool,
    payload: Any,
    audit: list[dict] | None = None,
) -> ToolResponse:
    """Wrap a tool-method result as a `ToolResponse` and optionally
    append an audit row."""
    if audit is not None:
        entry = {"op": op, "ok": ok}
        if isinstance(payload, dict):
            entry.update(payload)
        else:
            entry["result"] = payload
        audit.append(entry)
    text = json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)
    return ToolResponse(content=[TextBlock(type="text", text=text)])


# ===========================================================================
# Section 3 — Memory steps (one BaseStep per tool, two surfaces each)
# ===========================================================================


@R.register("memory_get")
class MemoryGet(BaseStep):
    """Read a single memory file (frontmatter + body, optional chunks)."""

    audit: list[dict] | None = None  # set by build_memory_toolkit

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


@R.register("memory_list")
class MemoryList(BaseStep):
    """List indexed files filtered by frontmatter fields, tags, or path prefix."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        result = memory_io.list_files(
            self.file_store,
            path_prefix=self.context.get("path_prefix"),
            tags=self.context.get("tags") or [],
            metadata=self.context.get("metadata") or {},
            limit=int(self.context.get("limit") or 100),
        )
        _set_answer(self.context, result)

    async def memory_list(
        self,
        path_prefix: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        limit: int = 100,
    ) -> ToolResponse:
        """List indexed files filtered by frontmatter fields, tags, or path prefix."""
        result = memory_io.list_files(
            self.file_store,
            path_prefix=path_prefix,
            tags=tags or [],
            metadata=metadata or {},
            limit=limit,
        )
        return _tool_response("memory_list", True, result, audit=self.audit)


@R.register("memory_backlinks")
class MemoryBacklinks(BaseStep):
    """Files linking TO a given path. Each entry carries the typed-edge predicate."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        _set_answer(self.context, memory_io.get_backlinks(self.file_store, path))

    async def memory_backlinks(self, path: str) -> ToolResponse:
        """Files linking TO a given path. Each entry carries the typed-edge predicate."""
        result = memory_io.get_backlinks(self.file_store, path)
        return _tool_response("memory_backlinks", True, result, audit=self.audit)


@R.register("memory_links")
class MemoryLinks(BaseStep):
    """Files a given path links to (resolved). Each entry carries the typed-edge predicate."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        _set_answer(self.context, memory_io.get_links(self.file_store, path))

    async def memory_links(self, path: str) -> ToolResponse:
        """Files a given path links to (resolved). Each entry carries the typed-edge predicate."""
        result = memory_io.get_links(self.file_store, path)
        return _tool_response("memory_links", True, result, audit=self.audit)


@R.register("memory_resolve_wikilink")
class MemoryResolveWikilink(BaseStep):
    """Resolve a `[[target]]` wikilink with full ambiguity context."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        wikilink: str = self.context.get("wikilink", "") or ""
        assert wikilink, "wikilink is required"
        payload = memory_io.resolve_wikilink(self.file_store, wikilink)
        self.context.response.success = bool(payload.get("exists"))
        _set_answer(self.context, payload)

    async def memory_resolve_wikilink(self, wikilink: str) -> ToolResponse:
        """Resolve a `[[target]]` wikilink with full ambiguity context."""
        payload = memory_io.resolve_wikilink(self.file_store, wikilink)
        return _tool_response(
            "memory_resolve_wikilink",
            bool(payload.get("exists")),
            payload,
            audit=self.audit,
        )


@R.register("memory_create")
class MemoryCreate(BaseStep):
    """Create a markdown file. Path-template gate + wikilink-uniqueness
    gate both fire unless `force=True`."""

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
        ok, payload = create_file_with_schema(
            self.file_store,
            target,
            metadata=metadata,
            content=content,
            overwrite=overwrite,
            force=force,
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
        """Create a markdown file. Path-template gate + wikilink-uniqueness
        gate both fire unless `force=True`."""
        target = Path(path)
        ok, payload = create_file_with_schema(
            self.file_store,
            target,
            metadata=dict(metadata or {}),
            content=content,
            overwrite=overwrite,
            force=force,
        )
        if ok:
            payload = {**payload, "path": str(target.resolve())}
        return _tool_response("memory_create", ok, payload, audit=self.audit)


@R.register("memory_delete")
class MemoryDelete(BaseStep):
    """Delete a file. Watcher removes from store + graph."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        ok, payload = memory_io.delete_file(path)
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def memory_delete(self, path: str) -> ToolResponse:
        """Delete a file. Watcher removes from store + graph."""
        ok, payload = memory_io.delete_file(path)
        return _tool_response("memory_delete", ok, payload, audit=self.audit)


@R.register("memory_rename")
class MemoryRename(BaseStep):
    """Rename a file and rewrite incoming wikilinks across the vault."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        old_path: str = self.context.get("old_path", "") or ""
        new_path: str = self.context.get("new_path", "") or ""
        assert old_path and new_path, "old_path and new_path are required"

        working_dir = Path(self.file_store.working_dir or Path.cwd()).resolve()

        ok, payload = memory_io.rename_file(
            self.file_store,
            working_dir,
            old_path=old_path,
            new_path=new_path,
        )
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def memory_rename(self, old_path: str, new_path: str) -> ToolResponse:
        """Rename a file and rewrite incoming wikilinks across the vault."""
        vr = getattr(self.file_store, "working_dir", None)
        working_dir = Path(vr).resolve() if vr else Path.cwd()
        ok, payload = memory_io.rename_file(
            self.file_store,
            working_dir,
            old_path=old_path,
            new_path=new_path,
        )
        return _tool_response("memory_rename", ok, payload, audit=self.audit)


@R.register("memory_property_update")
class MemoryPropertyUpdate(BaseStep):
    """Update a single YAML frontmatter key. value=null deletes the key.

    When key=='status', enforces the active → distilled → archived
    state machine via `update_status`. Pass `force=True` to bypass.
    Other keys go through the bare engine."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        key: str = self.context.get("key", "") or ""
        value = self.context.get("value")
        force: bool = bool(self.context.get("force", False))
        assert path and key, "path and key are required"

        if key == "status":
            ok, payload = update_status(path, value=value, force=force)
        else:
            ok, payload = memory_io.update_meta(path, key=key, value=value)
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def memory_property_update(
        self,
        path: str,
        key: str,
        value: Any = None,
        force: bool = False,
    ) -> ToolResponse:
        """Update a single YAML frontmatter key. value=null deletes the key.

        When key=='status', enforces the active → distilled → archived
        state machine. Pass `force=True` to bypass."""
        if key == "status":
            ok, payload = update_status(path, value=value, force=force)
        else:
            ok, payload = memory_io.update_meta(path, key=key, value=value)
        return _tool_response("memory_property_update", ok, payload, audit=self.audit)


@R.register("memory_update")
class MemoryUpdate(BaseStep):
    """Edit-style content update: replace `old_string` with `new_string`."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        old_string: str = self.context.get("old_string", "") or ""
        new_string: str = self.context.get("new_string", "") or ""
        replace_all: bool = bool(self.context.get("replace_all", False))
        assert path, "path is required"
        ok, payload = memory_io.update_body(
            path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def memory_update(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResponse:
        """Edit-style content update: replace `old_string` with `new_string`."""
        ok, payload = memory_io.update_body(
            path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        return _tool_response("memory_update", ok, payload, audit=self.audit)


@R.register("memory_archive")
class MemoryArchive(BaseStep):
    """Archive a file: flip `status: archived` then move to `<vault>/Archive/`."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        archive_dir_name: str = self.context.get("archive_dir", "Archive") or "Archive"
        assert path, "path is required"

        vr = getattr(self.file_store, "working_dir", None)
        working_dir = Path(vr).resolve() if vr else Path.cwd()

        ok, payload = memory_io.archive_file(working_dir, path, archive_dir=archive_dir_name)
        self.context.response.success = ok
        _set_answer(self.context, payload)

    async def memory_archive(self, path: str, archive_dir: str = "Archive") -> ToolResponse:
        """Archive a file: flip `status: archived` then move to `<vault>/<archive_dir>/`."""
        vr = getattr(self.file_store, "working_dir", None)
        working_dir = Path(vr).resolve() if vr else Path.cwd()
        ok, payload = memory_io.archive_file(working_dir, path, archive_dir=archive_dir)
        return _tool_response("memory_archive", ok, payload, audit=self.audit)


@R.register("memory_count_tokens")
class MemoryCountTokens(BaseStep):
    """Estimate tokens for a file body or raw text. One of `path`/`text` required.

    MCP-only — token counting is an editor concern, not part of the
    R-M-W loop. No agent toolkit projection."""

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


# ===========================================================================
# Section 4 — Toolkit factory
# ===========================================================================


# The 11 memory_* tools the Ingestor's ReActAgent gets bound to.
# Each name doubles as the BaseStep registration key AND the tool method
# name on that step.
MEMORY_TOOL_NAMES: tuple[str, ...] = (
    "memory_get",
    "memory_list",
    "memory_resolve_wikilink",
    "memory_backlinks",
    "memory_links",
    "memory_create",
    "memory_update",
    "memory_property_update",
    "memory_rename",
    "memory_delete",
    "memory_archive",
)


def build_memory_toolkit(
    app_context,
    audit: list[dict] | None = None,
    toolkit: Toolkit | None = None,
) -> Toolkit:
    """Bind every memory_* step's tool method to an agentscope `Toolkit`.

    For each name in `MEMORY_TOOL_NAMES`, instantiates the registered
    BaseStep against `app_context`, attaches the shared `audit` list,
    and registers the same-named class method as a tool function.
    agentscope introspects the method signature directly — there is no
    separate JSON schema layer.
    """
    toolkit = toolkit or Toolkit()
    for name in MEMORY_TOOL_NAMES:
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
