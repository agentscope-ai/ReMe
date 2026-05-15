"""Lint toolkit — atomic vault-health checks.

Separate from the agent toolkit. These are read-only diagnostics for
maintainer / CLI / scheduled-job use; agents typically don't need
them in their per-call working set.

Four atomic checks, each does one thing and returns pure data:

    check_dangling     — FileLinks pointing to non-existent nodes
    check_orphans      — nodes with no inlinks AND no outlinks
    check_collisions   — basenames resolving to >1 path (short-link
                         ambiguity)
    check_schema       — nodes violating frontmatter schema
                         (missing required fields, invalid status)

Maintainer compositions live in ``maintainer.py``; this file is the
underlying primitives. Each tool is also independently MCP/agent
callable for ad-hoc checks.

Bind via ``build_lint_toolkit`` (parallel to ``build_agent_toolkit``).
"""

from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit, ToolResponse

from ..component import R
from ..component.base_step import BaseStep
from ..steps.runtime_response import _set_answer, _tool_response
from ..enumeration import ComponentEnum


# Frontmatter schema — required keys for a well-formed memory file.
# Mirrors the validator that fires inside `memory_create`. Keep in sync
# with the path-template + status-state-machine rules in
# ``agent_toolkit``.
_REQUIRED_META_KEYS = ("title", "lifecycle", "scope", "source", "role")
_VALID_STATUS = {"active", "distilled", "archived"}


# ===========================================================================
# Section 1 — Atomic check primitives (pure functions; no side effects)
# ===========================================================================


def _scan_dangling(file_store) -> list[dict]:
    """Find FileLinks pointing to nodes that don't exist in the index.

    Returns one entry per dangling edge:
        {"source": <path>, "target": <broken_link_path>,
         "predicate": <str|None>, "anchor": <str|None>}
    """
    nodes = file_store.file_nodes
    out: list[dict] = []
    for src_path, node in nodes.items():
        for link in node.links:
            if not link.path:
                continue
            if link.path not in nodes:
                out.append({
                    "source": src_path,
                    "target": link.path,
                    "predicate": link.predicate,
                    "anchor": link.anchor,
                })
    return out


def _scan_orphans(file_store) -> list[str]:
    """Find nodes with no outlinks AND no inlinks.

    O(N + E): one pass to mark which paths are referenced by anyone.
    """
    nodes = file_store.file_nodes
    referenced: set[str] = set()
    for node in nodes.values():
        for link in node.links:
            if link.path:
                referenced.add(link.path)
    out: list[str] = []
    for path, node in nodes.items():
        has_out = any(link.path for link in node.links)
        has_in = path in referenced
        if not has_out and not has_in:
            out.append(path)
    return sorted(out)


def _scan_collisions(file_store) -> dict[str, list[str]]:
    """Find basenames (filename + ext) that resolve to >1 path.

    Mirrors ``utils.wikilink_resolver.collisions`` but operates over
    the local file_store index (no async iteration needed)."""
    by_name: dict[str, list[str]] = {}
    for path in file_store.file_nodes:
        by_name.setdefault(Path(path).name, []).append(path)
    return {name: sorted(paths) for name, paths in by_name.items() if len(paths) > 1}


def _scan_schema(file_store) -> list[dict]:
    """Find nodes whose frontmatter violates schema.

    Each entry: {"path": <path>, "errors": [<error_msg>, ...]}.
    """
    out: list[dict] = []
    for path, node in file_store.file_nodes.items():
        meta = node.front_matter.model_dump()
        errs: list[str] = []
        missing = [k for k in _REQUIRED_META_KEYS if not meta.get(k)]
        if missing:
            errs.append(f"missing required: {missing}")
        status = meta.get("status")
        if status is not None and status not in _VALID_STATUS:
            errs.append(f"invalid status: {status!r} (expected one of {sorted(_VALID_STATUS)})")
        if errs:
            out.append({"path": path, "errors": errs})
    return out


# ===========================================================================
# Section 2 — Step wrappers (one per check)
# ===========================================================================


@R.register("check_dangling")
class CheckDangling(BaseStep):
    """List every FileLink whose target is not in the graph."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        findings = _scan_dangling(self.file_store)
        _set_answer(self.context, {"count": len(findings), "findings": findings})

    async def check_dangling(self) -> ToolResponse:
        """List every FileLink whose target is not in the graph."""
        findings = _scan_dangling(self.file_store)
        return _tool_response(
            "check_dangling", True,
            {"count": len(findings), "findings": findings},
            audit=self.audit,
        )


@R.register("check_orphans")
class CheckOrphans(BaseStep):
    """List nodes with no inlinks AND no outlinks."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        orphans = _scan_orphans(self.file_store)
        _set_answer(self.context, {"count": len(orphans), "paths": orphans})

    async def check_orphans(self) -> ToolResponse:
        """List nodes with no inlinks AND no outlinks."""
        orphans = _scan_orphans(self.file_store)
        return _tool_response(
            "check_orphans", True,
            {"count": len(orphans), "paths": orphans},
            audit=self.audit,
        )


@R.register("check_collisions")
class CheckCollisions(BaseStep):
    """List basenames resolving to >1 path (short-link ambiguity)."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        groups = _scan_collisions(self.file_store)
        _set_answer(self.context, {"count": len(groups), "groups": groups})

    async def check_collisions(self) -> ToolResponse:
        """List basenames resolving to >1 path (short-link ambiguity)."""
        groups = _scan_collisions(self.file_store)
        return _tool_response(
            "check_collisions", True,
            {"count": len(groups), "groups": groups},
            audit=self.audit,
        )


@R.register("check_schema")
class CheckSchema(BaseStep):
    """List nodes whose frontmatter violates the memory schema."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        findings = _scan_schema(self.file_store)
        _set_answer(self.context, {"count": len(findings), "findings": findings})

    async def check_schema(self) -> ToolResponse:
        """List nodes whose frontmatter violates the memory schema."""
        findings = _scan_schema(self.file_store)
        return _tool_response(
            "check_schema", True,
            {"count": len(findings), "findings": findings},
            audit=self.audit,
        )


# ===========================================================================
# Section 3 — Toolkit factory
# ===========================================================================


LINT_TOOL_NAMES: tuple[str, ...] = (
    "check_dangling",
    "check_orphans",
    "check_collisions",
    "check_schema",
)


def build_lint_toolkit(
    app_context,
    audit: list[dict] | None = None,
    toolkit: Toolkit | None = None,
) -> Toolkit:
    """Bind every check_* step's tool method to an agentscope ``Toolkit``.

    Parallel to ``build_agent_toolkit`` but separate — these are
    maintainer/CLI tools and aren't bound into the agent's working set
    by default. Hosts that want a single toolkit with everything can
    pass the result of ``build_agent_toolkit`` as the ``toolkit``
    argument here.
    """
    toolkit = toolkit or Toolkit()
    for name in LINT_TOOL_NAMES:
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
