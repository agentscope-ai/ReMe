"""Event category — task workspace lifecycle.

Two tools:

    event_open      open or create today's event workspace; create the
                    folder-note index following the memory schema
    event_complete  close the workspace; mode='ingest' marks distilled
                    (and triggers the Ingestor service when wired);
                    mode='abandon' marks archived

The event index file ``events/{date}/{name}/{name}.md`` is created
through the same ``create_with_schema`` gate as ``memory_create`` — it
follows memory's frontmatter schema, status state machine, and path
template. The body starts empty; the agent maintains it as the working
context summary (Plan / Progress / Findings / Decisions / Next Steps)
via ``memory_update_body`` over the lifetime of the task.

Workspace materials (anything else under the event folder) accumulate
through ``file_upload``. By convention, agent puts user-supplied files
under ``user_uploads/`` so the Ingestor can distinguish ground truth
from agent-derived artifacts purely by path.

Status lifecycle (reuses memory state machine):

    active     — workspace open, agent writing; ingest gate accepts
    distilled  — ingest succeeded; topics updated, workspace frozen
    archived   — workspace abandoned (mode=abandon, force-set since
                 active → archived isn't a normal transition)
"""

from __future__ import annotations

from datetime import date as _date
from pathlib import Path

from agentscope.tool import ToolResponse

from ..component import R
from ..component.base_step import BaseStep
from .file_toolkit import resolve_vault_path
from .memory_toolkit import create_with_schema, update_status
from .runtime_response import _set_answer, _tool_response


# ===========================================================================
# Section 1 — Helpers
# ===========================================================================


def _today_str() -> str:
    return _date.today().isoformat()


def _index_relpath(date: str, name: str) -> str:
    """``events/{date}/{name}/{name}.md`` — folder-note convention."""
    return f"events/{date}/{name}/{name}.md"


def _resolve_event_index(file_store, name_or_path: str) -> Path:
    """Accept either a bare event name or an absolute / vault-relative
    path to the index file. Bare names are interpreted under today."""
    if "/" in name_or_path or name_or_path.endswith(".md"):
        return resolve_vault_path(file_store, name_or_path)
    return resolve_vault_path(file_store, _index_relpath(_today_str(), name_or_path))


def _allocate_name(file_store, date: str, name: str, force_new: bool) -> tuple[str, bool]:
    """Resolve the event name to use, handling collisions.

    Returns ``(final_name, created_now)``. When ``force_new`` is False
    and the workspace already exists, returns the existing name with
    ``created_now=False`` (idempotent open). When ``force_new`` is True,
    suffixes ``-2``, ``-3``, ... until a free slot is found.
    """
    candidate = name
    base = resolve_vault_path(file_store, _index_relpath(date, candidate))
    if not base.exists():
        return candidate, True
    if not force_new:
        return candidate, False
    suffix = 2
    while True:
        candidate = f"{name}-{suffix}"
        probe = resolve_vault_path(file_store, _index_relpath(date, candidate))
        if not probe.exists():
            return candidate, True
        suffix += 1


# Default frontmatter for the event index. Agent-curated workspaces
# default to lifecycle=streaming + role=observation + scope=instance +
# source=agent — these are the values memory_create's schema validator
# accepts for event-shaped files (see test fixtures in test_expert).
_EVENT_DEFAULTS = {
    "lifecycle": "streaming",
    "scope": "instance",
    "source": "agent",
    "role": "observation",
    "status": "active",
}


def _build_metadata(name: str, intent: str, related_topics: list[str] | None) -> dict:
    meta: dict = {
        "title": name,
        **_EVENT_DEFAULTS,
    }
    if intent:
        meta["intent"] = intent
    if related_topics:
        meta["related_topics"] = list(related_topics)
    return meta


# ===========================================================================
# Section 2 — Event tools
# ===========================================================================


@R.register("event_open")
class EventOpen(BaseStep):
    """Open or create today's event workspace.

    Idempotent: same name returns the existing workspace with
    ``created=False``. ``force_new=True`` allocates a fresh suffix
    instead. The folder-note index is created through the memory
    schema gate (path template + frontmatter validation)."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        name: str = self.context.get("name", "") or ""
        intent: str = self.context.get("intent", "") or ""
        related_topics = self.context.get("related_topics") or []
        date: str = self.context.get("date") or _today_str()
        force_new: bool = bool(self.context.get("force_new", False))
        assert name, "name is required"
        payload = self._open(name, intent, list(related_topics), date, force_new)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def event_open(
        self,
        name: str,
        intent: str = "",
        related_topics: list[str] | None = None,
        date: str | None = None,
        force_new: bool = False,
    ) -> ToolResponse:
        """Open or create today's event workspace.

        Args:
            name: workspace name (event id within the daily bucket).
            intent: short statement of the task purpose (lands in
                index frontmatter for the Ingestor to read).
            related_topics: optional list of topic paths the agent
                expects this task to touch.
            date: defaults to today; pass YYYY-MM-DD to target a
                specific daily bucket.
            force_new: if True and ``name`` is taken, allocate a fresh
                ``name-N`` suffix; otherwise return the existing workspace.

        Returns ``{path, name, created, date}``. Agent then uploads
        materials via ``file_upload`` under the returned path, with
        ``user_uploads/`` reserved for user-supplied files.
        """
        payload = self._open(
            name, intent, list(related_topics or []),
            date or _today_str(), force_new,
        )
        ok = "error" not in payload
        return _tool_response("event_open", ok, payload, audit=self.audit)

    def _open(
        self,
        name: str,
        intent: str,
        related_topics: list[str],
        date: str,
        force_new: bool,
    ) -> dict:
        final_name, created_now = _allocate_name(self.file_store, date, name, force_new)
        index_relpath = _index_relpath(date, final_name)
        if not created_now:
            existing = resolve_vault_path(self.file_store, index_relpath)
            return {
                "path": str(existing.parent),
                "index": str(existing),
                "name": final_name,
                "date": date,
                "created": False,
            }
        target = resolve_vault_path(self.file_store, index_relpath)
        metadata = _build_metadata(final_name, intent, related_topics)
        ok, payload = create_with_schema(
            self.file_store, target,
            metadata=metadata, content="",
            overwrite=False, force=False,
        )
        if not ok:
            return {
                "name": final_name,
                "date": date,
                "error": payload.get("error", "create failed"),
                "detail": payload,
            }
        return {
            "path": str(target.parent),
            "index": str(target),
            "name": final_name,
            "date": date,
            "created": True,
        }


@R.register("event_complete")
class EventComplete(BaseStep):
    """Close the event workspace.

    ``mode="ingest"`` flips status active → distilled (and is the
    hook for Ingestor invocation when wired). ``mode="abandon"``
    marks the workspace archived (force-set since active → archived
    isn't a normal state machine transition).

    Returns the event index path and (when ingest is wired) the list
    of topics updated by the Ingestor.
    """

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        name_or_path: str = (
            self.context.get("name") or self.context.get("path") or ""
        )
        mode: str = self.context.get("mode", "ingest") or "ingest"
        assert name_or_path, "name or path is required"
        assert mode in ("ingest", "abandon"), \
            f"mode must be 'ingest' or 'abandon', got {mode!r}"
        payload = await self._complete(name_or_path, mode)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def event_complete(
        self,
        name: str,
        mode: str = "ingest",
    ) -> ToolResponse:
        """Close the event workspace.

        Args:
            name: workspace name (under today) OR vault-relative /
                absolute path to the index file.
            mode: ``"ingest"`` (default) flips active → distilled and
                triggers ingestion; ``"abandon"`` marks archived.

        Returns ``{event_path, status, distilled, updated_topics}``.
        """
        assert mode in ("ingest", "abandon"), \
            f"mode must be 'ingest' or 'abandon', got {mode!r}"
        payload = await self._complete(name, mode)
        ok = "error" not in payload
        return _tool_response("event_complete", ok, payload, audit=self.audit)

    async def _complete(self, name_or_path: str, mode: str) -> dict:
        index_path = _resolve_event_index(self.file_store, name_or_path)
        if not index_path.is_file():
            return {
                "name": name_or_path,
                "error": "event index not found",
                "expected_at": str(index_path),
            }
        if mode == "abandon":
            ok, payload = update_status(index_path, value="archived", force=True)
            return {
                "event_path": str(index_path),
                "status": "archived" if ok else None,
                "distilled": False,
                "updated_topics": [],
                **({"error": payload.get("error")} if not ok else {}),
            }
        # mode == "ingest"
        # Ingestor invocation hook — when the Ingestor service exposes a
        # callable interface for "ingest one workspace", trigger it here
        # and capture updated_topics. Until then we just flip status to
        # distilled so downstream tools see the workspace as closed.
        ok, payload = update_status(index_path, value="distilled", force=False)
        if not ok:
            return {
                "event_path": str(index_path),
                "status": None,
                "distilled": False,
                "updated_topics": [],
                "error": payload.get("error", "status transition failed"),
                "detail": payload,
            }
        return {
            "event_path": str(index_path),
            "status": "distilled",
            "distilled": True,
            "updated_topics": [],  # populated when Ingestor is wired in
        }


EVENT_TOOL_NAMES: tuple[str, ...] = (
    "event_open",
    "event_complete",
)
