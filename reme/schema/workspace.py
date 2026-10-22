"""Per-workspace path views and owned state for multi-workspace applications."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .application_config import ApplicationConfig


@dataclass(frozen=True)
class WorkspaceContext:
    """Owns everything scoped to one workspace of an application.

    Built from ApplicationConfig when the application context is created.
    Beyond the resolved paths, the workspace owns its stateful component
    instances (stores, graphs, indexes, catalogs), in-memory execution state
    (drafts, search dedup, call budgets), session ownership records, and
    workspace-scoped locks. Shared, stateless clients stay on the
    ApplicationContext; per-call bindings reference this object.
    """

    alias: str
    workspace_path: Path
    metadata_path: Path | None
    session_path: Path | None
    mem_session_path: Path | None
    resource_path: Path | None
    daily_path: Path | None
    digest_path: Path | None
    # Workspace-scoped component instances keyed by component type then name.
    # Values are BaseComponent instances; typed as Any to keep this module
    # free of component imports (schema must not depend on components).
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Drafts and search dedup state, keyed by tool_context_id.
    tool_contexts: dict[str, Any] = field(default_factory=dict)
    # Search call budgets keyed by tool_context_id.
    search_call_budgets: dict[str, Any] = field(default_factory=dict)
    # Workspace-scoped named asyncio locks (e.g. dream integration).
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    @classmethod
    def from_config(cls, alias: str, workspace: str, config: "ApplicationConfig") -> "WorkspaceContext":
        """Resolve one alias of the config's workspaces mapping into absolute paths."""
        root = Path(workspace).absolute()

        def child(subdir: str) -> Path | None:
            return root / subdir if subdir else None

        return cls(
            alias=alias,
            workspace_path=root,
            metadata_path=child(config.metadata_dir),
            session_path=child(config.session_dir),
            mem_session_path=child(config.mem_session_dir),
            resource_path=child(config.resource_dir),
            daily_path=child(config.daily_dir),
            digest_path=child(config.digest_dir),
        )

    def storage_dirs(self) -> list[Path]:
        """Subdirectories the application materializes at startup, in configuration order."""
        return [
            path
            for path in (
                self.metadata_path,
                self.session_path,
                self.mem_session_path,
                self.resource_path,
                self.daily_path,
                self.digest_path,
            )
            if path is not None
        ]

    def lock(self, name: str) -> asyncio.Lock:
        """Return the workspace-scoped lock for ``name``, created lazily."""
        return self.locks.setdefault(name, asyncio.Lock())
