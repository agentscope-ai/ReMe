"""Per-workspace path views and owned state for multi-workspace applications."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .application_config import ApplicationConfig


@dataclass
class WorkspaceAgentState:
    """Session execution state owned by one agent wrapper within one workspace."""

    thread_tool_contexts: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkspaceCodexThreadState:
    """Authoritative Codex thread ownership shared by every wrapper in a workspace."""

    owned_thread_ids: set[str] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class WorkspaceContext:
    """Owns the paths, stateful components, execution state, and locks of one workspace."""

    alias: str
    workspace_path: Path
    metadata_path: Path | None
    session_path: Path | None
    mem_session_path: Path | None
    resource_path: Path | None
    daily_path: Path | None
    digest_path: Path | None
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_contexts: dict[str, Any] = field(default_factory=dict)
    search_call_budgets: dict[str, Any] = field(default_factory=dict)
    agent_states: dict[str, WorkspaceAgentState] = field(default_factory=dict)
    codex_thread_state: WorkspaceCodexThreadState = field(default_factory=WorkspaceCodexThreadState)
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def agent_state(self, wrapper_name: str) -> WorkspaceAgentState:
        """Return the named wrapper's state within this workspace."""
        return self.agent_states.setdefault(wrapper_name, WorkspaceAgentState())

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

    def get_component(self, ctype: str, name: str) -> Any:
        """Return a component owned by this workspace, never a default or shared substitute."""
        try:
            return self.components[ctype][name]
        except KeyError as exc:
            raise KeyError(f"Component '{name}' in {ctype} has no instance for workspace '{self.alias}'") from exc

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
