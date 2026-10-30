"""Application context: shared state container for components, jobs, and service."""

import asyncio
from collections.abc import Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..schema import ApplicationConfig, WorkspaceContext
from .component_registry import ComponentRegistry, create_application_registry

if TYPE_CHECKING:
    from .base_component import BaseComponent
    from .job import BaseJob
    from .service import BaseService


class _LegacyComponentView(MutableMapping):
    """Compatibility mapping combining shared components with default-workspace instances."""

    def __init__(self, context: "ApplicationContext") -> None:
        self._context = context

    def _default_components(self) -> dict:
        return next(iter(self._context.workspaces.values())).components

    def __getitem__(self, key: str) -> dict:
        if key in self._context.shared_components:
            return self._context.shared_components[key]
        return self._default_components()[key]

    def __setitem__(self, key: str, value: dict) -> None:
        if value and all(getattr(component, "workspace_scoped", False) is True for component in value.values()):
            self._default_components()[key] = value
            self._context.shared_components.pop(key, None)
        else:
            self._context.shared_components[key] = value
            self._default_components().pop(key, None)

    def __delitem__(self, key: str) -> None:
        if key in self._context.shared_components:
            del self._context.shared_components[key]
        else:
            del self._default_components()[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._context.shared_components
        yield from (key for key in self._default_components() if key not in self._context.shared_components)

    def __len__(self) -> int:
        return len(set(self._context.shared_components) | set(self._default_components()))


class ApplicationContext:
    """Application-wide configuration, shared resources, and workspace registry."""

    def __init__(self, *, registry: ComponentRegistry | None = None, **kwargs):
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)
        self.registry = registry or create_application_registry()
        self.workspaces: dict[str, WorkspaceContext] = {
            alias: WorkspaceContext.from_config(alias, path, self.app_config)
            for alias, path in self.app_config.workspaces.items()
        }

        self.workspace: WorkspaceContext = next(iter(self.workspaces.values()))
        self.service: "BaseService | None" = None
        self.shared_components: dict[str, dict[str, "BaseComponent"]] = {}
        self._components = _LegacyComponentView(self)
        self.jobs: dict[str, "BaseJob"] = {}
        self.thread_pool: ThreadPoolExecutor | None = None
        self.codex_thread_ownership_lock = asyncio.Lock()
        self.metadata: dict[str, Any] = {}

    @property
    def components(self) -> MutableMapping:
        """Legacy component lookup; internal execution uses explicit shared or workspace ownership."""
        return self._components

    @components.setter
    def components(self, value: dict) -> None:
        self.shared_components = {}
        for ctype, group in value.items():
            self._components[ctype] = group

    def get_component(self, ctype: str, name: str, alias: str | None = None) -> "BaseComponent | None":
        """Compatibility resolver for external callers that select a workspace by alias."""
        alias = alias or next(iter(self.workspaces))
        workspace = self.workspaces.get(alias)
        if workspace is None:
            raise KeyError(f"Unknown workspace alias {alias!r}")
        if name in workspace.components.get(ctype, {}):
            return workspace.get_component(ctype, name)
        if any(name in ws.components.get(ctype, {}) for ws in self.workspaces.values()):
            raise KeyError(f"Component '{name}' in {ctype} has no instance for workspace alias {alias!r}")
        return self.shared_components.get(ctype, {}).get(name)
