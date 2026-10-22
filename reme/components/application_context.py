"""Application context: shared state container for components, jobs, and service."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..schema import ApplicationConfig, WorkspaceContext
from .component_registry import ComponentRegistry, create_application_registry

if TYPE_CHECKING:
    from .base_component import BaseComponent
    from .job import BaseJob
    from .service import BaseService


class ApplicationContext:
    """Passive state container holding parsed config and wired components.

    The Application class performs the actual wiring (registry lookups and
    component instantiation); this class only stores the results so that
    components, jobs, and the service can find each other at runtime.
    """

    def __init__(self, *, registry: ComponentRegistry | None = None, **kwargs):
        # Parse raw kwargs into a typed, validated config object.
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)
        self.registry = registry or create_application_registry()

        # Ordered per-workspace path views; the first entry is the default workspace.
        self.workspaces: dict[str, WorkspaceContext] = {
            alias: WorkspaceContext.from_config(alias, path, self.app_config)
            for alias, path in self.app_config.workspaces.items()
        }
        self.workspace: WorkspaceContext = next(iter(self.workspaces.values()))

        # Populated by Application during initialization.
        self.service: "BaseService | None" = None
        self.components: dict[str, dict[str, "BaseComponent"]] = {}
        self.jobs: dict[str, "BaseJob"] = {}
        self.thread_pool: ThreadPoolExecutor | None = None

        # Application-lifetime shared state. Values remain available across Job and Step
        # invocations while this Application is running, so Jobs may keep cross-call state here.
        # This is in-memory state, not durable storage; use workspace files or a store when state
        # must survive an Application restart.
        self.metadata: dict[str, Any] = {}

    @property
    def workspace_components(self) -> dict[str, dict[str, dict[str, "BaseComponent"]]]:
        """Compatibility view of scoped instances, ctype -> name -> alias.

        The authoritative owner of these instances is each WorkspaceContext;
        this view exists for transitional readers.
        """
        grouped: dict[str, dict[str, dict[str, "BaseComponent"]]] = {}
        for alias, workspace in self.workspaces.items():
            for ctype, group in workspace.components.items():
                for name, instance in group.items():
                    grouped.setdefault(ctype, {}).setdefault(name, {})[alias] = instance
        return grouped

    def get_component(self, ctype: str, name: str, alias: str | None = None) -> "BaseComponent | None":
        """Resolve a component, honoring workspace bindings for scoped components.

        Returns None when the component is not configured at all. Raises
        KeyError when a workspace-scoped component has no instance for the
        requested alias — a missing scoped resource must fail, never fall back.
        """
        alias = alias or next(iter(self.workspaces))
        workspace = self.workspaces.get(alias)
        if workspace is None:
            raise KeyError(f"Unknown workspace alias {alias!r}")
        if name in workspace.components.get(ctype, {}):
            return workspace.components[ctype][name]
        if any(name in ws.components.get(ctype, {}) for ws in self.workspaces.values()):
            raise KeyError(f"Component '{name}' in {ctype} has no instance for workspace alias {alias!r}")
        return self.components.get(ctype, {}).get(name)
