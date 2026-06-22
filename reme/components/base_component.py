"""Base class for components with async lifecycle and dependency injection."""

import asyncio
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from ..enumeration import ComponentEnum
from ..utils import get_logger

if TYPE_CHECKING:
    from .application_context import ApplicationContext

T = TypeVar("T", bound="BaseComponent")


class ComponentMixin:
    """Shared state for components and steps: identity, config, workspace paths."""

    component_type = ComponentEnum.BASE

    def __init__(
        self,
        name: str | None = None,
        backend: str = "",
        app_context: "ApplicationContext | None" = None,
        **kwargs,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.backend: str = backend
        self.app_context: "ApplicationContext | None" = app_context
        self.kwargs: dict = dict(kwargs)

        logger = get_logger()
        self.logger = logger.bind(component=self.name) if hasattr(logger, "bind") else logger

    @property
    def workspace_path(self) -> Path:
        """Absolute workspace root directory (cwd when no app_context is attached)."""
        if self.app_context is None:
            return Path.cwd()
        return Path(self.app_context.app_config.workspace_dir).absolute()

    def to_workspace_relative(self, path: str | Path) -> str:
        """Convert `path` to a workspace-relative string; return absolute path when outside."""
        abs_path = Path(path).absolute()
        try:
            return str(abs_path.relative_to(self.workspace_path))
        except ValueError:
            return str(abs_path)


class Dependency:
    """Placeholder returned by ``BaseComponent.bind`` for an unresolved dependency.

    Resolved into a real component (or None) when the owning component starts.
    Accessing any attribute before resolution raises a clear error.
    """

    __slots__ = ("ctype", "name", "default_factory", "optional")

    def __init__(
        self,
        ctype: ComponentEnum,
        name: str,
        default_factory: Callable[[], Any] | None = None,
        optional: bool = True,
    ) -> None:
        self.ctype = ctype
        self.name = name
        self.default_factory = default_factory
        self.optional = optional

    def __repr__(self) -> str:
        suffix = "?" if self.optional else ""
        return f"<unresolved {self.ctype.value}:{self.name}{suffix}>"

    def __getattr__(self, item: str) -> Any:
        # Catches accidental use of the placeholder before start() resolves it.
        raise RuntimeError(
            f"Dependency {self.ctype.value}:{self.name} accessed before start() " f"(attribute '{item}')",
        )


class BaseComponent(ComponentMixin, ABC):
    """Async lifecycle base class with bind-based dependency injection."""

    component_type = ComponentEnum.BASE

    def __init__(
        self,
        name: str | None = None,
        backend: str = "",
        app_context: "ApplicationContext | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, backend=backend, app_context=app_context, **kwargs)

        self._is_started: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        # Components created via bind() default_factory in standalone mode;
        # their lifecycle is owned by this component.
        self._owned: list["BaseComponent"] = []

    @property
    def is_started(self) -> bool:
        """Whether the component has been started."""
        return self._is_started

    # ----- Dependency declaration ----------------------------------------

    @staticmethod
    def bind(
        name: str | None,
        base_cls: type[T],
        *,
        default_factory: Callable[[], T] | None = None,
        optional: bool = True,
    ) -> T | None:
        """Declare a dependency on another component.

        Returns a ``Dependency`` placeholder resolved into the real component
        (or None / a factory-produced instance) when ``start`` runs. An empty
        `name` short-circuits to None so callers can skip optional wiring.
        """
        if not name:
            return None
        ctype = getattr(base_cls, "component_type", None)
        if not isinstance(ctype, ComponentEnum) or ctype is ComponentEnum.BASE:
            raise TypeError(
                f"{base_cls.__name__} must declare a non-BASE ComponentEnum 'component_type'",
            )
        return cast(T, Dependency(ctype, name, default_factory, optional))

    @property
    def dependencies(self) -> list[Dependency]:
        """All unresolved dependency placeholders on this instance."""
        return [v for v in self.__dict__.values() if isinstance(v, Dependency)]

    async def _resolve_bindings(self) -> None:
        """Replace every ``Dependency`` attribute with its resolved target."""
        for attr, dep in list(self.__dict__.items()):
            if isinstance(dep, Dependency):
                self._resolve_one(attr, dep)

    def _resolve_one(self, attr: str, dep: Dependency) -> None:
        """Resolve a single dependency, dispatching by mode."""
        if self.app_context is None:
            self._resolve_standalone(attr, dep)
        else:
            self._resolve_from_context(attr, dep)

    def _resolve_standalone(self, attr: str, dep: Dependency) -> None:
        """Standalone mode: use default_factory, or fall back to None when optional.

        Required dependencies without a factory keep the placeholder so any
        attribute access surfaces a clear error at the call site.
        """
        if dep.default_factory is not None:
            instance = dep.default_factory()
            setattr(self, attr, instance)
            if isinstance(instance, BaseComponent):
                self._owned.append(instance)
        elif dep.optional:
            setattr(self, attr, None)

    def _resolve_from_context(self, attr: str, dep: Dependency) -> None:
        """Context-bound mode: look up the component from ``app_context.components``."""
        target = self.app_context.components.get(dep.ctype, {}).get(dep.name)
        if target is not None:
            setattr(self, attr, target)
        elif dep.optional:
            setattr(self, attr, None)
        else:
            raise ValueError(f"{dep.ctype.value} '{dep.name}' not found.")

    # ----- Workspace path helpers --------------------------------------------

    @property
    def workspace_metadata_path(self) -> Path:
        """Workspace metadata directory: ``<workspace>/<metadata_dir>``."""
        if self.app_context is None:
            return Path.cwd() / "metadata"
        return self.workspace_path / self.app_context.app_config.metadata_dir

    @property
    def component_metadata_path(self) -> Path:
        """Per-component metadata directory under the workspace."""
        return self.workspace_metadata_path / self.component_type.value

    # ----- Lifecycle hooks (override in subclasses) ----------------------

    async def _start(self) -> None:
        """Subclass hook called once after dependencies are resolved."""

    async def _close(self) -> None:
        """Subclass hook called once during ``close``."""

    async def dump(self) -> None:
        """Persist in-memory state to disk. Override when persistence is needed."""

    async def load(self) -> None:
        """Restore in-memory state from disk. Override when persistence is needed."""

    # ----- Lifecycle control --------------------------------------------

    async def start(self) -> None:
        """Start the component once: resolve deps → start owned → run _start."""
        async with self._lock:
            if self._is_started:
                return
            await self._resolve_bindings()
            for owned in self._owned:
                await owned.start()
            await self._start()
            self._is_started = True

    async def close(self) -> None:
        """Close the component once: run _close → close owned in reverse order."""
        async with self._lock:
            if not self._is_started:
                return
            first_error: BaseException | None = None
            try:
                await self._close()
            except BaseException as exc:
                first_error = exc
            finally:
                for owned in reversed(self._owned):
                    try:
                        await owned.close()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                        else:
                            self.logger.exception(f"Failed to close owned component {owned.name}: {exc}")
                self._is_started = False
            if first_error is not None:
                raise first_error

    async def restart(self) -> None:
        """Close then start the component."""
        await self.close()
        await self.start()

    async def __call__(self, **kwargs):
        raise NotImplementedError

    async def __aenter__(self) -> "BaseComponent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
