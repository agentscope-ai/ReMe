"""Base class for components."""

import asyncio
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from ..enumeration import ComponentEnum
from ..utils import get_logger

if TYPE_CHECKING:
    from .application_context import ApplicationContext

T = TypeVar("T", bound="BaseComponent")


class Dependency:
    """Declared dependency: bind() return value, instance attribute placeholder, and topological-sort edge."""

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
        # Guard against using the dependency before start() resolves it.
        raise RuntimeError(
            f"Dependency {self.ctype.value}:{self.name} accessed before start() (attribute '{item}')",
        )


class BaseComponent(ABC):
    """Async lifecycle base class with bind-based dependency injection."""

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
        self.logger = get_logger()
        if hasattr(self.logger, "bind"):
            self.logger = self.logger.bind(component=self.name)

        self._is_started: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        # Components created from bind() default_factory in standalone mode (auto-managed lifecycle).
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
        """Declare a dependency on another component; resolved at start(). Empty name → None."""
        if not name:
            return None
        ctype = getattr(base_cls, "component_type", None)
        if not isinstance(ctype, ComponentEnum) or ctype is ComponentEnum.BASE:
            raise TypeError(f"{base_cls.__name__} must declare a non-BASE ComponentEnum 'component_type'")
        return cast(T, Dependency(ctype, name, default_factory, optional))

    @property
    def dependencies(self) -> list[Dependency]:
        """All unresolved bindings declared on this instance."""
        return [v for v in self.__dict__.values() if isinstance(v, Dependency)]

    async def _resolve_bindings(self) -> None:
        """Replace Dependency placeholders with real components (or default_factory / None for optional)."""
        for attr, value in list(self.__dict__.items()):
            if not isinstance(value, Dependency):
                continue
            if self.app_context is None:
                # Standalone mode: factory or (optional → None) or keep placeholder.
                if value.default_factory is not None:
                    instance = value.default_factory()
                    setattr(self, attr, instance)
                    if isinstance(instance, BaseComponent):
                        self._owned.append(instance)
                elif value.optional:
                    setattr(self, attr, None)
            else:
                target = self.app_context.components.get(value.ctype, {}).get(value.name)
                if target is not None:
                    setattr(self, attr, target)
                elif value.optional:
                    setattr(self, attr, None)
                else:
                    raise ValueError(f"{value.ctype.value} '{value.name}' not found.")

    # ----- Lookup --------------------------------------------------------

    @property
    def vault_path(self) -> Path:
        """Resolved vault root path from app context or cwd."""
        if self.app_context is None:
            return Path.cwd()
        return Path(self.app_context.app_config.vault_dir).absolute()

    @property
    def vault_metadata_path(self) -> Path:
        """Resolved metadata directory: vault_path / metadata_dir, or absolute metadata_dir."""
        if self.app_context is None:
            return Path.cwd() / "metadata"
        return self.vault_path / self.app_context.app_config.metadata_dir

    def to_vault_relative(self, path: str | Path) -> str:
        """Return path relative to vault_path; absolute path string if outside."""
        abs_path = Path(path).absolute()
        try:
            return str(abs_path.relative_to(self.vault_path))
        except ValueError:
            return str(abs_path)

    # ----- Lifecycle -----------------------------------------------------

    async def _start(self) -> None:
        """Subclass hook: start logic."""

    async def _close(self) -> None:
        """Subclass hook: close logic."""

    async def dump(self) -> None:
        """Persist in-memory state to disk. Override in subclasses that need persistence."""

    async def load(self) -> None:
        """Restore in-memory state from disk. Override in subclasses that need persistence."""

    async def start(self) -> None:
        """Resolve bindings → start owned fallbacks → _start(). No-op if already started."""
        async with self._lock:
            if self._is_started:
                return
            await self._resolve_bindings()
            for owned in self._owned:
                await owned.start()
            await self._start()
            self._is_started = True

    async def close(self) -> None:
        """_close() → close owned fallbacks in reverse. No-op if not started."""
        async with self._lock:
            if not self._is_started:
                return
            await self._close()
            for owned in reversed(self._owned):
                await owned.close()
            self._is_started = False

    async def restart(self) -> None:
        """Close then start."""
        await self.close()
        await self.start()

    async def __call__(self, **kwargs):
        raise NotImplementedError

    async def __aenter__(self) -> "BaseComponent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
