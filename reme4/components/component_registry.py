"""Global registry mapping ``(ComponentEnum, name) -> component class``."""

from typing import Callable, TypeVar, cast

from .base_component import BaseComponent
from ..enumeration import ComponentEnum
from ..utils import get_logger

T = TypeVar("T", bound=BaseComponent)


class ComponentRegistry:
    """Two-level registry: ``component_type -> name -> class``.

    Supports both direct calls — ``R.register(MyClass, "name")`` — and
    decorator usage — ``@R.register("name")``.
    """

    def __init__(self) -> None:
        self._registry: dict[ComponentEnum, dict[str, type[BaseComponent]]] = {}
        self.logger = get_logger()

    def _do_register(self, cls: type[T], name: str) -> type[T]:
        """Insert `cls` under its ``component_type`` group; warn on overwrite."""
        component_type = getattr(cls, "component_type", None)
        if not isinstance(component_type, ComponentEnum):
            raise TypeError(
                f"{cls.__name__} must have a ComponentEnum 'component_type' attribute",
            )
        if not name:
            raise ValueError("Component name cannot be empty")

        group = self._registry.setdefault(component_type, {})
        if name in group:
            self.logger.warning(
                f"Component '{name}' already registered for {component_type}, overwriting",
            )
        group[name] = cls
        return cls

    def register(
        self,
        cls_or_name: type[T] | str,
        name: str | None = None,
    ) -> Callable[[type[T]], type[T]] | type[T]:
        """Register a component class directly, or return a decorator that does so."""
        # Direct call: register(MyClass) or register(MyClass, "alias").
        if isinstance(cls_or_name, type):
            cls = cast(type[T], cls_or_name)
            return self._do_register(cls, name if name is not None else cls.__name__)

        # Decorator call: @R.register("alias") — must receive a string name.
        if not isinstance(cls_or_name, str):
            raise TypeError(f"Expected a class or string, got {type(cls_or_name).__name__}")

        registration_name = cls_or_name

        def decorator(decorated_cls: type[T]) -> type[T]:
            return self._do_register(decorated_cls, registration_name)

        return decorator

    def get(self, component_type: ComponentEnum, name: str) -> type[BaseComponent] | None:
        """Look up a registered class; return None if not found."""
        return self._registry.get(component_type, {}).get(name)

    def get_all(self, component_type: ComponentEnum) -> dict[str, type[BaseComponent]]:
        """Return a shallow copy of all classes registered under `component_type`."""
        return dict(self._registry.get(component_type, {}))

    def unregister(self, component_type: ComponentEnum, name: str) -> bool:
        """Remove an entry; return True if it existed, False otherwise."""
        if (group := self._registry.get(component_type)) and name in group:
            del group[name]
            return True
        return False

    def clear(self) -> None:
        """Drop every registered entry."""
        self._registry.clear()


# Process-wide singleton used throughout the codebase.
R = ComponentRegistry()
