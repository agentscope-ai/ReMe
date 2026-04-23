"""
Component registry module.

Provides a global registry for managing component class registration and lookup.
Supports two registration methods:
1. Direct registration: R.register(MyClass, "name")
2. Decorator registration: @R.register("name")
"""

from typing import Callable, TypeVar, cast

from .base_component import BaseComponent
from ..enumeration import ComponentEnum
from ..utils import get_logger

T = TypeVar("T", bound=BaseComponent)


class ComponentRegistry:
    """Registry for managing component class registration and lookup."""

    def __init__(self) -> None:
        self._registry: dict[ComponentEnum, dict[str, type[BaseComponent]]] = {}
        self.logger = get_logger()

    def _do_register(self, cls: type[T], name: str) -> type[T]:
        """Register a component class with the given name."""
        component_type = getattr(cls, "component_type", None)
        if not isinstance(component_type, ComponentEnum):
            raise TypeError(f"{cls.__name__} must have a ComponentEnum 'component_type' attribute")
        if not name:
            raise ValueError("Component name cannot be empty")

        group = self._registry.setdefault(component_type, {})
        if name in group:
            self.logger.warning(f"Component '{name}' already registered for {component_type}, overwriting")
        group[name] = cls
        return cls

    def register(
            self,
            cls_or_name: type[T] | str,
            name: str | None = None,
    ) -> Callable[[type[T]], type[T]] | type[T]:
        """Register a component class. Supports direct and decorator modes."""
        # Direct registration: R.register(MyClass, "name")
        if isinstance(cls_or_name, type):
            return self._do_register(cast(type[T], cls_or_name), name if name is not None else cls_or_name.__name__)

        # Decorator mode: @R.register("name")
        if not isinstance(cls_or_name, str):
            raise TypeError(f"Expected a class or string, got {type(cls_or_name).__name__}")

        def decorator(decorated_cls: type[T]) -> type[T]:
            return self._do_register(decorated_cls, cls_or_name)

        return decorator

    def get(self, component_type: ComponentEnum, name: str) -> type[BaseComponent] | None:
        """Get a registered component class by type and name."""
        return self._registry.get(component_type, {}).get(name)

    def get_all(self, component_type: ComponentEnum) -> dict[str, type[BaseComponent]]:
        """Get all registered components of a given type."""
        return dict(self._registry.get(component_type, {}))

    def unregister(self, component_type: ComponentEnum, name: str) -> bool:
        """Remove a component from the registry. Returns True if found."""
        if (group := self._registry.get(component_type)) and name in group:
            del group[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered components."""
        self._registry.clear()


R = ComponentRegistry()
