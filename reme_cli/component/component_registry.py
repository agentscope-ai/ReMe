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
        if not hasattr(cls, 'component_type'):
            raise TypeError(f"{cls.__name__} must have 'component_type' attribute")
        if not name:
            raise ValueError("Component name cannot be empty")

        component_type = cls.component_type
        if name in self._registry[component_type]:
            self.logger.warning(f"Component '{name}' already registered for {component_type}, overwriting")

        self._registry[component_type][name] = cls
        return cls

    def register(
            self, cls_or_name: type[T] | str, name: str | None = None
    ) -> Callable[[type[T]], type[T]] | type[T]:
        # Direct registration: R.register(MyClass, "name")
        if isinstance(cls_or_name, type):
            return self._do_register(cast(type[T], cls_or_name), name or cls_or_name.__name__)

        # Decorator mode: @R.register("name")
        decorator_name = cls_or_name

        def decorator(decorated_cls: type[T]) -> type[T]:
            return self._do_register(decorated_cls, decorator_name or decorated_cls.__name__)

        return decorator

    def get(self, component_type: ComponentEnum, name: str) -> type[BaseComponent] | None:
        return self._registry.get(component_type, {}).get(name)

    def get_all(self, component_type: ComponentEnum) -> dict[str, type[BaseComponent]]:
        return dict(self._registry.get(component_type, {}))

    def unregister(self, component_type: ComponentEnum, name: str) -> bool:
        if name in self._registry.get(component_type, {}):
            del self._registry[component_type][name]
            return True
        return False

    def clear(self) -> None:
        self._registry.clear()


R = ComponentRegistry()
