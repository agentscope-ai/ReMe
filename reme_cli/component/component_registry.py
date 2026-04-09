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

T = TypeVar("T", bound=BaseComponent)


class ComponentRegistry:
    """Registry for managing component class registration and lookup.

    Components are organized by type (ComponentEnum), with each type
    containing multiple named component implementations.

    Attributes:
        _registry: Internal storage structure, format: {ComponentEnum: {name: component_class}}

    Usage:
        # Direct registration
        R.register(OpenAIChatModel, "openai")

        # Decorator registration
        @R.register("openai")
        class OpenAIChatModel(BaseComponent):
            ...

        # Get registered component
        cls = R.get(ComponentEnum.LLM, "openai")
    """

    def __init__(self) -> None:
        """Initialize the component registry."""
        self._registry: dict[ComponentEnum, dict[str, type[BaseComponent]]] = {}

    def _do_register(self, cls: type[T], name: str) -> type[T]:
        """Internal method to register a class.

        Args:
            cls: The component class to register.
            name: The name identifier for the component.

        Returns:
            The registered class.
        """
        component_type = cls.component_type
        if component_type not in self._registry:
            self._registry[component_type] = {}
        self._registry[component_type][name] = cls
        return cls

    def register(
        self, cls_or_name: type[T] | str, name: str | None = None
    ) -> Callable[[type[T]], type[T]] | type[T]:
        """Register a component class.

        Supports two calling patterns:
        - Direct: R.register(MyClass, "name") -> returns MyClass
        - Decorator: @R.register("name") -> returns decorator function

        Args:
            cls_or_name: Either a class to register (direct mode), or a name string
                for decorator mode.
            name: Optional name when using direct registration with a class.
                If not provided, the class name will be used.

        Returns:
            Either the registered class (direct mode) or a decorator function.

        Example:
            # Direct registration
            R.register(OpenAIChatModel, "openai")

            # Decorator registration
            @R.register("openai")
            class OpenAIChatModel(BaseComponent):
                ...
        """
        # Direct registration: R.register(MyClass, "name")
        if isinstance(cls_or_name, type):
            cls = cast(type[T], cls_or_name)
            _key = name or cls.__name__
            return self._do_register(cls, _key)

        # Decorator mode: @R.register("name")
        _decorator_name = cls_or_name

        def decorator(decorated_cls: type[T]) -> type[T]:
            key = _decorator_name or decorated_cls.__name__
            return self._do_register(decorated_cls, key)

        return decorator

    def get(self, component_type: ComponentEnum, name: str) -> type[BaseComponent] | None:
        """Get a registered component by type and name.

        Args:
            component_type: The component type enum value.
            name: The registered name of the component.

        Returns:
            The component class if found, None otherwise.
        """
        return self._registry.get(component_type, {}).get(name)

    def get_all(self, component_type: ComponentEnum) -> dict[str, type[BaseComponent]]:
        """Get all registered components for a given type.

        Args:
            component_type: The component type enum value.

        Returns:
            Dictionary mapping component names to their classes.
            Returns empty dict if no components registered for the type.
        """
        return self._registry.get(component_type, {})


# Global registry instance
R = ComponentRegistry()