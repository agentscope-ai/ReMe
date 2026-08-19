"""Registry mapping ``(ComponentEnum, backend)`` to implementation classes."""

from typing import Callable, TypeVar, cast

from .base_component import ComponentMixin
from ..enumeration import ComponentEnum

T = TypeVar("T", bound=ComponentMixin)


class ComponentRegistry:
    """Two-level registry: ``component_type -> name -> class``.

    Supports both direct calls — ``R.register(MyClass, "name")`` — and
    decorator usage — ``@R.register("name")``.
    """

    def __init__(self) -> None:
        self._registry: dict[ComponentEnum, dict[str, type[ComponentMixin]]] = {}
        self._owners: dict[tuple[ComponentEnum, str], str] = {}

    def _do_register(self, cls: type[T], name: str, *, owner: str | None = None) -> type[T]:
        """Insert ``cls`` under its component type and reject ambiguous providers."""
        component_type = getattr(cls, "component_type", None)
        if not isinstance(component_type, ComponentEnum):
            raise TypeError(
                f"{cls.__name__} must have a ComponentEnum 'component_type' attribute",
            )
        if not name:
            raise ValueError("Component name cannot be empty")

        group = self._registry.setdefault(component_type, {})
        key = (component_type, name)
        if name in group:
            existing = group[name]
            existing_owner = self._owners[key]
            new_owner = owner or cls.__module__
            if existing is cls and existing_owner == new_owner:
                return cls
            raise ValueError(
                f"Backend '{component_type.value}:{name}' is provided by both " f"'{existing_owner}' and '{new_owner}'",
            )
        group[name] = cls
        self._owners[key] = owner or cls.__module__
        return cls

    def add(self, name: str, cls: type[T], *, owner: str) -> type[T]:
        """Register one explicitly owned plugin contribution."""
        return self._do_register(cls, name, owner=owner)

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

    def get(self, component_type: ComponentEnum, name: str) -> type[ComponentMixin] | None:
        """Look up a registered class; return None if not found."""
        return self._registry.get(component_type, {}).get(name)

    def get_all(self, component_type: ComponentEnum) -> dict[str, type[ComponentMixin]]:
        """Return a shallow copy of all classes registered under `component_type`."""
        return dict(self._registry.get(component_type, {}))

    def unregister(self, component_type: ComponentEnum, name: str) -> bool:
        """Remove an entry; return True if it existed, False otherwise."""
        if (group := self._registry.get(component_type)) and name in group:
            del group[name]
            self._owners.pop((component_type, name), None)
            return True
        return False

    def clear(self) -> None:
        """Drop every registered entry."""
        self._registry.clear()
        self._owners.clear()

    def copy(self) -> "ComponentRegistry":
        """Return an independent registry containing the same providers."""
        copied = ComponentRegistry()
        for component_type, group in self._registry.items():
            for name, implementation in group.items():
                copied.add(name, implementation, owner=self._owners[(component_type, name)])
        return copied


# Process-wide singleton used throughout the codebase.
R = ComponentRegistry()
