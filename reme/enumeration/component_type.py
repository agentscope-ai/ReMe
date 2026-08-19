"""Extensible component type identifiers."""

import re
from typing import TypeAlias

from .component_enum import ComponentEnum

ComponentType: TypeAlias = ComponentEnum | str
_COMPONENT_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")


def component_type_name(value: ComponentType) -> str:
    """Return the canonical string name for a built-in or plugin component type."""
    if isinstance(value, ComponentEnum):
        return value.value
    if not isinstance(value, str):
        raise TypeError(f"Component type must be a string, got {type(value).__name__}")
    name = value.strip()
    if not name:
        raise ValueError("Component type cannot be empty")
    if _COMPONENT_TYPE_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"Invalid component type {name!r}; use lowercase letters and numbers separated by '.', '_' or '-'",
        )
    return name
