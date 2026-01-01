"""Defines the standard data types supported by JSON Schema."""

from enum import Enum


class JsonSchemaEnum(Enum):
    """Enumeration of valid JSON Schema data types."""

    STRING = str
    NUMBER = float
    INTEGER = int
    OBJECT = dict
    ARRAY = list
    BOOLEAN = bool

    def __str__(self) -> str:
        """Returns the string representation of the enum value."""
        return self.name.lower()
