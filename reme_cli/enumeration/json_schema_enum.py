"""Defines the standard data types supported by JSON Schema.

This enum maps common JSON Schema primitive types to their corresponding
Python runtime types, and provides a convenient string representation
compatible with JSON Schema (`"string"`, `"number"`, etc.).
"""

from enum import Enum


class JsonSchemaEnum(Enum):
    """Enumeration of valid JSON Schema data types.

    The enum value is the corresponding Python type, while the string
    representation (`str(...)`) is the canonical JSON Schema type name.
    """

    # Textual data
    STRING = str

    # Numeric values, including integers and floats
    NUMBER = float

    # Integer-only numeric values
    INTEGER = int

    # JSON objects (key-value mappings)
    OBJECT = dict

    # Ordered JSON lists/arrays
    ARRAY = list

    # Boolean values: true / false
    BOOLEAN = bool

    # Null / None values
    NULL = type(None)

    def __str__(self) -> str:
        """Return the lowercase JSON Schema type name for this enum member."""
        return self.name.lower()
