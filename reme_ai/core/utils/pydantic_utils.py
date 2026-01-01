"""
Utility module for dynamic Pydantic model generation based on schema definitions.
"""

from typing import Any, Literal

from pydantic import create_model, Field

from . import snake_to_camel
from ..enumeration import JsonSchemaEnum
from ..schema import ToolAttr, Request

TYPE_MAPPING = {str(t): t.value for t in JsonSchemaEnum}


def create_pydantic_model(name: str, parameters: ToolAttr | None = None) -> type[Request]:
    """
    Recursively generates a Pydantic model from a ToolAttr schema definition.
    """
    fields = {}

    if not parameters or not parameters.properties:
        return create_model(f"{snake_to_camel(name)}Model", __base__=Request)

    for field_name, attr in parameters.properties.items():
        # 1. Determine the base field type
        if attr.type == "object" and attr.properties:
            # Handle nested objects recursively
            field_type = create_pydantic_model(field_name, attr)

        elif attr.type == "array" and attr.items:
            # Handle array/list types
            if isinstance(attr.items, ToolAttr):
                if attr.items.type == "object":
                    inner_type = create_pydantic_model(f"{field_name}_item", attr.items)
                else:
                    inner_type = TYPE_MAPPING.get(attr.items.type, Any)
                field_type = list[inner_type]
            else:
                # Fallback for simple dictionary item definitions
                field_type = list[Any]

        else:
            # Handle primitive types
            field_type = TYPE_MAPPING.get(attr.type, Any)

        # 2. Handle enumeration constraints
        if attr.enum:
            # Dynamically create a Literal type from the enum list
            field_type = Literal[tuple(attr.enum)]  # type: ignore

        # 3. Determine requirement status and default values
        is_required = False
        if parameters.required and field_name in parameters.required:
            is_required = True

        # 4. Construct Field metadata
        field_info = Field(default=... if is_required else None, description=attr.description)

        if not is_required:
            field_type = field_type | None

        fields[field_name] = (field_type, field_info)

    # Dynamically construct the final Pydantic model class
    return create_model(f"{snake_to_camel(name)}Model", **fields, __base__=Request)
