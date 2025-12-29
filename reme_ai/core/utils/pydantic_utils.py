from typing import Dict, Optional

from pydantic import create_model, Field

from . import snake_to_camel
from ..schema import ToolAttr, Request

TYPE_MAPPING = {
    "string": str,
    "array": list,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
}


def create_pydantic_model(flow_name: str, input_schema: Dict[str, ToolAttr] = None):
    fields = {}

    if input_schema:
        for param_name, param_config in input_schema.items():
            assert param_config.type in TYPE_MAPPING, f"flow_name={flow_name} had invalid type: {param_config.type}!"
            field_type = TYPE_MAPPING[param_config.type]

            if not param_config.required:
                fields[param_name] = (Optional[field_type], Field(default=None, description=param_config.description))
            else:
                fields[param_name] = (field_type, Field(default=..., description=param_config.description))

    return create_model(f"{snake_to_camel(flow_name)}Model", __base__=Request, **fields)
