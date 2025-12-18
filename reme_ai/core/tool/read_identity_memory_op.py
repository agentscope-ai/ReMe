from flowllm.core.storage import CacheHandler

from . import BaseMemoryToolOp
from .. import C, BaseAsyncToolOp
from ..schema import ToolCall


@C.register_op()
class ReadIdentityMemoryOp(BaseAsyncToolOp):
    """Utility operation that prompts the model for explicit reflection text."""
    def __init__(self, ):


    def build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "",
                "input_schema": {
                    "reflection": {
                        "type": "string",
                        "description": self.get_prompt("reflection"),
                        "required": True,
                    },
                },
            },
        )

    async def async_execute(self):
        cache: CacheHandler = CacheHandler()

        if self.add_output_reflection:
            self.set_output(self.input_dict["reflection"])
        else:
            self.set_output(self.get_prompt("reflection_output"))
