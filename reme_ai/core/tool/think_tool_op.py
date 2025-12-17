from .. import C, BaseAsyncToolOp
from ..schema import ToolCall


@C.register_op()
class ThinkToolOp(BaseAsyncToolOp):
    """Utility operation that prompts the model for explicit reflection text."""

    def __init__(self, add_output_reflection: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.add_output_reflection: bool = add_output_reflection

    def build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "name": "think_tool",
                "description": self.get_prompt("tool"),
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
        if self.add_output_reflection:
            self.set_output(self.input_dict["reflection"])
        else:
            self.set_output(self.get_prompt("reflection_output"))
