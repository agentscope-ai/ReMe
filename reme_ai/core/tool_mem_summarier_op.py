from . import BaseAsyncToolOp, C
from .schema import ToolCall


@C.register_op()
class MetaMemSummarizerOp(BaseAsyncToolOp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_tool_call(self) -> ToolCall:
        return ToolCall(**{
            "description": "" 
            ""
        })

    async def async_execute(self): ...
