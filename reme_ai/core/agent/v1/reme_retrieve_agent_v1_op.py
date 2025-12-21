from typing import List, Dict

from flowllm.core.op import BaseAsyncToolOp

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ...enumeration import Role
from ...schema import Message, ToolCall


@C.register_op()
class ReMeRetrieveAgentV1Op(BaseMemoryAgentOp):

    def __init__(
            self,
            enable_tool_memory: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.enable_tool_memory = enable_tool_memory

    def build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "input_schema": {
                    "workspace_id": {
                        "type": "string",
                        "description": "workspace_id",
                        "required": True,
                    },
                    "query": {
                        "type": "string",
                        "description": "query",
                        "required": False,
                    },
                    "messages": {
                        "type": "array",
                        "description": "messages",
                        "required": False,
                        "items": {"type": "object"},
                    },
                },
            },
        )

    async def _read_meta_memories(self) -> str:
        from ...tool import ReadMetaMemoryOp

        op = ReadMetaMemoryOp(enable_tool_memory=self.enable_tool_memory,
                              enable_identity_memory=False)
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        now_time: str = self.get_now_time()
        meta_memory_info = await self._read_meta_memories()
        context: str = self.format_messages(self.get_messages())

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            meta_memory_info=meta_memory_info,
            context=context,
        )

        user_message = self.get_prompt("user_message")
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]
        return messages

    async def _acting_step(
            self,
            assistant_message: Message,
            tool_op_dict: Dict[str, BaseAsyncToolOp],
            step: int,
            **kwargs,
    ) -> List[Message]:
        return await super()._acting_step(assistant_message,
                                          tool_op_dict,
                                          step,
                                          workspace_id=self.workspace_id)
