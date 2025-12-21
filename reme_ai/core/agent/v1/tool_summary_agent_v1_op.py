from typing import List, Dict

from flowllm.core.op import BaseAsyncToolOp

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ...enumeration import Role, MemoryType
from ...schema import Message, ToolCall


@C.register_op()
class ToolSummaryAgentV1Op(BaseMemoryAgentOp):
    memory_type: MemoryType = MemoryType.TOOL

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
                    "memory_target": {
                        "type": "string",
                        "description": "memory_target",
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
                    "ref_memory_id": {
                        "type": "string",
                        "description": "ref_memory_id",
                        "required": True,
                    },
                },
            },
        )

    async def build_messages(self) -> List[Message]:
        now_time: str = self.get_now_time()
        context: str = self.format_messages(self.get_messages())

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    now_time=now_time,
                    context=context,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
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
                                          workspace_id=self.workspace_id,
                                          ref_memory_id=self.context["ref_memory_id"],
                                          memory_target=self.memory_target,
                                          memory_type=self.memory_type.value,
                                          author=self.author)
