import datetime
from typing import List

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
                        "description": "tool_name to extract guidelines for",
                        "required": True,
                    },
                    "query": {
                        "type": "string",
                        "description": "tool_name (same as memory_target)",
                        "required": False,
                    },
                    "messages": {
                        "type": "array",
                        "description": "messages containing tool calls and results",
                        "required": False,
                        "items": {"type": "string"},
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
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        context: str = ""
        if "query" in self.context:
            context += f"Tool Name: {self.context['query']}\n\n"
        if "messages" in self.context:
            context += self.format_messages(self.context["messages"])

        assert context, "input_dict must contain either `query` or `messages`"

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.get_prompt("system_prompt").format(
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

