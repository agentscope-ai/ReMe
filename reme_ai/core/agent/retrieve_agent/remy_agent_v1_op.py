import datetime
from typing import List

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ...enumeration import Role
from ...schema import Message, ToolCall
from ...tool.retrieve_tool.read_meta_memory_op import ReadMetaMemoryOp


@C.register_op()
class ReMeRetrieveAgentV1Op(BaseMemoryAgentOp):

    def build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "input_schema": {
                    "workspace_id": {
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
                        "items": {"type": "string"},
                    },
                },
            },
        )

    async def _load_meta_memories(self) -> str:
        op = ReadMetaMemoryOp(language=self.language)
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        """Build the initial messages for the retrieve agent.

        Returns:
            List[Message]: List containing the system prompt message.
        """
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        meta_memory_info = await self._load_meta_memories()

        context: str = ""
        if "query" in self.input_dict:
            context += self.input_dict["query"]
        if "messages" in self.input_dict:
            context += self.format_messages(self.input_dict["messages"])

        assert context, "input_dict must contain either `query` or `messages`"

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            context=context,
            meta_memory_info=meta_memory_info,
        )

        user_message = self.get_prompt("user_message")

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]
        return messages
