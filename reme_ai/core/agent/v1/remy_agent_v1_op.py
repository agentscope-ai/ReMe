from typing import List

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ...enumeration import Role
from ...schema import Message, ToolCall


@C.register_op()
class ReMyAgentV1Op(BaseMemoryAgentOp):

    def __init__(
        self,
        enable_tool_memory: bool = True,
            enable_identity_memory: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enable_tool_memory = enable_tool_memory
        self.enable_identity_memory = enable_identity_memory

    def build_tool_call(self) -> ToolCall:
        """Build the tool call schema for ReMy agent.

        Returns:
            ToolCall: Tool call configuration with query/messages input schema.
        """
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

    async def _read_identity_memory(self) -> str:
        from ...tool import ReadIdentityMemoryOp

        op = ReadIdentityMemoryOp()
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def _read_meta_memories(self) -> str:
        from ...tool import ReadMetaMemoryOp

        op = ReadMetaMemoryOp(enable_tool_memory=self.enable_tool_memory,
                              enable_identity_memory=self.enable_identity_memory)
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        now_time: str = self.get_now_time()
        identity_memory = await self._read_identity_memory()
        meta_memory_info = await self._read_meta_memories()

        messages: List[Message] = []
        if self.context.get("query"):
            messages.append(Message(role=Role.USER, content=self.context.query))
        elif self.context.get("messages"):
            raw_messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in self.context["messages"]]
            messages.extend([msg for msg in raw_messages if msg.role is not Role.SYSTEM])
        else:
            raise ValueError("input_dict must contain either `query` or `messages`")

        # Build system prompt with memory context
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            identity_memory=identity_memory,
            meta_memory_info=meta_memory_info,
        )

        messages = [Message(role=Role.SYSTEM, content=system_prompt)] + messages
        return messages

