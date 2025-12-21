import re
from typing import List, Dict

from loguru import logger

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ... import utils
from ...enumeration import Role
from ...schema import Message, ToolCall, MemoryNode
from ....core import BaseAsyncToolOp


@C.register_op()
class ReMeSummaryAgentV1Op(BaseMemoryAgentOp):

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
        """Build the tool call schema for ReMeSummaryAgent.

        Returns:
            ToolCall: Tool call configuration with workspace_id/query/messages input schema.
        """
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "input_schema": {
                    "workspace_id": {
                        "type": "string",
                        "description": "workspace id",
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

    async def _add_history_memory(self) -> MemoryNode:
        from ...tool import AddHistoryMemoryOp

        op = AddHistoryMemoryOp()
        await op.async_call(workspace_id=self.workspace_id, messages=self.get_messages(), author=self.author)
        return op.output

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
        now_time: str = utils.get_now_time()
        memory_node: MemoryNode = await self._add_history_memory()
        identity_memory = await self._read_identity_memory()
        meta_memory_info = await self._read_meta_memories()
        context: str = utils.format_messages(self.get_messages())

        logger.info(f"now_time={now_time} memory_node={memory_node} identity_memory={identity_memory} "
                    f"meta_memory_info={meta_memory_info} context={context}")

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            identity_memory=identity_memory,
            meta_memory_info=meta_memory_info,
            context=context,
        )

        user_message = self.get_prompt("user_message")
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]

        self.context["ref_memory_id"] = memory_node.memory_id
        return messages

    async def _reasoning_step(
        self,
        messages: List[Message],
        tool_op_dict: Dict[str, BaseAsyncToolOp],
        step: int,
    ) -> tuple[Message, bool]:
        meta_memory_info = await self._read_meta_memories()
        system_messages = [message for message in messages if message.role == Role.SYSTEM]
        if system_messages:
            system_message = system_messages[0]
            pattern = r'("- <memory_type>\(<memory_target>\): <description>"\n)(.*?)(\n\n)'
            replacement = rf'\g<1>{meta_memory_info}\g<3>'
            system_message.content = re.sub(pattern, replacement, system_message.content, flags=re.DOTALL)

        return await super()._reasoning_step(messages, tool_op_dict, step)

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
                                          author=self.author)
