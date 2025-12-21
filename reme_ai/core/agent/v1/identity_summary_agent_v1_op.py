"""Identity summary agent for extracting and storing agent self-cognition memories.

This module provides the IdentitySummaryAgentV1Op class for analyzing conversation context
to extract and update identity memories (self-cognition, personality, current state).
"""

from typing import List, Dict

from flowllm.core.op import BaseAsyncToolOp

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ... import utils
from ...enumeration import Role, MemoryType
from ...schema import Message, ToolCall


@C.register_op()
class IdentitySummaryAgentV1Op(BaseMemoryAgentOp):
    """Agent for extracting and storing identity memories.

    This agent analyzes conversation context to extract identity-related information
    such as self-cognition, personality traits, and current state, then updates
    the identity memory using file-based storage.
    """

    memory_type: MemoryType = MemoryType.IDENTITY

    def build_tool_call(self) -> ToolCall:
        """Build the tool call schema for identity summary agent.

        Returns:
            ToolCall: Tool call configuration with workspace_id/query/messages input schema.
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

    async def build_messages(self) -> List[Message]:
        """Build the initial messages for the identity summary agent.

        Constructs messages by:
        1. Combining query and messages into context
        2. Building system prompt with context and current time
        3. Adding user message to trigger analysis

        Returns:
            List[Message]: Complete message list with system prompt and user message.
        """
        now_time: str = utils.get_now_time()
        context: str = utils.format_messages(self.get_messages())

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    now_time=now_time,
                    context=context,
                    memory_type=self.memory_type.value,
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
                                          author=self.author)