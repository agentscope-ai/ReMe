"""ReMy conversational agent for AI assistant interactions.

This module provides the ReMy agent that handles conversational interactions,
integrating identity memory and meta memory into the system prompt, then
running a standard ReAct loop with available tools.
"""

import datetime
from typing import List

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ...enumeration import Role
from ...schema import Message, ToolCall
from ...tool.read_identity_memory_op import ReadIdentityMemoryOp
from ...tool.read_meta_memory_op import ReadMetaMemoryOp


@C.register_op()
class ReMyAgentV1Op(BaseMemoryAgentOp):
    """Conversational AI assistant agent with memory integration."""

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
                        "description": "workspace identifier",
                        "required": True,
                    },
                    "query": {
                        "type": "string",
                        "description": "user query string",
                        "required": False,
                    },
                    "messages": {
                        "type": "array",
                        "description": "conversation messages",
                        "required": False,
                        "items": {"type": "string"},
                    },
                },
            },
        )

    async def _load_identity_memory(self) -> str:
        """Load identity memory from file storage.

        Returns:
            str: Identity memory content or empty string if not found.
        """
        op = ReadIdentityMemoryOp(language=self.language)
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def _load_meta_memories(self) -> str:
        """Load meta memory information from file storage.

        Returns:
            str: Formatted meta memory information or empty string.
        """
        op = ReadMetaMemoryOp(language=self.language, include_identity=False)
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        """Build the initial messages for the conversational agent.

        Constructs messages by:
        1. Converting query to user message OR using provided messages (without system)
        2. Loading identity_memory and meta_memory
        3. Building system prompt from memories
        4. Combining system prompt with conversation messages

        Returns:
            List[Message]: Complete message list with system prompt and conversation.
        """
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        identity_memory = await self._load_identity_memory()
        meta_memory_info = await self._load_meta_memories()

        conversation_messages: List[Message] = []
        if "query" in self.input_dict and self.input_dict["query"]:
            query = self.input_dict["query"]
            conversation_messages.append(Message(role=Role.USER, content=query))
        elif "messages" in self.input_dict and self.input_dict["messages"]:
            raw_messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in self.input_dict["messages"]]
            conversation_messages.extend([msg for msg in raw_messages if msg.role is not Role.SYSTEM])
        else:
            raise ValueError("input_dict must contain either `query` or `messages`")

        # Build system prompt with memory context
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            identity_memory=identity_memory,
            meta_memory_info=meta_memory_info,
        )

        messages = [Message(role=Role.SYSTEM, content=system_prompt)] + conversation_messages
        return messages

