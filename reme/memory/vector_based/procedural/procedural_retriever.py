"""Procedural memory retriever agent implementation."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.schema import Message
from ....core.utils import format_messages


class ProceduralRetriever(BaseMemoryAgent):
    """Agent responsible for retrieving procedural memories."""

    memory_type: MemoryType = MemoryType.PROCEDURAL

    async def build_messages(self) -> list[Message]:
        """Build messages with system prompt and user message."""
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        return [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=await self._read_meta_memories(),
                    context=context,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]
