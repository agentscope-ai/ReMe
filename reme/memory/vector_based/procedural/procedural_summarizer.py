"""Procedural memory summarizer agent implementation."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.schema import Message
from ....core.utils import format_messages


class ProceduralSummarizer(BaseMemoryAgent):
    """Agent responsible for summarizing procedural memories."""

    memory_type: MemoryType = MemoryType.PROCEDURAL

    async def build_messages(self) -> list[Message]:
        return [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    context=self.description + "\n" + format_messages(self.get_messages()),
                    outcome="successful task completion" if self.success else "task failure",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]

    async def _reasoning_step(self, messages: list[Message], step: int, **kwargs) -> tuple[Message, bool]:
        return await super()._reasoning_step(messages, step, **kwargs)

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        """Execute tool calls with memory_target, memory_type, and author context."""
        messages: list[Message] = await super()._acting_step(
            assistant_message,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            ref_memory_id=self.ref_memory_id,
            author=self.author,
            **kwargs,
        )

        return messages
