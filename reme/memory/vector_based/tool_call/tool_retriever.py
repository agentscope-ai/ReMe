"""Tool memory retriever agent for retrieving tool usage experiences through vector search."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.op import BaseTool
from ....core.schema import Message
from ....core.utils import format_messages


class ToolRetriever(BaseMemoryAgent):
    """Retrieve tool memories through vector search and history reading.

    Tool memories represent knowledge about tool usage including:
    - Successful tool invocations and their parameters
    - Failed tool calls and lessons learned
    - Tool selection strategies for different scenarios
    - Parameter optimization patterns
    """

    memory_type: MemoryType = MemoryType.TOOL

    def __init__(self, return_memory_nodes: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_memory_nodes: bool = return_memory_nodes

    async def build_messages(self) -> list[Message]:
        """Build messages with tool memory retrieval context."""
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = self.description + "\n" + format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    context=context.strip(),
                ),
            ),
        ]

    async def _acting_step(
        self,
        assistant_message: Message,
        tools: list[BaseTool],
        step: int,
        stage: str = "",
        **kwargs,
    ) -> tuple[list[BaseTool], list[Message]]:
        """Execute tool calls with memory context."""
        return await super()._acting_step(
            assistant_message,
            tools,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            retrieved_nodes=self.retrieved_nodes,
            **kwargs,
        )

    async def execute(self):
        result = await super().execute()
        if self.return_memory_nodes:
            result["answer"] = "\n".join(
                [
                    n.format(
                        include_memory_id=False,
                        include_when_to_use=True,
                        include_content=True,
                        include_message_time=False,
                        ref_memory_id_key="",
                    )
                    for n in self.retrieved_nodes
                ],
            )

        result["retrieved_nodes"] = self.retrieved_nodes
        return result
