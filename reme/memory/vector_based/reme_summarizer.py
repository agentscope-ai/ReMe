"""ReMe summarizer agent that orchestrates multiple memory agents to summarize information."""

from .base_memory_agent import BaseMemoryAgent
from ...core.enumeration import Role
from ...core.op import BaseTool
from ...core.schema import Message
from ...core.utils import format_messages


class ReMeSummarizer(BaseMemoryAgent):
    """Orchestrates multiple memory agents to summarize and store information across different memory types."""

    async def build_messages(self) -> list[Message]:
        add_history_tool: BaseTool | None = self.pop_tool("add_history")
        if add_history_tool is not None:
            await add_history_tool.call(
                messages=self.messages,
                description=self.description,
                author=self.author,
                service_context=self.service_context,
            )
            self.context.history_node = add_history_tool.context.history_node

        context = self.context.description + "\n" + format_messages(self.context.messages)
        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=self.meta_memory_info,
                    context=context.strip(),
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
        tools: list[BaseTool],
        step: int,
        stage: str = "",
        **kwargs,
    ) -> tuple[list[BaseTool], list[Message]]:
        return await super()._acting_step(
            assistant_message,
            tools,
            step,
            description=self.description,
            messages=self.messages,
            history_node=self.history_node,
            author=self.author,
            **kwargs,
        )

    async def react(self, messages: list[Message], tools: list["BaseTool"], stage: str = ""):
        """Run single ReAct step - only one tool call iteration."""
        used_tools: list[BaseTool] = []
        assistant_message, should_act = await self._reasoning_step(messages, tools, step=0, stage=stage)
        success = True

        if should_act:
            t_tools, tool_messages = await self._acting_step(assistant_message, tools, step=0, stage=stage)
            used_tools.extend(t_tools)
            messages.extend(tool_messages)

        return used_tools, messages, success

    async def execute(self):
        result = await super().execute()
        tools: list[BaseTool] = result["tools"]

        success = True
        messages = []
        tools_result = []
        memory_nodes = []

        if tools:
            delegate_task_tool = tools[0]
            agents: list[BaseMemoryAgent] = delegate_task_tool.response.metadata["agents"]

            for agent in agents:
                success = success and agent.response.success
                messages.extend(agent.response.metadata["messages"])
                tools_result.extend(agent.response.metadata["tools"])
                memory_nodes.extend(agent.response.metadata["memory_nodes"])

        return {
            "answer": memory_nodes,
            "success": True,
            "messages": messages,
            "tools": tools_result,
        }
