"""ReMe retriever agent that orchestrates multiple memory agents to retrieve information."""

from .base_memory_agent import BaseMemoryAgent
from ...core.enumeration import Role
from ...core.op import BaseTool
from ...core.schema import Message
from ...core.utils import format_messages


class ReMeRetriever(BaseMemoryAgent):
    """Orchestrate multiple memory agents to retrieve information."""

    async def build_messages(self) -> list[Message]:
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = self.description + "\n" + format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        return [
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
            query=self.query,
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

        answer = []
        success = True
        messages = []
        tools_result = []
        retrieved_nodes = []

        if tools:
            delegate_task_tool = tools[0]
            agents: list[BaseMemoryAgent] = delegate_task_tool.response.metadata["agents"]
            for agent in agents:
                answer.append(agent.response.answer)
                success = success and agent.response.success
                messages.extend(agent.response.metadata["messages"])
                tools_result.extend(agent.response.metadata["tools"])
                retrieved_nodes.extend(agent.response.metadata["retrieved_nodes"])

        return {
            "answer": "\n".join(answer),
            "success": True,
            "messages": messages,
            "tools": tools_result,
            "retrieved_nodes": retrieved_nodes,
        }
