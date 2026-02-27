"""Personal memory summarizer agent for two-phase personal memory processing."""

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.op import BaseTool
from ....core.schema import Message


class PersonalSummarizer(BaseMemoryAgent):
    """Two-phase personal memory processor: retrieve/add memories then update profile."""

    memory_type: MemoryType = MemoryType.PERSONAL

    async def _build_s1_messages(self) -> list[Message]:
        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message_s1",
                    context=self.context.history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                ),
            ),
        ]

    async def _build_s2_messages(self, profiles: str) -> list[Message]:
        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message_s2",
                    profiles=profiles,
                    context=self.context.history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
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
            stage=stage,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            history_node=self.history_node,
            author=self.author,
            retrieved_nodes=self.retrieved_nodes,
            **kwargs,
        )

    async def execute(self):
        memory_tools = []
        profile_tools = []
        read_all_profiles_tool: BaseTool | None = None
        for i, tool in enumerate(self.tools):
            tool_name = tool.tool_call.name
            if tool_name == "read_all_profiles":
                read_all_profiles_tool = tool
            elif "_memory" in tool_name:
                memory_tools.append(tool)
            elif "_profile" in tool_name:
                profile_tools.append(tool)
            else:
                raise ValueError(f"[{self.__class__.__name__}] unknown tool_name={tool_name}")
            logger.info(f"[{self.__class__.__name__}] tool_call[{i}]={tool.tool_call.simple_input_dump(as_dict=False)}")

        stage = "s1-memory"
        messages_s1 = await self._build_s1_messages()
        for i, message in enumerate(messages_s1):
            role = message.name or message.role
            logger.info(f"[{self.__class__.__name__} {stage}] role={role} {message.simple_dump(as_dict=False)}")
        tools_s1, messages_s1, success_s1 = await self.react(messages_s1, memory_tools, stage=stage)

        if read_all_profiles_tool is not None:
            profiles = await read_all_profiles_tool.call(
                memory_target=self.memory_target,
                service_context=self.service_context,
            )
        else:
            profiles = ""

        if profile_tools:
            stage = "s2-profile"
            messages_s2 = await self._build_s2_messages(profiles)
            for i, message in enumerate(messages_s2):
                role = message.name or message.role
                logger.info(f"[{self.__class__.__name__} {stage}] role={role} {message.simple_dump(as_dict=False)}")
            tools_s2, messages_s2, success_s2 = await self.react(messages_s2, profile_tools, stage=stage)
        else:
            tools_s2, messages_s2, success_s2 = [], [], True

        answer = (messages_s1[-1].content if success_s1 and messages_s1 else "") + (
            messages_s2[-1].content if success_s2 and messages_s2 else ""
        )
        success = success_s1 and success_s2
        messages = messages_s1 + messages_s2
        tools = tools_s1 + tools_s2
        memory_nodes = []
        for tool in tools:
            if tool.memory_nodes:
                memory_nodes.extend(tool.memory_nodes)

        return {
            "answer": answer,
            "success": success,
            "messages": messages,
            "tools": tools,
            "memory_nodes": memory_nodes,
        }
