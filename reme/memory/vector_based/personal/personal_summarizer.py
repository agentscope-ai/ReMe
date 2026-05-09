"""Personal memory summarizer agent for two-phase personal memory processing."""

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType, Role
from ....core.op import BaseTool
from ....core.schema import Message

# Optional profile tools used to pre-load profile context; consumed by the
# summarizer itself and never exposed to the stage-two ReAct loop.
_PROFILE_CONTEXT_TOOLS: tuple[str, ...] = ("retrieve_profile", "read_all_profiles")


class PersonalSummarizer(BaseMemoryAgent):
    """Two-phase personal memory processor: add memories, then update profiles."""

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

    def _partition_tools(self) -> tuple[list[BaseTool], list[BaseTool], BaseTool | None]:
        """Split attached tools into memory tools, profile tools, and a profile context tool."""
        memory_tools: list[BaseTool] = []
        profile_tools: list[BaseTool] = []
        profile_context_tool: BaseTool | None = None
        for i, tool in enumerate(self.tools):
            name = tool.tool_call.name
            if name in _PROFILE_CONTEXT_TOOLS:
                profile_context_tool = tool
            elif "_memory" in name:
                memory_tools.append(tool)
            elif "_profile" in name:
                profile_tools.append(tool)
            else:
                raise ValueError(f"[{self.__class__.__name__}] unknown tool_name={name}")
            logger.info(f"[{self.__class__.__name__}] tool_call[{i}]={tool.tool_call.simple_input_dump(as_dict=False)}")
        return memory_tools, profile_tools, profile_context_tool

    async def _preload_user_profile(self, tool: BaseTool | None) -> str:
        """Invoke the profile context tool to obtain inline profile text."""
        if tool is None:
            return ""
        call_kwargs: dict = {
            "memory_target": self.memory_target,
            "service_context": self.service_context,
            "retrieved_nodes": self.retrieved_nodes,
        }
        if tool.tool_call.name == "retrieve_profile":
            call_kwargs["query"] = self.context.history_node.content
        return await tool.call(**call_kwargs)

    async def _run_stage(
        self,
        stage: str,
        messages: list[Message],
        tools: list[BaseTool],
    ) -> tuple[list[BaseTool], list[Message], bool]:
        for message in messages:
            role = message.name or message.role
            logger.info(f"[{self.__class__.__name__} {stage}] role={role} {message.simple_dump(as_dict=False)}")
        return await self.react(messages, tools, stage=stage)

    async def execute(self):
        memory_tools, profile_tools, profile_context_tool = self._partition_tools()

        messages_s1 = await self._build_s1_messages()
        tools_s1, messages_s1, success_s1 = await self._run_stage("s1-memory", messages_s1, memory_tools)

        if profile_tools:
            profiles = await self._preload_user_profile(profile_context_tool)
            messages_s2 = await self._build_s2_messages(profiles)
            tools_s2, messages_s2, success_s2 = await self._run_stage("s2-profile", messages_s2, profile_tools)
        else:
            tools_s2, messages_s2, success_s2 = [], [], True

        answer = (messages_s1[-1].content if success_s1 and messages_s1 else "") + (
            messages_s2[-1].content if success_s2 and messages_s2 else ""
        )
        tools = tools_s1 + tools_s2
        memory_nodes = [node for tool in tools for node in (tool.memory_nodes or [])]

        return {
            "answer": answer,
            "success": success_s1 and success_s2,
            "messages": messages_s1 + messages_s2,
            "tools": tools,
            "memory_nodes": memory_nodes,
        }
