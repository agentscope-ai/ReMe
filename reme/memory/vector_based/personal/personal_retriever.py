"""Personal memory retriever agent for retrieving personal memories through vector search."""

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType, Role
from ....core.op import BaseTool
from ....core.schema import Message
from ....core.utils import format_messages

_PROFILE_TOOL_NAMES: tuple[str, ...] = ("retrieve_profile", "read_all_profiles")
_EMPTY_PROFILE_RESULTS: tuple[str, ...] = ("", "No profiles found.", "No new profiles found.")


class PersonalRetriever(BaseMemoryAgent):
    """Retrieve personal memories through vector search and history reading."""

    memory_type: MemoryType = MemoryType.PERSONAL

    def __init__(self, return_memory_nodes: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_memory_nodes: bool = return_memory_nodes

    def _get_context(self) -> str:
        if self.context.get("query"):
            return self.context.query.strip()
        if self.context.get("messages"):
            return (self.description + "\n" + format_messages(self.context.messages)).strip()
        raise ValueError("input must have either `query` or `messages`")

    async def _build_s1_messages(self, context: str) -> list[Message]:
        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message_s1",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    context=context,
                ),
            ),
        ]

    async def _build_s2_messages(self, context: str, profiles: str) -> list[Message]:
        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message_s2",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    profiles=profiles,
                    context=context,
                ),
            ),
        ]

    def _partition_tools(self) -> tuple[list[BaseTool], list[BaseTool]]:
        profile_tools: list[BaseTool] = []
        memory_tools: list[BaseTool] = []
        for i, tool in enumerate(self.tools):
            name = tool.tool_call.name
            if name in _PROFILE_TOOL_NAMES:
                profile_tools.append(tool)
            else:
                memory_tools.append(tool)
            logger.info(f"[{self.__class__.__name__}] tool_call[{i}]={tool.tool_call.simple_input_dump(as_dict=False)}")
        return profile_tools, memory_tools

    @staticmethod
    def _extract_profile_context(tools: list[BaseTool]) -> str:
        outputs = []
        for tool in tools:
            response = getattr(tool, "response", None)
            answer = getattr(response, "answer", "")
            if answer and answer not in _EMPTY_PROFILE_RESULTS:
                outputs.append(answer)
        return "\n".join(outputs)

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
        context = self._get_context()
        profile_tools, memory_tools = self._partition_tools()

        tools_s1: list[BaseTool] = []
        messages_s1: list[Message] = []
        success_s1 = True
        profiles = ""
        if profile_tools:
            messages_s1 = await self._build_s1_messages(context)
            tools_s1, messages_s1, success_s1 = await self._run_stage("s1-profile", messages_s1, profile_tools)
            profiles = self._extract_profile_context(tools_s1)

        messages_s2 = await self._build_s2_messages(context, profiles)
        tools_s2, messages_s2, success_s2 = await self._run_stage("s2-memory", messages_s2, memory_tools)

        answer = messages_s2[-1].content if success_s2 and messages_s2 else ""
        result = {
            "answer": answer,
            "success": success_s1 and success_s2,
            "messages": messages_s1 + messages_s2,
            "tools": tools_s1 + tools_s2,
        }
        if self.return_memory_nodes:
            result["answer"] = "\n".join(
                [
                    n.format(
                        include_memory_id=False,
                        include_when_to_use=False,
                        include_content=True,
                        include_message_time=True,
                        ref_memory_id_key="",
                    )
                    for n in self.retrieved_nodes
                ],
            )

        result["retrieved_nodes"] = self.retrieved_nodes
        return result
