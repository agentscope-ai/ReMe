"""ReMe classes for simplified configuration and execution."""

import sys
from pathlib import Path

from .agent.memory import (
    BaseMemoryAgent,
    ReMeSummarizer,
    ReMeRetriever,
    PersonalV1Summarizer,
    PersonalV1Retriever,
    PersonalSummarizer,
    PersonalRetriever,
    ProceduralSummarizer,
    ProceduralRetriever,
    ToolSummarizer,
    ToolRetriever,
)
from .config import ReMeConfigParser
from .core import Application
from .core.enumeration import MemoryType
from .core.schema import Message
from .tool.memory import (
    RetrieveMemory,
    DelegateTask,
    ReadHistory,
    ProfileHandler,
    MemoryHandler,
    AddDraftAndRetrieveSimilarMemory,
    AddDraftAndReadAllProfiles,
    UpdateProfile,
    AddHistory,
    ReadAllProfiles,
    AddMemory,
    ReturnRelevantContent,
    UpdateProfileV1,
)


class ReMe(Application):
    """ReMe with config file support and flow execution methods."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        enable_logo: bool = True,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        target_user_names: list[str] | None = None,
        target_task_names: list[str] | None = None,
        target_tool_names: list[str] | None = None,
        profile_dir: str = "reme_profile",
        **kwargs,
    ):
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            enable_logo=enable_logo,
            parser=ReMeConfigParser,
            llm=llm,
            embedding_model=embedding_model,
            vector_store=vector_store,
            token_counter=token_counter,
            **kwargs,
        )
        memory_target_type_mapping: dict[str, MemoryType] = {}
        if target_user_names:
            for name in target_user_names:
                assert name not in memory_target_type_mapping, f"target_user_names={name} is already used."
                memory_target_type_mapping[name] = MemoryType.PERSONAL

        if target_task_names:
            for name in target_task_names:
                assert name not in memory_target_type_mapping, f"target_task_names={name} is already used."
                memory_target_type_mapping[name] = MemoryType.PROCEDURAL

        if target_tool_names:
            for name in target_tool_names:
                assert name not in memory_target_type_mapping, f"target_tool_names={name} is already used."
                memory_target_type_mapping[name] = MemoryType.TOOL

        self.service_context.memory_target_type_mapping = memory_target_type_mapping
        self.profile_dir: str = profile_dir

    def _add_meta_memory(self, memory_type: str | MemoryType, memory_target: str):
        """Register or validate a memory target with the given memory type."""
        if memory_target in self.service_context.memory_target_type_mapping:
            assert self.service_context.memory_target_type_mapping[memory_target] is memory_type
        else:
            self.service_context.memory_target_type_mapping[memory_target] = MemoryType(memory_type)

    async def summary_memory(
        self,
        messages: list[Message | dict],
        description: str = "",
        user_name: str | list[str] = "",
        task_name: str | list[str] = "",
        tool_name: str | list[str] = "",
        enable_thinking_params: bool = True,
        version: str = "default",
        retrieve_top_k: int = 20,
        return_dict: bool = False,
        **kwargs,
    ) -> str | dict:
        """Summarize personal, procedural and tool memories for the given context."""
        format_messages: list[Message] = []
        for message in messages:
            if isinstance(message, dict):
                assert message.get("time_created"), "message must have time_created field."
            message = Message(**message)
            format_messages.append(message)

        personal_summarizer: BaseMemoryAgent
        if version == "default":
            personal_summarizer = PersonalSummarizer(
                tools=[
                    AddDraftAndRetrieveSimilarMemory(
                        top_k=retrieve_top_k,
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_when_to_use=False,
                        enable_multiple=True,
                    ),
                    AddMemory(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_when_to_use=False,
                        enable_multiple=True,
                    ),
                    AddDraftAndReadAllProfiles(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_multiple=True,
                        profile_dir=self.profile_dir,
                    ),
                    UpdateProfile(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_multiple=True,
                        profile_dir=self.profile_dir,
                    ),
                ],
            )

        elif version == "v1":
            personal_summarizer = PersonalV1Summarizer(
                tools=[
                    AddDraftAndRetrieveSimilarMemory(
                        top_k=retrieve_top_k,
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_when_to_use=False,
                        enable_multiple=True,
                    ),
                    AddMemory(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_when_to_use=False,
                        enable_multiple=True,
                    ),
                    AddDraftAndReadAllProfiles(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_multiple=True,
                        profile_dir=self.profile_dir,
                    ),
                    UpdateProfileV1(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_multiple=True,
                        profile_dir=self.profile_dir,
                    ),
                ],
            )

        else:
            raise NotImplementedError

        procedural_summarizer: BaseMemoryAgent
        if version in ["default", "v1"]:
            procedural_summarizer = ProceduralSummarizer(tools=[])
        else:
            raise NotImplementedError

        tool_summarizer: BaseMemoryAgent
        if version in ["default", "v1"]:
            tool_summarizer = ToolSummarizer(tools=[])
        else:
            raise NotImplementedError

        memory_agents = []
        if user_name:
            if isinstance(user_name, str):
                for message in format_messages:
                    message.name = user_name
                self._add_meta_memory(MemoryType.PERSONAL, user_name)
            elif isinstance(user_name, list):
                for name in user_name:
                    self._add_meta_memory(MemoryType.PERSONAL, name)
            else:
                raise RuntimeError("user_name must be str or list[str]")
            memory_agents.append(personal_summarizer)

        if task_name:
            if isinstance(task_name, str):
                self._add_meta_memory(MemoryType.PROCEDURAL, task_name)
            elif isinstance(task_name, list):
                for name in task_name:
                    self._add_meta_memory(MemoryType.PROCEDURAL, name)
            else:
                raise RuntimeError("task_name must be str or list[str]")
            memory_agents.append(procedural_summarizer)

        if tool_name:
            if isinstance(tool_name, str):
                self._add_meta_memory(MemoryType.TOOL, tool_name)
            elif isinstance(tool_name, list):
                for name in tool_name:
                    self._add_meta_memory(MemoryType.TOOL, name)
            else:
                raise RuntimeError("tool_name must be str or list[str]")
            memory_agents.append(tool_summarizer)

        if not memory_agents:
            memory_agents = [personal_summarizer, procedural_summarizer, tool_summarizer]

        reme_summarizer: BaseMemoryAgent
        if version in ["default", "v1"]:
            reme_summarizer = ReMeSummarizer(tools=[AddHistory(), DelegateTask(memory_agents=memory_agents)])
        else:
            raise NotImplementedError

        result = await reme_summarizer.call(
            messages=format_messages,
            description=description,
            service_context=self.service_context,
            **kwargs,
        )

        if return_dict:
            return result
        else:
            return result["answer"]

    async def retrieve_memory(
        self,
        query: str = "",
        description: str = "",
        messages: list[dict] | None = None,
        user_name: str | list[str] = "",
        task_name: str | list[str] = "",
        tool_name: str | list[str] = "",
        enable_thinking_params: bool = True,
        version: str = "default",
        retrieve_top_k: int = 20,
        enable_time_filter: bool = True,
        return_dict: bool = False,
        **kwargs,
    ) -> str | dict:
        """Retrieve relevant personal, procedural and tool memories for a query."""

        personal_retriever: BaseMemoryAgent
        if version == "default":
            personal_retriever = PersonalRetriever(
                tools=[
                    ReadAllProfiles(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        profile_dir=self.profile_dir,
                    ),
                    RetrieveMemory(
                        top_k=retrieve_top_k,
                        enable_thinking_params=enable_thinking_params,
                        enable_time_filter=enable_time_filter,
                        enable_multiple=True
                    ),
                    ReadHistory(
                        enable_multiple=True,
                        enable_thinking_params=enable_thinking_params,
                    ),
                ],
            )

        elif version == "v1":
            personal_retriever = PersonalV1Retriever(
                return_memory_nodes=False,
                tools=[
                    ReadAllProfiles(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        profile_dir=self.profile_dir,
                    ),
                    RetrieveMemory(
                        top_k=retrieve_top_k,
                        enable_thinking_params=enable_thinking_params,
                        enable_time_filter=enable_time_filter,
                        enable_multiple=True
                    ),
                    ReadHistory(
                        enable_multiple=True,
                        enable_thinking_params=enable_thinking_params,
                    ),
                    ReturnRelevantContent(enable_multiple=True),
                ],
            )

        else:
            raise NotImplementedError

        procedural_retriever: BaseMemoryAgent
        if version in ["default", "v1"]:
            procedural_retriever = ProceduralRetriever(tools=[])
        else:
            raise NotImplementedError

        tool_retriever: BaseMemoryAgent
        if version in ["default", "v1"]:
            tool_retriever = ToolRetriever(tools=[])
        else:
            raise NotImplementedError

        memory_agents = []
        if user_name:
            if isinstance(user_name, str):
                self._add_meta_memory(MemoryType.PERSONAL, user_name)
            elif isinstance(user_name, list):
                for name in user_name:
                    self._add_meta_memory(MemoryType.PERSONAL, name)
            else:
                raise RuntimeError("user_name must be str or list[str]")
            memory_agents.append(personal_retriever)

        if task_name:
            if isinstance(task_name, str):
                self._add_meta_memory(MemoryType.PROCEDURAL, task_name)
            elif isinstance(task_name, list):
                for name in task_name:
                    self._add_meta_memory(MemoryType.PROCEDURAL, name)
            else:
                raise RuntimeError("task_name must be str or list[str]")
            memory_agents.append(procedural_retriever)

        if tool_name:
            if isinstance(tool_name, str):
                self._add_meta_memory(MemoryType.TOOL, tool_name)
            elif isinstance(tool_name, list):
                for name in tool_name:
                    self._add_meta_memory(MemoryType.TOOL, name)
            else:
                raise RuntimeError("tool_name must be str or list[str]")
            memory_agents.append(tool_retriever)

        if not memory_agents:
            memory_agents = [personal_retriever, procedural_retriever, tool_retriever]

        reme_retriever: BaseMemoryAgent
        if version in ["default", "v1"]:
            reme_retriever = ReMeRetriever(tools=[DelegateTask(memory_agents=memory_agents)])
        else:
            raise NotImplementedError

        result = await reme_retriever.call(
            query=query,
            messages=messages,
            description=description,
            service_context=self.service_context,
            **kwargs,
        )

        if return_dict:
            return result
        else:
            return result["answer"]

    @property
    def profile_path(self) -> Path:
        """Get the path to the profile directory."""
        return Path(self.profile_dir) / self.vector_store.collection_name

    def get_memory_handler(self, memory_target: str) -> MemoryHandler:
        """Get the memory handler for the specified memory target."""
        return MemoryHandler(memory_target=memory_target, service_context=self.service_context)

    def get_profile_handler(self, user_name: str) -> ProfileHandler:
        """Get the profile handler for the specified user."""
        return ProfileHandler(memory_target=user_name, profile_path=self.profile_path)

    async def context_offload(self):
        """working memory summary"""

    async def context_reload(self):
        """working memory retrieve"""


def main():
    """Main entry point for running ReMe from command line."""
    with ReMe(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
