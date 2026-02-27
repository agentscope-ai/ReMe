"""ReMe classes for simplified configuration and execution."""

import sys
from pathlib import Path

from .config import ReMeConfigParser
from .core import Application
from .core.enumeration import MemoryType, Role
from .core.schema import Message, MemoryNode
from .memory.tools import (
    AddDraftAndRetrieveSimilarMemory,
    AddHistory,
    AddMemory,
    DelegateTask,
    ReadAllProfiles,
    ReadHistory,
    RetrieveMemory,
    UpdateProfilesV1,
)
from .memory.tools.profiles.profile_handler import ProfileHandler
from .memory.tools.record.memory_handler import MemoryHandler
from .memory.vector_based import (
    BaseMemoryAgent,
    PersonalRetriever,
    PersonalSummarizer,
    ProceduralRetriever,
    ProceduralSummarizer,
    ReMeRetriever,
    ReMeSummarizer,
    ToolRetriever,
    ToolSummarizer,
)


class ReMe(Application):
    """ReMe with config file support and flow execution methods."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        working_dir: str = ".reme",
        config_path: str = "vector",
        enable_logo: bool = True,
        log_to_console: bool = True,
        default_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_vector_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        target_user_names: list[str] | None = None,
        target_task_names: list[str] | None = None,
        target_tool_names: list[str] | None = None,
        **kwargs,
    ):
        """Initialize ReMe with config.

        Example:
            ```python
            reme = ReMe(...)
            await reme.start()
            await reme.summarize_memory(...)
            await reme.retrieve_memory(...)
            await reme.close()
            ```
        """
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            working_dir=working_dir,
            config_path=config_path,
            enable_logo=enable_logo,
            log_to_console=log_to_console,
            parser=ReMeConfigParser,
            default_llm_config=default_llm_config,
            default_embedding_model_config=default_embedding_model_config,
            default_vector_store_config=default_vector_store_config,
            default_token_counter_config=default_token_counter_config,
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

        profile_path = Path(self.service_context.service_config.working_dir) / "profile"
        profile_path.mkdir(parents=True, exist_ok=True)
        self.profile_dir: str = str(profile_path)

    def _add_meta_memory(self, memory_type: str | MemoryType, memory_target: str):
        """Register or validate a memory target with the given memory type."""
        if memory_target in self.service_context.memory_target_type_mapping:
            assert self.service_context.memory_target_type_mapping[memory_target] is memory_type
        else:
            self.service_context.memory_target_type_mapping[memory_target] = MemoryType(memory_type)

    @staticmethod
    def _resolve_memory_target(
        user_name: str = "",
        task_name: str = "",
        tool_name: str = "",
    ) -> tuple[MemoryType, str]:
        """Resolve memory type and target from user_name, task_name, or tool_name.

        Args:
            user_name: User name for personal memory
            task_name: Task name for procedural memory
            tool_name: Tool name for tool memory

        Returns:
            tuple: (memory_type, memory_target)

        Raises:
            RuntimeError: If none or multiple memory targets are specified
        """
        if user_name:
            memory_type = MemoryType.PERSONAL
            memory_target = user_name
            assert not task_name and not tool_name, "Cannot add task and tool memory when user memory is specified"

        elif task_name:
            memory_type = MemoryType.PROCEDURAL
            memory_target = task_name
            assert not user_name and not tool_name, "Cannot add user and tool memory when task memory is specified"

        elif tool_name:
            memory_type = MemoryType.TOOL
            memory_target = tool_name
            assert not user_name and not task_name, "Cannot add user and task memory when tool memory is specified"

        else:
            raise RuntimeError("Must specify user_name, task_name, or tool_name")

        return memory_type, memory_target

    async def summarize_memory(
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
        llm_config_name: str = "default",
        **kwargs,
    ) -> str | dict:
        """Summarize personal, procedural and tool memories for the given context."""
        format_messages: list[Message] = []
        for message in messages:
            if isinstance(message, dict):
                assert message.get("time_created"), "message must have time_created field."
            message = Message(**message)
            format_messages.append(message)

        if version == "default":
            personal_summarizer: BaseMemoryAgent = PersonalSummarizer(
                llm=llm_config_name,
                tools=[
                    AddDraftAndRetrieveSimilarMemory(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_when_to_use=False,
                        enable_multiple=True,
                        top_k=retrieve_top_k,
                    ),
                    AddMemory(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_when_to_use=False,
                        enable_multiple=True,
                    ),
                    ReadAllProfiles(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        profile_dir=self.profile_dir,
                    ),
                    UpdateProfilesV1(
                        enable_thinking_params=enable_thinking_params,
                        enable_memory_target=False,
                        enable_multiple=True,
                        profile_dir=self.profile_dir,
                    ),
                ],
            )

        else:
            raise NotImplementedError(f"version={version} is not supported")

        procedural_summarizer: BaseMemoryAgent = ProceduralSummarizer(tools=[])
        tool_summarizer: BaseMemoryAgent = ToolSummarizer(tools=[])

        memory_agents = []
        memory_targets = []
        if user_name:
            if isinstance(user_name, str):
                for message in format_messages:
                    if message.role is Role.USER:
                        message.name = user_name
                self._add_meta_memory(MemoryType.PERSONAL, user_name)
                memory_targets.append(user_name)
            elif isinstance(user_name, list):
                for name in user_name:
                    self._add_meta_memory(MemoryType.PERSONAL, name)
                    memory_targets.append(name)
            else:
                raise RuntimeError("user_name must be str or list[str]")
            memory_agents.append(personal_summarizer)

        if task_name:
            if isinstance(task_name, str):
                self._add_meta_memory(MemoryType.PROCEDURAL, task_name)
                memory_targets.append(task_name)
            elif isinstance(task_name, list):
                for name in task_name:
                    self._add_meta_memory(MemoryType.PROCEDURAL, name)
                    memory_targets.append(name)
            else:
                raise RuntimeError("task_name must be str or list[str]")
            memory_agents.append(procedural_summarizer)

        if tool_name:
            if isinstance(tool_name, str):
                self._add_meta_memory(MemoryType.TOOL, tool_name)
                memory_targets.append(tool_name)
            elif isinstance(tool_name, list):
                for name in tool_name:
                    self._add_meta_memory(MemoryType.TOOL, name)
                    memory_targets.append(name)
            else:
                raise RuntimeError("tool_name must be str or list[str]")
            memory_agents.append(tool_summarizer)

        if not memory_agents:
            memory_agents = [personal_summarizer, procedural_summarizer, tool_summarizer]

        reme_summarizer: BaseMemoryAgent = ReMeSummarizer(
            tools=[AddHistory(), DelegateTask(memory_agents=memory_agents)],
        )

        result = await reme_summarizer.call(
            messages=format_messages,
            description=description,
            service_context=self.service_context,
            memory_targets=memory_targets,
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
        llm_config_name: str = "default",
        **kwargs,
    ) -> str | dict:
        """Retrieve relevant personal, procedural and tool memories for a query."""

        if version == "default":
            personal_retriever: BaseMemoryAgent = PersonalRetriever(
                llm=llm_config_name,
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
                        enable_multiple=True,
                    ),
                    ReadHistory(
                        enable_thinking_params=enable_thinking_params,
                        enable_multiple=True,
                    ),
                ],
            )
        else:
            raise NotImplementedError(f"version={version} is not supported")

        procedural_retriever: BaseMemoryAgent = ProceduralRetriever(tools=[])
        tool_retriever: BaseMemoryAgent = ToolRetriever(tools=[])

        memory_agents = []
        memory_targets = []
        if user_name:
            if isinstance(user_name, str):
                self._add_meta_memory(MemoryType.PERSONAL, user_name)
                memory_targets.append(user_name)
            elif isinstance(user_name, list):
                for name in user_name:
                    self._add_meta_memory(MemoryType.PERSONAL, name)
                    memory_targets.append(name)
            else:
                raise RuntimeError("user_name must be str or list[str]")
            memory_agents.append(personal_retriever)

        if task_name:
            if isinstance(task_name, str):
                self._add_meta_memory(MemoryType.PROCEDURAL, task_name)
                memory_targets.append(task_name)
            elif isinstance(task_name, list):
                for name in task_name:
                    self._add_meta_memory(MemoryType.PROCEDURAL, name)
                    memory_targets.append(name)
            else:
                raise RuntimeError("task_name must be str or list[str]")
            memory_agents.append(procedural_retriever)

        if tool_name:
            if isinstance(tool_name, str):
                self._add_meta_memory(MemoryType.TOOL, tool_name)
                memory_targets.append(tool_name)
            elif isinstance(tool_name, list):
                for name in tool_name:
                    self._add_meta_memory(MemoryType.TOOL, name)
                    memory_targets.append(name)
            else:
                raise RuntimeError("tool_name must be str or list[str]")
            memory_agents.append(tool_retriever)

        if not memory_agents:
            memory_agents = [personal_retriever, procedural_retriever, tool_retriever]

        reme_retriever: BaseMemoryAgent = ReMeRetriever(
            tools=[DelegateTask(memory_agents=memory_agents)],
        )

        result = await reme_retriever.call(
            query=query,
            messages=messages,
            description=description,
            service_context=self.service_context,
            memory_targets=memory_targets,
            **kwargs,
        )

        if return_dict:
            return result
        else:
            return result["answer"]

    async def add_memory(
        self,
        memory_content: str,
        user_name: str = "",
        task_name: str = "",
        tool_name: str = "",
        when_to_use: str = "",
        message_time: str = "",
        ref_memory_id: str = "",
        author: str = "",
        score: float = 0.0,
        **kwargs,
    ):
        """Add memory to the vector store.

        Args:
            memory_content: The content of the memory to add
            user_name: User name for personal memory
            task_name: Task name for procedural memory
            tool_name: Tool name for tool memory
            when_to_use: Description of when this memory should be used
            message_time: Timestamp of the message
            ref_memory_id: Reference to another memory ID
            author: Author of the memory
            score: Score/importance of the memory
            **kwargs: Additional metadata

        Returns:
            MemoryNode: The created memory node
        """
        memory_type, memory_target = self._resolve_memory_target(user_name, task_name, tool_name)
        self._add_meta_memory(memory_type, memory_target)

        handler = self.get_memory_handler(memory_target)
        memory_node = await handler.add(
            content=memory_content,
            when_to_use=when_to_use,
            message_time=message_time,
            ref_memory_id=ref_memory_id,
            author=author,
            score=score,
            **kwargs,
        )
        return memory_node

    async def get_memory(
        self,
        memory_id: str,
    ):
        """Get a memory node by its memory_id.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            MemoryNode: The retrieved memory node
        """
        vector_node = await self.default_vector_store.get(memory_id)
        return MemoryNode.from_vector_node(vector_node)

    async def delete_memory(
        self,
        memory_id: str,
    ):
        """Delete a memory node by its memory_id.

        Args:
            memory_id: The ID of the memory to delete
        """
        await self.default_vector_store.delete(memory_id)

    async def delete_all(self):
        """Delete all memory nodes in the vector store."""
        await self.default_vector_store.delete_all()

    async def update_memory(
        self,
        memory_id: str,
        user_name: str = "",
        task_name: str = "",
        tool_name: str = "",
        memory_content: str | None = None,
        when_to_use: str | None = None,
        message_time: str | None = None,
        ref_memory_id: str | None = None,
        author: str | None = None,
        score: float | None = None,
        **kwargs,
    ):
        """Update a memory node's content and/or metadata.

        Args:
            memory_id: The ID of the memory to update
            user_name: User name for personal memory
            task_name: Task name for procedural memory
            tool_name: Tool name for tool memory
            memory_content: New content for the memory (optional)
            when_to_use: New description of when to use (optional)
            message_time: New timestamp (optional)
            ref_memory_id: New reference memory ID (optional)
            author: New author (optional)
            score: New score/importance (optional)
            **kwargs: Additional metadata to update

        Returns:
            MemoryNode: The updated memory node
        """
        memory_type, memory_target = self._resolve_memory_target(user_name, task_name, tool_name)
        self._add_meta_memory(memory_type, memory_target)

        handler = self.get_memory_handler(memory_target)
        memory_node = await handler.update(
            memory_id=memory_id,
            content=memory_content,
            when_to_use=when_to_use,
            message_time=message_time,
            ref_memory_id=ref_memory_id,
            author=author,
            score=score,
            **kwargs,
        )
        return memory_node

    async def list_memory(
        self,
        user_name: str = "",
        task_name: str = "",
        tool_name: str = "",
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = True,
    ):
        """List memory nodes with optional filtering and sorting.

        Args:
            user_name: User name for personal memory
            task_name: Task name for procedural memory
            tool_name: Tool name for tool memory
            filters: Additional filters to apply (optional)
            limit: Maximum number of results to return (optional)
            sort_key: Field to sort by (optional)
            reverse: Sort in reverse order (default: True)

        Returns:
            list[MemoryNode]: List of memory nodes
        """
        memory_type, memory_target = self._resolve_memory_target(user_name, task_name, tool_name)
        self._add_meta_memory(memory_type, memory_target)

        handler = self.get_memory_handler(memory_target)
        memory_nodes = await handler.list(
            filters=filters,
            limit=limit,
            sort_key=sort_key,
            reverse=reverse,
        )
        return memory_nodes

    def get_memory_handler(self, memory_target: str) -> MemoryHandler:
        """Get the memory handler for the specified memory target."""
        return MemoryHandler(memory_target=memory_target, service_context=self.service_context)

    @property
    def profile_path(self) -> Path:
        """Get the path to the profile directory."""
        return Path(self.profile_dir) / self.default_vector_store.collection_name

    def get_profile_handler(self, user_name: str) -> ProfileHandler:
        """Get the profile handler for the specified user."""
        return ProfileHandler(memory_target=user_name, profile_path=self.profile_path)


def main():
    """Main entry point for running ReMe from command line."""
    from . import extension  # noqa: F401  # pylint: disable=unused-import
    from . import memory  # noqa: F401  # pylint: disable=unused-import

    ReMe(*sys.argv[1:], config_path="service").run_service()


if __name__ == "__main__":
    main()
