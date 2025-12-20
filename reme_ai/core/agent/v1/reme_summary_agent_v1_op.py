"""ReMeSummaryAgent for orchestrating memory summarization workflow.

This module provides the ReMeSummaryAgentV1Op class that coordinates the entire
memory summarization process: adding history memory, reading identity/meta memories,
creating new meta memories if needed, adding summary memories, and delegating to
specialized summary agents for different memory types.
"""

import datetime
import re
from typing import List, Dict

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ...enumeration import Role
from ...schema import Message, ToolCall
from ....core import BaseAsyncToolOp


@C.register_op()
class ReMeSummaryAgentV1Op(BaseMemoryAgentOp):
    """Agent for orchestrating the complete memory summarization workflow.

    This agent performs the following steps:
    1. Adds history memory and obtains ref_memory_id
    2. Reads identity memory
    3. Loads meta memories
    4. Creates new meta memories if context contains uncovered important information
    5. Adds summary memory for quick context recall
    6. Delegates to specialized summary agents (identity, personal, procedural, tool)
       for detailed memory extraction and storage
    """

    def __init__(
        self,
        enable_tool_memory: bool = True,
        enable_identity_memory: bool = False,
        **kwargs
    ):
        """Initialize the ReMeSummaryAgentV1Op.

        Args:
            enable_tool_memory: Whether to enable TOOL type meta memory. Defaults to True.
            enable_identity_memory: Whether to enable IDENTITY type meta memory. Defaults to False.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.enable_tool_memory = enable_tool_memory
        self.enable_identity_memory = enable_identity_memory

    def build_tool_call(self) -> ToolCall:
        """Build the tool call schema for ReMeSummaryAgent.

        Returns:
            ToolCall: Tool call configuration with workspace_id/query/messages input schema.
        """
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "input_schema": {
                    "workspace_id": {
                        "type": "string",
                        "description": "workspace identifier",
                        "required": True,
                    },
                    "query": {
                        "type": "string",
                        "description": "query text",
                        "required": False,
                    },
                    "messages": {
                        "type": "array",
                        "description": "conversation messages",
                        "required": False,
                        "items": {"type": "string"},
                    },
                },
            },
        )

    async def _add_history_memory(self) -> str:
        """Add history memory and return the ref_memory_id.

        Returns:
            str: The memory_id of the created history memory.
        """
        from ...tool import AddHistoryMemoryOp

        messages = self.context.get("messages", [])
        if not messages:
            return ""

        op = AddHistoryMemoryOp(language=self.language)
        await op.async_call(
            workspace_id=self.workspace_id,
            messages=messages,
        )

        # Extract memory_id from output
        if hasattr(op.output, "memory_id"):
            return op.output.memory_id
        return ""

    async def _read_identity_memory(self) -> str:
        """Read identity memory.

        Returns:
            str: The identity memory content.
        """
        from ...tool import ReadIdentityMemoryOp

        op = ReadIdentityMemoryOp(language=self.language)
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def _load_meta_memories(self) -> str:
        """Load meta memories.

        Returns:
            str: Formatted meta memory information.
        """
        from ...tool import ReadMetaMemoryOp

        op = ReadMetaMemoryOp(
            enable_tool_memory=self.enable_tool_memory,
            enable_identity_memory=self.enable_identity_memory,
            language=self.language,
        )
        await op.async_call(workspace_id=self.workspace_id)
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        """Build the initial messages for the summary agent.

        Constructs messages by:
        1. Adding history memory and getting ref_memory_id
        2. Reading identity memory
        3. Loading meta memories
        4. Combining query and messages into context
        5. Building system prompt with all information
        6. Adding user message to trigger analysis

        Returns:
            List[Message]: Complete message list with system prompt and user message.
        """
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Step 1: Add history memory and get ref_memory_id
        ref_memory_id = await self._add_history_memory()
        self.context["ref_memory_id"] = ref_memory_id  # Store for later use

        # Step 2: Read identity memory
        identity_memory = await self._read_identity_memory()
        self.context["identity_memory"] = identity_memory  # Store for later use

        # Step 3: Load meta memories
        meta_memory_info = await self._load_meta_memories()

        # Step 4: Build context from query and messages
        context: str = ""
        if "query" in self.context:
            context += self.context["query"]
        if "messages" in self.context:
            context += self.format_messages(self.context["messages"])

        assert context, "input_dict must contain either `query` or `messages`"

        # Step 5: Build system prompt
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            context=context,
            ref_memory_id=ref_memory_id,
            identity_memory=identity_memory,
            meta_memory_info=meta_memory_info,
        )

        # Step 6: Build user message
        user_message = self.get_prompt("user_message")

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]
        return messages

    async def _reasoning_step(
        self,
        messages: List[Message],
        tool_op_dict: Dict[str, BaseAsyncToolOp],
        step: int,
    ) -> tuple[Message, bool]:
        """Override reasoning step to reload meta memories before each reasoning iteration.

        Before executing the base class's reasoning step, this method:
        1. Reloads meta memories to get the latest information
        2. Updates the system prompt with the refreshed meta memory info using regex replacement

        Args:
            messages: Current conversation messages
            tool_op_dict: Dictionary of available tool operations
            step: Current step number

        Returns:
            tuple[Message, bool]: Assistant message and whether to continue acting
        """
        # Reload meta memories to get the latest information
        meta_memory_info = await self._load_meta_memories()

        # Find the system message and update meta_memory_info section using regex
        system_messages = [message for message in messages if message.role == Role.SYSTEM]
        if system_messages:
            system_message = system_messages[0]
            # Use regex to replace the meta_memory_info section
            # Pattern matches the format description line and content after it until next empty line
            # The format line "- <memory_type>(<memory_target>): <description>" stays unchanged
            # Only the actual meta_memory_info content after it gets replaced
            pattern = r'("- <memory_type>\(<memory_target>\): <description>"\n)(.*?)(\n\n)'
            replacement = rf'\g<1>{meta_memory_info}\g<3>'
            system_message.content = re.sub(pattern, replacement, system_message.content, flags=re.DOTALL)

        # Execute base class's reasoning step
        return await super()._reasoning_step(messages, tool_op_dict, step)

