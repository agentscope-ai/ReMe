"""Identity agent operation for managing identity memories.

This module provides the IdentityAgentOp class for creating, retrieving,
and updating identity memories (self-cognition, personality, current state).
"""

from typing import List

from flowllm.core.schema import VectorNode

from ... import C, BaseAsyncToolOp
from ...enumeration import MemoryType, IdentityMode
from ...schema import MemoryNode, ToolCall


@C.register_op()
class IdentityAgentOp(BaseAsyncToolOp):
    """Agent for managing identity memories.

    This agent supports three modes:
    - new_memory: Create a new identity memory from `init_identity_memory`
    - get_memory: Retrieve existing identity memories
    - update_memory: Update identity memory based on query and messages
    """

    def __init__(
            self,
            init_identity_memory: str = "",
            **kwargs,
    ):
        """Initialize the IdentityAgentOp.

        Args:
            init_identity_memory: Initial identity memory content (e.g., "My name is Remy,
                I am an optimistic and positive person, I am good at communicating with everyone").
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.init_identity_memory: str = init_identity_memory

    def build_tool_call(self) -> ToolCall:
        """Build tool call schema for identity memory operations.

        Returns:
            ToolCall: Tool call schema for identity operations.
        """
        return ToolCall(
            **{
                "description": "Identity memory tool for managing self-cognition memories.",
                "input_schema": {
                    "workspace_id": {
                        "type": "string",
                        "description": "Workspace identifier for identity memory storage.",
                        "required": True,
                    },
                    "query": {
                        "type": "string",
                        "description": "Query for retrieving or context for updating identity.",
                        "required": False,
                    },
                    "messages": {
                        "type": "array",
                        "description": "Conversation messages for identity update context.",
                        "required": False,
                        "items": {"type": "object"},
                    },
                    "mode": {
                        "type": "string",
                        "description": "Operation mode: 'new', 'get', or 'update'.",
                        "required": True,
                        "enum": [m.value for m in IdentityMode],
                    },
                },
            },
        )

    @property
    def workspace_id(self) -> str:
        return self.input_dict.get("workspace_id", "default")

    @property
    def mode(self) -> IdentityMode:
        mode_str = self.input_dict.get("mode", IdentityMode.GET.value)
        return IdentityMode(mode_str)

    @property
    def query(self) -> str:
        return self.input_dict.get("query", "")

    @property
    def messages(self) -> List[dict]:
        return self.input_dict.get("messages", [])

    def _build_memory_node(self, content: str) -> MemoryNode:
        """Build a MemoryNode for identity memory.

        Args:
            content: The memory content.

        Returns:
            MemoryNode: The constructed memory node.
        """
        return MemoryNode(
            workspace_id=self.workspace_id,
            memory_type=MemoryType.IDENTITY,
            memory_target="self",
            content=content,
            author=self.llm_config.model_name,
        )

    async def _create_identity_memory(self) -> str:
        """Create a new identity memory from init_memory.

        Returns:
            str: Result message.
        """
        if not self.init_identity_memory:
            return "No init_memory provided for creating identity memory."

        memory_node = self._build_memory_node(self.init_identity_memory)

        # Delete existing memory with same ID (upsert behavior)
        await self.vector_store.async_delete(
            node_ids=[memory_node.memory_id],
            workspace_id=self.workspace_id,
        )

        # Insert new memory
        vector_node = memory_node.to_vector_node()
        await self.vector_store.async_insert(
            nodes=[vector_node],
            workspace_id=self.workspace_id,
        )

        return f"Successfully created identity memory (id={memory_node.memory_id}) in workspace={self.workspace_id}."

    async def _get_identity_memory(self) -> str:
        """Retrieve identity memories.

        Returns:
            str: Formatted memory content or not found message.
        """

        nodes: List[VectorNode] = await self.vector_store.async_search(
            query="my identity and self-cognition",
            workspace_id=self.workspace_id,
            top_k=10,
            filter_dict={
                "metadata.memory_type": [MemoryType.IDENTITY.value],
                "metadata.memory_target": ["self"],
            },
        )

        if not nodes:
            return f"No identity memories found in workspace={self.workspace_id}."
        
        memory = MemoryNode.from_vector_node(nodes[0])
        return memory.content

    async def _update_identity_memory(self) -> str:
        """Update identity memory based on query and messages.

        Uses LLM to analyze the context and generate updated identity memory.

        Returns:
            str: Result message.
        """
        # First, get existing identity memories
        existing_memories = await self._get_identity_memory()

        # Build context from messages
        messages_context = ""
        if self.messages:
            messages_context = "\n".join([
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in self.messages
            ])

        # Build prompt for LLM to update identity
        update_prompt = self.prompt_format(
            prompt_name="update_prompt",
            existing_memories=existing_memories,
            query=self.query,
            messages_context=messages_context,
        )

        # Call LLM to generate updated identity memory
        response = await self.llm.achat_text(
            user_prompt=update_prompt,
        )

        if not response:
            return "Failed to generate updated identity memory."

        # Create new memory node with updated content
        memory_node = self._build_memory_node(response)

        # Delete old identity memories
        old_nodes: List[VectorNode] = await self.vector_store.async_search(
            query="my identity and self-cognition",
            workspace_id=self.workspace_id,
            top_k=100,
            filter_dict={
                "metadata.memory_type": [MemoryType.IDENTITY.value],
                "metadata.memory_target": ["self"],
            },
        )

        if old_nodes:
            old_ids = [n.unique_id for n in old_nodes]
            await self.vector_store.async_delete(
                node_ids=old_ids,
                workspace_id=self.workspace_id,
            )

        # Insert updated memory
        vector_node = memory_node.to_vector_node()
        await self.vector_store.async_insert(
            nodes=[vector_node],
            workspace_id=self.workspace_id,
        )

        return f"Successfully updated identity memory (id={memory_node.memory_id}) in workspace={self.workspace_id}."

    async def async_execute(self):
        """Execute the identity memory operation based on mode.

        Supports three modes:
        - new: Create identity memory from init_memory
        - get: Retrieve existing identity memories
        - update: Update identity memory using LLM analysis
        """
        mode = self.mode

        if mode == IdentityMode.NEW:
            result = await self._create_identity_memory()
        elif mode == IdentityMode.GET:
            result = await self._get_identity_memory()
        elif mode == IdentityMode.UPDATE:
            result = await self._update_identity_memory()
        else:
            result = f"Unknown mode: {mode}"

        self.set_output(result)
