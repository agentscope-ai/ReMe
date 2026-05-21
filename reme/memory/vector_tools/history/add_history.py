"""Add history tool"""

import json

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall, MemoryNode, Message
from ....core.utils import format_messages
from .history_chunking import split_history_messages


class AddHistory(BaseMemoryTool):
    """Tool to add historical dialogue to vector store"""

    def __init__(
        self,
        chunk_strategy: str = "hybrid",
        turn_block_size: int = 3,
        max_chunk_tokens: int = 800,
        **kwargs,
    ):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.chunk_strategy = chunk_strategy
        self.turn_block_size = turn_block_size
        self.max_chunk_tokens = max_chunk_tokens

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "Add original history dialogue.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def execute(self):
        """Execute the add history operation"""
        self.context.messages = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        history_content: str = self.context.description + "\n" + format_messages(self.context.messages)
        history_content = history_content.strip()
        history_chunks = split_history_messages(
            self.context.messages,
            chunk_strategy=self.chunk_strategy,
            turn_block_size=self.turn_block_size,
            max_chunk_tokens=self.max_chunk_tokens,
        )
        history_node = MemoryNode(
            memory_type=MemoryType.HISTORY,
            when_to_use=history_content[:1024],
            content=history_content,
            author=self.author,
            metadata={
                "messages": json.dumps(
                    [m.model_dump(exclude_none=True) for m in self.context.messages],
                    ensure_ascii=False,
                ),
                "node_kind": "history",
                "chunk_count": len(history_chunks),
                "chunk_strategy": self.chunk_strategy,
                "turn_block_size": self.turn_block_size,
                "max_chunk_tokens": self.max_chunk_tokens,
            },
        )
        self.context.history_node = history_node
        logger.info(f"Adding history node: {history_node.model_dump_json(indent=2)}")

        chunk_nodes = [
            MemoryNode(
                memory_id=f"{history_node.memory_id}:chunk:{chunk.index:04d}",
                memory_type=MemoryType.HISTORY,
                memory_target=history_node.memory_target,
                when_to_use=chunk.content,
                content=chunk.content,
                author=self.author,
                ref_memory_id=history_node.memory_id,
                metadata={
                    "node_kind": "history_chunk",
                    "history_id": history_node.memory_id,
                    "chunk_index": chunk.index,
                    "start_message_index": chunk.start_message_index,
                    "end_message_index": chunk.end_message_index,
                    "token_count": chunk.token_count,
                    "chunk_strategy": self.chunk_strategy,
                },
            )
            for chunk in history_chunks
        ]

        vector_nodes = [history_node.to_vector_node(), *[node.to_vector_node() for node in chunk_nodes]]
        existing_chunks = await self.vector_store.list(
            filters={
                "node_kind": "history_chunk",
                "history_id": history_node.memory_id,
            },
        )
        delete_ids = [node.vector_id for node in vector_nodes]
        delete_ids.extend(node.vector_id for node in existing_chunks if node.vector_id not in delete_ids)
        await self.vector_store.delete(delete_ids)
        await self.vector_store.insert(vector_nodes)

        return f"Successfully added history: {history_node.memory_id} ({len(chunk_nodes)} chunks)"
