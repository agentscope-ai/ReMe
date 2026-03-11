"""Read history memory tool"""

import json

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import MemoryNode, ToolCall, Message
from ....core.utils import format_messages, cosine_similarity


class ReadHistoryV2(BaseMemoryTool):
    """Read history memory tool"""

    def __init__(self, message_block_size: int = 4, vector_top_k: int = 5, **kwargs):
        kwargs.setdefault("enable_multiple", False)
        super().__init__(name="read_history", **kwargs)
        self.message_block_size: int = message_block_size
        self.vector_top_k: int = vector_top_k

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "Read original history dialogue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "history_id": {
                            "type": "string",
                            "description": "history_id",
                        },
                        "query": {
                            "type": "string",
                            "description": "Query to filter the history",
                        },
                    },
                    "required": ["history_id", "query"],
                },
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for multiple histories"""
        return ToolCall(
            **{
                "description": "Read multiple original history dialogues by their IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "history_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "history_id": {
                                        "type": "string",
                                        "description": "history_id",
                                    },
                                    "query": {
                                        "type": "string",
                                        "description": "Query to filter the history",
                                    },
                                },
                                "required": ["history_id", "query"],
                            },
                            "description": "List of history items to read, each containing history_id and query",
                        },
                    },
                    "required": ["history_items"],
                },
            },
        )

    async def execute(self):
        """Execute the tool call"""
        if self.enable_multiple:
            history_items = self.context.history_items
        else:
            history_items = [{"history_id": self.context.history_id, "query": self.context.query}]

        if not history_items:
            output = "No history_ids provided."
            logger.warning(output)
            return output

        all_results = []
        for history_item in history_items:
            history_id = history_item["history_id"]
            query = history_item["query"]
            query_embedding = await self.embedding_model.get_embedding(query)

            history_node: MemoryNode = await self.vector_store.get(vector_ids=history_id)
            messages = json.loads(history_node.metadata["messages"])
            messages = [Message(**m) for m in messages]

            message_blocks = []
            for i in range(0, len(messages), self.message_block_size):
                block = messages[i : i + self.message_block_size]
                message_blocks.append(block)

            block_similarities = []
            for block in message_blocks:
                block_text = format_messages(block, add_index=False)
                block_embedding = await self.embedding_model.get_embedding(block_text)
                similarity = cosine_similarity(query_embedding, block_embedding)
                block_similarities.append((similarity, block_text))

            block_similarities.sort(key=lambda x: x[0], reverse=True)
            top_k_blocks = block_similarities[: self.vector_top_k]
            result_text = "\n".join([block_text for _, block_text in top_k_blocks])
            all_results.append(result_text)

        if not all_results:
            history_ids = [item["history_id"] for item in history_items]
            output = f"No data found for history_ids={history_ids}."
            logger.warning(output)
            return output

        output = "\n\n".join(all_results)
        history_ids = [item["history_id"] for item in history_items]
        logger.info(f"Successfully read {len(all_results)} history result(s): {history_ids}")
        return output
