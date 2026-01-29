"""Read history memory tool"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import MemoryNode, ToolCall


class ReadHistory(BaseMemoryTool):
    """Read history memory tool"""

    def __init__(self, **kwargs):
        kwargs.setdefault("enable_multiple", False)
        super().__init__(**kwargs)

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        if self.enable_multiple:
            return ToolCall(
                **{
                    "description": "Read multiple original history dialogues by their IDs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "history_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of history IDs to read",
                            },
                        },
                        "required": ["history_ids"],
                    },
                },
            )
        else:
            return ToolCall(
                **{
                    "description": "Read a single original history dialogue by its ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "history_id": {
                                "type": "string",
                                "description": "The history ID to read",
                            },
                        },
                        "required": ["history_id"],
                    },
                },
            )

    async def execute(self):
        # 统一获取 history_ids（单数模式转为列表）
        history_ids = self.context.history_ids if self.enable_multiple else [self.context.history_id]
        
        if not history_ids or (len(history_ids) == 1 and not history_ids[0]):
            output = "No history_ids provided."
            logger.warning(output)
            return output

        nodes = await self.vector_store.get(vector_ids=history_ids)

        if not nodes:
            output = f"No data found for history_ids={history_ids}."
            logger.warning(output)
            return output

        # 统一处理节点
        results = []
        for node in nodes:
            memory_node: MemoryNode = MemoryNode.from_vector_node(node)
            self.retrieved_nodes.append(memory_node)
            results.append(f"Historical Dialogue[{memory_node.memory_id}]\n{memory_node.content}")

        output = "\n\n".join(results) if self.enable_multiple else results[0]
        logger.info(f"Successfully read {len(nodes)} history memory_node(s): {history_ids}")
        return output
