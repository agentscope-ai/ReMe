"""Retrieve relevant profile rows."""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import MemoryNode, ToolCall


class RetrieveProfile(BaseMemoryTool):
    """Tool to retrieve relevant profiles using the configured backend."""

    def __init__(self, top_k: int = 5, enable_memory_target: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.enable_memory_target = enable_memory_target

    def _build_query_parameters(self) -> dict:
        properties = {
            "query": {
                "type": "string",
                "description": "query",
            },
        }
        required = ["query"]
        if self.enable_memory_target:
            properties["memory_target"] = {
                "type": "string",
                "description": "memory_target",
            }
            required.append("memory_target")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Retrieve relevant user profiles using semantic matching.",
                "parameters": self._build_query_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Retrieve relevant user profiles using semantic matching.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_items": {
                            "type": "array",
                            "description": "List of query items.",
                            "items": self._build_query_parameters(),
                        },
                    },
                    "required": ["query_items"],
                },
            },
        )

    async def execute(self):
        if self.enable_multiple:
            query_items = self.context.get("query_items", [])
        else:
            query_items = [self.context]

        queries_by_target: dict[str, list[str]] = {}
        for item in query_items:
            target = item["memory_target"] if self.enable_memory_target else self.memory_target
            queries_by_target.setdefault(target, []).append(item["query"])

        profile_nodes: list[MemoryNode] = []
        for target, queries in queries_by_target.items():
            profile_handler = self.get_profile_handler(target)
            nodes, _ = await profile_handler.aretrieve(
                query=queries,
                limit=self.top_k,
                add_profile_id=True,
                add_history_id=True,
            )
            profile_nodes.extend(nodes)

        seen_ids = {node.memory_id: node for node in self.retrieved_nodes if node.memory_id}
        new_nodes = []
        for node in profile_nodes:
            if node.memory_id not in seen_ids:
                seen_ids[node.memory_id] = node
                new_nodes.append(node)
        self.retrieved_nodes.extend(new_nodes)

        if not new_nodes:
            output = "No new profiles found."
        else:
            output = "\n".join(
                [ProfileHandler.format_node(node, add_profile_id=True, add_history_id=True) for node in new_nodes],
            )

        logger.info(f"Retrieved {len(profile_nodes)} profiles, {len(new_nodes)} new after deduplication")
        return output
