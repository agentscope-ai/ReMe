"""Retrieve memory from vector store"""

from loguru import logger

from .memory_handler import MemoryHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall, MemoryNode
from ....core.utils import deduplicate_memories


class RetrieveMemory(BaseMemoryTool):
    """Tool to retrieve memories using similarity search"""

    def __init__(
        self,
        top_k: int = 20,
        enable_memory_target: bool = False,
        enable_time_filter: bool = False,
        hybrid_threshold: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k: int = top_k
        self.enable_memory_target: bool = enable_memory_target
        self.enable_time_filter: bool = enable_time_filter
        self.hybrid_threshold: float | None = hybrid_threshold

    def _build_query_parameters(self) -> dict:
        """Build the query parameters schema based on enabled features."""
        properties = {
            "query": {
                "type": "string",
                "description": "query",
            },
        }
        required = ["query"]

        if self.enable_time_filter:
            properties["time_filter"] = {
                "type": "string",
                "description": "Optional time filter to narrow down search results by date. "
                "Format: single date '20200101' for exact date match, "
                "or date range '20200101,20200102' for inclusive range filtering.",
            }

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
                "description": "Retrieve relevant memories from the vector store using semantic similarity search.",
                "parameters": self._build_query_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Retrieve relevant memories from the vector store using semantic similarity search.",
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

        queries_by_target: dict[str, list[dict]] = {}
        for item in query_items:
            if self.enable_memory_target:
                target = item["memory_target"]
            else:
                target = self.memory_target
            if target not in queries_by_target:
                queries_by_target[target] = []

            filters = {}
            time_filter = item.get("time_filter")
            if time_filter:
                time_filter = time_filter.strip()
                if "," in time_filter:
                    start, end = time_filter.split(",")
                    filters = {"time_int": [int(start.strip()), int(end.strip())]}
                else:
                    filters = {"time_int": [int(time_filter), int(time_filter)]}

            queries_by_target[target].append(
                {
                    "query": item["query"],
                    "limit": self.top_k,
                    "filters": filters,
                },
            )

        # Execute batch searches for each target
        memory_nodes: list[MemoryNode] = []
        for target, searches in queries_by_target.items():
            handler = MemoryHandler(target, self.service_context)
            if self.hybrid_threshold is not None:
                nodes = await handler.batch_search(searches, self.hybrid_threshold)
                nodes = nodes[: self.top_k]
            else:
                nodes = await handler.batch_search(searches)
            memory_nodes.extend(nodes)

        memory_nodes = deduplicate_memories(memory_nodes)
        retrieved_ids = {n.memory_id for n in self.retrieved_nodes if n.memory_id}
        new_nodes = [n for n in memory_nodes if n.memory_id not in retrieved_ids]
        self.retrieved_nodes.extend(new_nodes)

        if not new_nodes:
            output = "No new memories found."
        else:
            output = "\n".join([n.format(ref_memory_id_key="history_id") for n in new_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} memories, {len(new_nodes)} new after deduplication")
        return output
