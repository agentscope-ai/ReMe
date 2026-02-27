"""Memory merging operation module.

This module provides functionality to merge multiple retrieved memories
into a single formatted context string for use in LLM responses.
"""

from typing import List

from loguru import logger

from ....core.op import BaseOp
from ....core.schema.memory_node import MemoryNode


class MergeMemory(BaseOp):
    """Merge multiple memories into a single formatted context.

    This operation takes a list of retrieved memories and formats them into
    a single context string that can be used to guide LLM responses. It includes
    instructions for the LLM to consider the helpful parts from these memories.
    """

    async def execute(self):
        """Execute the memory merging operation.

        Merges memories from context metadata into a formatted string with
        instructions for the LLM. Stores the merged result in response.answer.
        """
        memory_list: List[MemoryNode] = self.context.response.metadata["memory_list"]

        if not memory_list:
            return

        content_collector = ["Previous Memory"]
        for memory in memory_list:
            if not memory.content:
                continue

            content_collector.append(f"- {memory.when_to_use} {memory.content}\n")
        content_collector.append(
            "Please consider the helpful parts from these in answering the question, "
            "to make the response more comprehensive and substantial.",
        )
        self.context.response.answer = "\n".join(content_collector)
        logger.info(f"response.answer={self.context.response.answer}")
