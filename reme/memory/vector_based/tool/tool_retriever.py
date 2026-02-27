"""Tool memory retriever agent implementation."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType


class ToolRetriever(BaseMemoryAgent):
    """Agent responsible for retrieving tool-related memories."""

    memory_type: MemoryType = MemoryType.TOOL
