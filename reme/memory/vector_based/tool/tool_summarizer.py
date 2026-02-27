"""Tool memory summarizer agent implementation."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType


class ToolSummarizer(BaseMemoryAgent):
    """Agent responsible for summarizing tool-related memories."""

    memory_type: MemoryType = MemoryType.TOOL
