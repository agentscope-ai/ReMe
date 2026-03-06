"""File-based Memory Module.

This module provides memory management components for CoPaw (Cooperative Paw) agents,
including memory formatting, compaction, summarization, and file I/O operations.

Components:
    - ReMeInMemoryMemory: Extended InMemoryMemory with bugfixes and summary support
    - AsMsgHandler: Handles AgentScope message statistics, formatting, and context checking
    - Summarizer: Generates memory summaries using LLM
    - Compactor: Compacts memory content to reduce token usage
    - ToolResultCompactor: Truncates large tool results and saves full content to files
    - ContextChecker: Checks context size and splits messages for compaction
"""

from .as_msg_handler import AsMsgHandler
from .component.compactor import Compactor
from .component.context_checker import ContextChecker
from .component.summarizer import Summarizer
from .component.tool_result_compactor import ToolResultCompactor
from .reme_in_memory_memory import ReMeInMemoryMemory

__all__ = [
    "AsMsgHandler",
    "ReMeInMemoryMemory",
    "Summarizer",
    "Compactor",
    "ContextChecker",
    "ToolResultCompactor",
]
