"""File-based Memory Module.

This module provides memory management components for CoPaw (Cooperative Paw) agents,
including memory formatting, compaction, summarization, and file I/O operations.

Components:
    - MemoryFormatter: Converts message lists to formatted strings with token limiting
    - ReMeInMemoryMemory: Extended InMemoryMemory with bugfixes and summary support
    - Summarizer: Generates memory summaries using LLM
    - Compactor: Compacts memory content to reduce token usage
    - ToolResultCompactor: Truncates large tool results and saves full content to files
    - FileIO: File I/O operations with configurable working directory
"""

from . import utils
from .compactor import Compactor
from .file_io import FileIO
from .memory_formatter import MemoryFormatter
from .reme_chat_formatter import ReMeChatFormatter
from .reme_in_memory_memory import ReMeInMemoryMemory
from .summarizer import Summarizer
from .tool_result_compactor import ToolResultCompactor

__all__ = [
    "MemoryFormatter",
    "ReMeInMemoryMemory",
    "Summarizer",
    "Compactor",
    "ToolResultCompactor",
    "FileIO",
    "utils",
    "ReMeChatFormatter",
]
