"""File-based CoPaw Memory Module.

This module provides memory management components for CoPaw (Cooperative Paw) agents,
including memory formatting, compaction, summarization, and file I/O operations.

Components:
    - MemoryFormatter: Converts message lists to formatted strings with token limiting
    - CoPawInMemoryMemory: Extended InMemoryMemory with bugfixes and summary support
    - Summarizer: Generates memory summaries using LLM
    - Compactor: Compacts memory content to reduce token usage
    - ToolResultCompactor: Truncates large tool results and saves full content to files
    - FileIO: File I/O operations with configurable working directory
"""

from . import utils
from .compactor import Compactor
from .copaw_in_memory_memory import CoPawInMemoryMemory
from .file_io import FileIO
from .memory_formatter import MemoryFormatter
from .summarizer import Summarizer
from .tool_result_compactor import ToolResultCompactor

__all__ = [
    "MemoryFormatter",
    "CoPawInMemoryMemory",
    "Summarizer",
    "Compactor",
    "ToolResultCompactor",
    "FileIO",
    "utils",
]
