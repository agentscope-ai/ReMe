"""
ReMe Copaw Application Module

This module provides the ReMeCopaw class, a specialized application built on top of
ReMe's core Application framework. It integrates memory management capabilities
including memory compaction, summarization, tool result management, and semantic
memory search functionality.

Key Features:
    - Memory compaction and summarization for long conversations
    - Tool result compaction with file-based storage for large outputs
    - Semantic memory search using vector and full-text search
    - Configurable embedding models and vector store backends
    - Async task management for background summarization
"""

import asyncio
import logging
import os
import platform
from pathlib import Path

from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase
from agentscope.token import HuggingFaceTokenCounter
from agentscope.tool import Toolkit, ToolResponse

from .config import ReMeConfigParser
from .core import Application
from .memory.file_based_copaw import Compactor, Summarizer, ToolResultCompactor, CoPawInMemoryMemory
from .memory.tools import MemorySearch

# Module-level logger for tracking application events and errors
logger = logging.getLogger(__name__)


class ReMeCopaw(Application):
    """
    ReMe Copaw Application Class

    A specialized application class that extends ReMe's core Application framework
    with advanced memory management capabilities. This class is designed to handle
    long-running conversations by providing intelligent memory compaction,
    summarization, and semantic search features.

    Attributes:
        working_path (Path): Absolute path to the working directory for storing data
        memory_path (Path): Path to the memory storage directory
        tool_result_path (Path): Path to store large tool result files
        chat_model (ChatModelBase): Language model for generating summaries and processing
        formatter (FormatterBase): Formatter for structuring model inputs/outputs
        token_counter (HuggingFaceTokenCounter): Token counting utility for length management
        toolkit (Toolkit): Collection of tools available to the application
        max_input_length (int): Maximum allowed input length in tokens
        memory_compact_threshold (int): Threshold at which memory compaction triggers
        language (str): Language code for localization ("zh" for Chinese, empty for English)
        vector_weight (float): Weight for vector search in hybrid search (0.0-1.0)
        candidate_multiplier (float): Multiplier for candidate retrieval in search
        tool_result_threshold (int): Size threshold for tool result compaction
        retention_days (int): Number of days to retain tool result files
        summary_tasks (list[asyncio.Task]): List of background summarization tasks
    """

    def __init__(
        self,
        working_dir: str,
        chat_model: ChatModelBase,
        formatter: FormatterBase,
        token_counter: HuggingFaceTokenCounter,
        toolkit: Toolkit,
        max_input_length: int,
        memory_compact_ratio: float,
        language: str = "zh",
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
        tool_result_threshold: int = 1000,
        retention_days: int = 7,
    ):
        # Initialize working directory structure
        # All application data will be stored under this path
        self.working_path = Path(working_dir).absolute()
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Create memory storage directory for persistent memory files
        self.memory_path = self.working_path / "memory"
        self.memory_path.mkdir(parents=True, exist_ok=True)

        # Create tool result directory for storing large tool outputs
        self.tool_result_path = self.working_path / "tool_result"
        self.tool_result_path.mkdir(parents=True, exist_ok=True)

        # Store references to core components
        self.chat_model: ChatModelBase = chat_model
        self.formatter: FormatterBase = formatter
        self.token_counter: HuggingFaceTokenCounter = token_counter
        self.toolkit: Toolkit = toolkit

        # Initialize runtime parameters (will be updated via update_params)
        self.max_input_length: int = 0
        self.memory_compact_threshold: int = 0
        self.language: str = ""

        # Store configuration parameters
        self.vector_weight: float = vector_weight
        self.candidate_multiplier: float = candidate_multiplier
        self.tool_result_threshold: int = tool_result_threshold
        self.retention_days: int = retention_days

        # Apply initial parameter configuration
        self.update_params(
            max_input_length=max_input_length,
            memory_compact_ratio=memory_compact_ratio,
            language=language,
        )

        # Retrieve embedding configuration from environment variables
        # These settings control the vector search capabilities
        (
            embedding_api_key,
            embedding_base_url,
            embedding_model_name,
            embedding_dimensions,
            embedding_cache_enabled,
            embedding_max_cache_size,
            embedding_max_input_length,
            embedding_max_batch_size,
        ) = self.get_emb_envs()

        # Determine if vector search should be enabled based on configuration
        # Vector search requires either an API key or a local model name
        vector_enabled = bool(embedding_api_key) or bool(embedding_model_name)
        if vector_enabled:
            logger.info("Vector search enabled.")
        else:
            logger.warning(
                "Vector search disabled. Memory search functionality will be restricted. "
                "To enable, configure: EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL_NAME.",
            )

        # Check if full-text search (FTS) is enabled via environment variable
        fts_enabled = os.environ.get("FTS_ENABLED", "true").lower() == "true"

        # Determine the memory store backend to use
        # "auto" selects based on platform (local for Windows, chroma otherwise)
        memory_store_backend = os.environ.get("MEMORY_STORE_BACKEND", "auto")
        if memory_store_backend == "auto":
            memory_backend = "local" if platform.system() == "Windows" else "chroma"
        else:
            memory_backend = memory_store_backend

        # Initialize the parent Application class with configuration
        # Initialize the parent Application class with comprehensive configuration
        super().__init__(
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            working_dir=str(self.working_path),
            config_path="copaw",
            enable_logo=False,
            log_to_console=False,
            parser=ReMeConfigParser,
            default_embedding_model_config={
                "model_name": embedding_model_name,
                "dimensions": embedding_dimensions,
                "enable_cache": embedding_cache_enabled,
                "use_dimensions": False,
                "max_cache_size": embedding_max_cache_size,
                "max_input_length": embedding_max_input_length,
                "max_batch_size": embedding_max_batch_size,
            },
            default_file_store_config={
                "backend": memory_backend,
                "store_name": "copaw",
                "vector_enabled": vector_enabled,
                "fts_enabled": fts_enabled,
            },
            default_file_watcher_config={
                "watch_paths": [
                    str(self.working_path / "MEMORY.md"),
                    str(self.working_path / "memory.md"),
                    str(self.memory_path),
                ],
            },
        )

        # Initialize list to track background summarization tasks
        self.summary_tasks: list[asyncio.Task] = []

    def update_params(
        self,
        max_input_length: int,
        memory_compact_ratio: float,
        language: str,
    ):
        """
        Update runtime parameters for memory management.

        This method allows dynamic adjustment of memory-related parameters during
        runtime. It recalculates the memory compaction threshold based on the
        new input length and compaction ratio.

        Args:
            max_input_length (int): New maximum input length in tokens
            memory_compact_ratio (float): Ratio at which to trigger compaction (0.0-1.0)
            language (str): Language code for localization ("zh" or other)

        Note:
            The memory_compact_threshold is calculated as:
            max_input_length * memory_compact_ratio * 0.9
            The 0.9 factor provides a safety margin before reaching the absolute limit
        """
        # Update the maximum allowed input length
        self.max_input_length = max_input_length

        # Calculate compaction threshold with safety margin
        # This ensures compaction happens before hitting the hard limit
        self.memory_compact_threshold = int(max_input_length * memory_compact_ratio * 0.9)

        # Set language for localization
        if language == "zh":
            self.language = "zh"
        else:
            self.language = ""

    @staticmethod
    def _safe_str(key: str, default: str) -> str:
        """
        Safely retrieve a string value from an environment variable.

        Args:
            key (str): The name of the environment variable to retrieve
            default (str): The default value to return if the variable is not set

        Returns:
            str: The value of the environment variable, or the default if not set
        """
        return os.environ.get(key, default)

    @staticmethod
    def _safe_int(key: str, default: int) -> int:
        """
        Safely retrieve an integer value from an environment variable.

        This method handles cases where the environment variable is not set
        or contains a non-integer value by returning the specified default.

        Args:
            key (str): The name of the environment variable to retrieve
            default (int): The default value to return on failure or if not set

        Returns:
            int: The integer value of the environment variable, or the default

        Note:
            Logs a warning if the value exists but cannot be parsed as an integer
        """
        value = os.environ.get(key)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid int value '{value}' for key '{key}', using default {default}")
            return default

    def get_emb_envs(self):
        """
        Retrieve all embedding-related configuration from environment variables.

        This method collects all settings needed for the embedding service,
        including API credentials, model configuration, and caching parameters.

        Environment Variables:
            EMBEDDING_API_KEY: API key for the embedding service
            EMBEDDING_BASE_URL: Base URL for the embedding API (default: dashscope)
            EMBEDDING_MODEL_NAME: Name of the embedding model to use
            EMBEDDING_DIMENSIONS: Vector dimensions (default: 1024)
            EMBEDDING_CACHE_ENABLED: Whether to enable caching (default: true)
            EMBEDDING_MAX_CACHE_SIZE: Maximum cache entries (default: 2000)
            EMBEDDING_MAX_INPUT_LENGTH: Max input text length (default: 8192)
            EMBEDDING_MAX_BATCH_SIZE: Max batch size for requests (default: 10)

        Returns:
            tuple: A tuple containing all embedding configuration values in order:
                (api_key, base_url, model_name, dimensions, cache_enabled,
                 max_cache_size, max_input_length, max_batch_size)
        """
        # API authentication and endpoint configuration
        embedding_api_key = self._safe_str("EMBEDDING_API_KEY", "")
        embedding_base_url = self._safe_str("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        embedding_model_name = self._safe_str("EMBEDDING_MODEL_NAME", "")

        # Model and vector configuration
        embedding_dimensions = self._safe_int("EMBEDDING_DIMENSIONS", 1024)

        # Caching configuration for performance optimization
        embedding_cache_enabled = self._safe_str("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
        embedding_max_cache_size = self._safe_int("EMBEDDING_MAX_CACHE_SIZE", 2000)

        # Input processing limits
        embedding_max_input_length = self._safe_int("EMBEDDING_MAX_INPUT_LENGTH", 8192)
        embedding_max_batch_size = self._safe_int("EMBEDDING_MAX_BATCH_SIZE", 10)

        return (
            embedding_api_key,
            embedding_base_url,
            embedding_model_name,
            embedding_dimensions,
            embedding_cache_enabled,
            embedding_max_cache_size,
            embedding_max_input_length,
            embedding_max_batch_size,
        )

    def _cleanup_tool_results(self) -> int:
        """
        Clean up expired tool result files from the tool result directory.

        This method removes tool result files that have exceeded the retention
        period specified during initialization. It helps manage disk space by
        automatically removing old, unused tool outputs.

        Returns:
            int: The number of files that were successfully deleted

        Note:
            Exceptions during cleanup are logged but do not raise errors,
            ensuring the application continues to function even if cleanup fails
        """
        try:
            # Create a compactor instance with current configuration
            compactor = ToolResultCompactor(
                tool_result_dir=self.tool_result_path,
                tool_result_threshold=self.tool_result_threshold,
                retention_days=self.retention_days,
            )
            # Execute cleanup and return count of deleted files
            return compactor.cleanup_expired_files()
        except Exception as e:
            # Log exception details but return 0 to indicate failure gracefully
            logger.exception(f"Error cleaning up tool results: {e}")
            return 0

    async def start(self):
        """
        Start the application lifecycle.

        This method initializes the application by calling the parent class's
        start method and performs initial cleanup of expired tool result files.

        Returns:
            The result from the parent class's start method

        Note:
            Tool result cleanup runs after successful startup to ensure
            the application is fully initialized before performing maintenance
        """
        # Initialize parent application components
        result = await super().start()
        # Perform initial cleanup of old tool result files
        self._cleanup_tool_results()
        return result

    async def close(self) -> bool:
        """
        Close the application and perform cleanup.

        This method performs final cleanup of expired tool result files before
        shutting down the application through the parent class's close method.

        Returns:
            bool: True if shutdown was successful, False otherwise

        Note:
            Cleanup is performed before calling parent close to ensure
            all resources are available during the cleanup process
        """
        # Clean up tool results before shutting down
        self._cleanup_tool_results()
        # Shutdown parent application components
        return await super().close()

    async def compact_tool_result(
        self,
        messages: list[Msg],
    ) -> list[Msg]:
        """
        Compact tool results by truncating large outputs and saving full content to files.

        This method processes a list of messages and identifies tool results that exceed
        the configured size threshold. Large tool outputs are truncated in the message
        list while their full content is saved to files for later retrieval.

        Args:
            messages (list[Msg]): List of messages to process for tool result compaction

        Returns:
            list[Msg]: The processed message list with large tool results compacted

        Note:
            - Tool results below the threshold remain unchanged in the messages
            - Large results are replaced with truncated versions and file references
            - Expired files are cleaned up as part of the compaction process
            - If compaction fails, the original messages are returned unchanged
        """
        try:
            # Create compactor with instance configuration
            compactor = ToolResultCompactor(
                tool_result_dir=self.tool_result_path,
                tool_result_threshold=self.tool_result_threshold,
                retention_days=self.retention_days,
            )
            # Set the messages context for the compactor to process
            compactor.context["messages"] = messages

            # Execute compaction and get processed messages
            result = await compactor.execute()

            # Clean up any expired tool result files during compaction
            compactor.cleanup_expired_files()

            return result

        except Exception as e:
            # Log the error and return original messages to maintain functionality
            logger.exception(f"Error compacting tool results: {e}")
            return messages

    async def compact_memory(self, messages: list[Msg], previous_summary: str = "") -> str:
        """
        Compact a list of messages into a condensed summary.

        This method uses the Compactor to reduce the length of message history
        while preserving essential information. It's useful when conversation
        history approaches the maximum input length limit.

        Args:
            messages (list[Msg]): The list of messages to compact
            previous_summary (str): Optional previous summary to incorporate
                into the compaction process for continuity

        Returns:
            str: A compacted summary of the messages, or empty string on failure

        Note:
            - Compaction uses the configured language model to generate summaries
            - The compaction threshold determines when compaction is triggered
            - If compaction fails, an empty string is returned
        """
        try:
            # Initialize compactor with current configuration
            compactor = Compactor(
                memory_compact_threshold=self.memory_compact_threshold,
                chat_model=self.chat_model,
                formatter=self.formatter,
                token_counter=self.token_counter,
                language=self.language,
            )

            # Execute compaction with optional previous summary context
            return await compactor.call(messages=messages, previous_summary=previous_summary)

        except Exception as e:
            # Log error and return empty string to indicate failure
            logger.exception(f"Error compacting memory: {e}")
            return ""

    async def summary_memory(self, messages: list[Msg]) -> str:
        """
        Generate a comprehensive summary of the given messages.

        This method uses the Summarizer to create a detailed summary of the
        conversation history, which can be stored as persistent memory. Unlike
        compaction, summarization aims to capture key information in a format
        suitable for long-term storage and retrieval.

        Args:
            messages (list[Msg]): The list of messages to summarize

        Returns:
            str: A generated summary of the messages, or empty string on failure

        Note:
            - Summarization may use tools from the toolkit to enhance the summary
            - The summary is typically stored in the memory directory
            - If summarization fails, an empty string is returned
        """
        try:
            # Initialize summarizer with working directories and configuration
            compactor = Summarizer(
                working_dir=str(self.working_path),
                memory_dir=str(self.memory_path),
                memory_compact_threshold=self.memory_compact_threshold,
                chat_model=self.chat_model,
                formatter=self.formatter,
                token_counter=self.token_counter,
                toolkit=self.toolkit,
                language=self.language,
            )

            # Execute summarization on the provided messages
            return await compactor.call(messages=messages)

        except Exception as e:
            # Log error and return empty string to indicate failure
            logger.exception(f"Error summarizing memory: {e}")
            return ""

    async def await_summary_tasks(self) -> str:
        """
        Wait for all background summary tasks to complete and collect results.

        This method iterates through all pending summary tasks, waits for their
        completion, and collects their results or error information. It's used
        to synchronize with background summarization operations before shutdown
        or when results are needed.

        Returns:
            str: A concatenated string containing the status and results of
                all summary tasks, with each task on a new line

        Note:
            - Completed tasks are processed immediately without waiting
            - Incomplete tasks are awaited with a timeout
            - Cancelled tasks and exceptions are logged and included in results
            - The task list is cleared after processing all tasks
        """
        result = ""
        for task in self.summary_tasks:
            if task.done():
                # Task has already completed, check its status
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    result += "Summary task was cancelled.\n"
                else:
                    # Check if the task raised an exception
                    exc = task.exception()
                    if exc is not None:
                        logger.exception(f"Summary task failed: {exc}")
                        result += f"Summary task failed: {exc}\n"
                    else:
                        # Task completed successfully, collect result
                        task_result = task.result()
                        logger.info(f"Summary task completed: {task_result}")
                        result += f"Summary task completed: {task_result}\n"

            else:
                # Task is still running, wait for it to complete
                try:
                    task_result = await task
                    logger.info(f"Summary task completed: {task_result}")
                    result += f"Summary task completed: {task_result}\n"

                except asyncio.CancelledError:
                    logger.warning("Summary task was cancelled while waiting.")
                    result += "Summary task was cancelled.\n"
                except Exception as e:
                    logger.exception(f"Summary task failed: {e}")
                    result += f"Summary task failed: {e}\n"

        # Clear the task list after processing all tasks
        self.summary_tasks.clear()
        return result

    def add_async_summary_task(self, messages: list[Msg]):
        """
        Add an asynchronous summary task for the given messages.

        This method creates a background task to summarize the provided messages
        without blocking the main execution flow. Before adding a new task, it
        cleans up any completed tasks from the task list to prevent memory leaks.

        Args:
            messages (list[Msg]): The list of messages to be summarized in the
                background task

        Note:
            - Completed tasks are removed from the tracking list before adding
            - Task status (success, failure, cancellation) is logged for monitoring
            - The new task is created using asyncio.create_task for true async execution
            - Failed or cancelled tasks are logged but do not prevent new tasks
        """
        # Clean up completed summary tasks before adding a new one
        remaining_tasks = []
        for task in self.summary_tasks:
            if task.done():
                # Process completed task status
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.exception(f"Summary task failed: {exc}")
                else:
                    # Log successful completion with result summary
                    result = task.result()
                    logger.info(f"Summary task completed: {result}")
            else:
                # Keep incomplete tasks in the tracking list
                remaining_tasks.append(task)
        self.summary_tasks = remaining_tasks

        # Create and track the new background summarization task
        task = asyncio.create_task(self.summary_memory(messages=messages))
        self.summary_tasks.append(task)

    async def memory_search(self, query: str, max_results: int = 5, min_score: float = 0.1) -> ToolResponse:
        """
        Perform semantic memory search using vector and full-text search.

        This method searches the memory store for content relevant to the given query
        using a hybrid approach combining vector similarity search and full-text search.
        Results are ranked by relevance and filtered by the minimum score threshold.

        Args:
            query (str): The search query string. Must not be empty.
            max_results (int): Maximum number of results to return (1-100, default: 5)
            min_score (float): Minimum relevance score threshold (0.001-0.999, default: 0.1)

        Returns:
            ToolResponse: A ToolResponse containing the search results as text,
                or an error message if the query is empty

        Note:
            - Vector search weight is controlled by self.vector_weight
            - Candidate retrieval uses self.candidate_multiplier for broader search
            - Parameters are validated and clamped to valid ranges
            - Requires vector search to be enabled via embedding configuration
        """
        # Validate query parameter
        if not query:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: No query provided.",
                    ),
                ],
            )

        # Validate and clamp max_results to valid range [1, 100]
        if isinstance(max_results, int):
            max_results = min(max(max_results, 1), 100)
        else:
            max_results = 5

        # Validate and clamp min_score to valid range [0.001, 0.999]
        if isinstance(min_score, (int, float)):
            min_score = min(max(min_score, 0.001), 0.999)
        else:
            min_score = 0.1

        # Initialize memory search tool with configured weights
        search_tool = MemorySearch(
            vector_weight=self.vector_weight,
            candidate_multiplier=self.candidate_multiplier,
        )

        # Execute the search with validated parameters
        search_result = await search_tool.call(
            query=query,
            max_results=max_results,
            min_score=min_score,
            service_context=self.service_context,
        )

        # Return results wrapped in ToolResponse format
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=search_result,
                ),
            ],
        )

    def get_in_memory_memory(self):
        """
        Create and return an in-memory memory instance.

        This method instantiates a CoPawInMemoryMemory object configured with
        the current application's token counter, formatter, and input length limits.
        The in-memory memory provides fast, temporary storage for conversation
        context without persistence.

        Returns:
            CoPawInMemoryMemory: A configured in-memory memory instance ready
                for storing and retrieving conversation messages

        Note:
            - In-memory memory is volatile and cleared when the instance is destroyed
            - Useful for managing conversation context within a single session
            - Shares the same token counter and formatter as the main application
        """
        return CoPawInMemoryMemory(
            token_counter=self.token_counter,
            formatter=self.formatter,
            max_input_length=self.max_input_length,
        )
