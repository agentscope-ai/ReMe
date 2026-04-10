"""
ReMe Light Application Module

This module provides the ReMeLight class, a specialized application built on top of
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
from pathlib import Path

from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase
from agentscope.token import HuggingFaceTokenCounter
from agentscope.tool import Toolkit, ToolResponse

from .config import ReMeConfigParser
from .core import Application
from .core.utils import get_logger
from .memory.file_based import ReMeInMemoryMemory
from .memory.file_based.components import (
    Compactor,
    ContextChecker,
    Summarizer,
    ToolResultCompactor,
)
from .memory.file_based.tools import FileIO, MemorySearch
from .memory.file_based.utils import AsMsgHandler

logger = get_logger()


class ReMeLight(Application):
    """
    ReMe Light Application Class.

    A lightweight memory-enabled application that provides semantic search,
    memory compaction, summarization, and tool result management capabilities.
    Built on top of the core Application framework with integrated vector store
    and file-based memory management.

    This class is designed for applications requiring:
        - Long conversation memory management with automatic compaction
        - Semantic search over stored memories using hybrid vector/text search
        - Background summarization of conversation history
        - Automatic cleanup of expired tool results

    Attributes:
        working_path (Path): Absolute path to the working directory.
        memory_path (Path): Path to the memory storage directory.
        tool_result_path (Path): Path to the tool result storage directory.
        dialog_path (Path): Path to the dialog storage directory for raw conversation records.
        vector_weight (float): Weight for vector search in hybrid search (0-1).
        candidate_multiplier (float): Multiplier for candidate retrieval count.
        summary_tasks (list[asyncio.Task]): List of active background summary tasks.
    """

    def __init__(
        self,
        working_dir: str = ".reme",
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        default_as_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_file_store_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
        enable_load_env: bool = False,
    ):
        """
        Initialize the ReMeLight application.

        Sets up the working directory structure, configures API connections,
        and initializes memory management components.

        Args:
            working_dir (str): Base directory for all application data storage.
                Defaults to ".reme". Will be created if it doesn't exist.
            llm_api_key (str | None): API key for the language model service.
                If None, will attempt to use environment variables.
            llm_base_url (str | None): Base URL for the language model API endpoint.
                If None, will use the default endpoint.
            embedding_api_key (str | None): API key for the embedding model service.
                If None, will attempt to use environment variables.
            embedding_base_url (str | None): Base URL for the embedding API endpoint.
                If None, will use the default endpoint.
            default_as_llm_config (dict | None): Default configuration dictionary
                for AgentScope language model. Overrides default settings.
            default_embedding_model_config (dict | None): Default configuration
                dictionary for the embedding model.
            default_file_store_config (dict | None): Default configuration
                dictionary for the file storage backend.
            default_file_watcher_config (dict | None): Default configuration
                dictionary for the file watcher. If ``watch_paths`` is included,
                it is used as-is. Otherwise the built-in watch paths (MEMORY.md,
                memory.md, and the memory directory) are used.
            vector_weight (float): Weight assigned to vector similarity search
                in hybrid search operations. Range [0.0, 1.0], default 0.7.
                Higher values prioritize semantic similarity over keyword matching.
            candidate_multiplier (float): Multiplier applied to max_results when
                retrieving candidates for re-ranking. Default 3.0 means 3x more
                candidates are retrieved than the final result count.
            enable_load_env (bool): Whether to load environment variables from
                .env file. Defaults to False.

        Note:
            The following directory structure will be created:
                - {working_dir}/           - Root working directory
                - {working_dir}/memory/    - Memory storage files
                - {working_dir}/tool_result/ - Compacted tool result files
                - {working_dir}/dialog/    - Raw conversation records
        """
        # Initialize working directory structure
        self.working_path = Path(working_dir).absolute()
        self.working_path.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.working_path / "memory"
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.tool_result_path = self.working_path / "tool_result"
        self.tool_result_path.mkdir(parents=True, exist_ok=True)
        self.dialog_path = self.working_path / "dialog"
        self.dialog_path.mkdir(parents=True, exist_ok=True)

        self.vector_weight: float = vector_weight
        self.candidate_multiplier: float = candidate_multiplier

        # Build the file watcher config: use provided watch_paths if given, otherwise use defaults
        _default_watch_paths = [
            str(self.working_path / "MEMORY.md"),
            str(self.working_path / "memory.md"),
            str(self.memory_path),
        ]
        if default_file_watcher_config and default_file_watcher_config.get("watch_paths"):
            _merged_file_watcher_config = default_file_watcher_config
        else:
            _merged_file_watcher_config = {
                **(default_file_watcher_config or {}),
                "watch_paths": _default_watch_paths,
            }

        # Initialize the parent Application class with comprehensive configuration
        super().__init__(
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            working_dir=str(self.working_path),
            config_path="light",
            enable_logo=False,
            log_to_console=False,
            enable_load_env=enable_load_env,
            parser=ReMeConfigParser,
            default_as_llm_config=default_as_llm_config,
            default_embedding_model_config=default_embedding_model_config,
            default_file_store_config=default_file_store_config,
            default_file_watcher_config=_merged_file_watcher_config,
        )

        # Initialize list to track background summarization tasks
        self.summary_tasks: list[asyncio.Task] = []

    @staticmethod
    def calculate_memory_compact_threshold(max_input_length: float, compact_ratio: float) -> int:
        """Calculate the memory compaction threshold based on input length and ratio.

        Args:
            max_input_length: Maximum input length in tokens.
            compact_ratio: Ratio of the input length to use as the threshold.

        Returns:
            Computed compaction threshold as an integer.
        """
        return int(max_input_length * compact_ratio * 0.95)

    def _cleanup_tool_results(self) -> int:
        """
        Clean up expired tool result files from the tool result directory.

        This method removes tool result files that have exceeded the retention
        period. It helps manage disk space by automatically removing old, unused
        tool outputs.

        Returns:
            int: The number of files that were successfully deleted
        """
        try:
            # Create a compactor instance with default configuration
            compactor = ToolResultCompactor(tool_result_dir=self.tool_result_path)
            # Execute cleanup and return count of deleted files
            return compactor.cleanup_expired_files()
        except Exception as e:
            # Log exception details but return 0 to indicate failure gracefully
            logger.exception(f"Error cleaning up tool results: {e}")
            return 0

    async def start(self):
        """
        Start the application lifecycle.

        Initializes all application components by calling the parent class start
        method, then performs initial cleanup of expired tool result files.

        Returns:
            The result from the parent Application.start() method.

        Note:
            This method should be called before using any other application
            functionality. It ensures all services are properly initialized.
        """
        result = await super().start()
        # Perform initial cleanup of any expired tool result files
        self._cleanup_tool_results()
        return result

    async def close(self) -> bool:
        """
        Close the application and perform cleanup.

        Performs final cleanup of expired tool result files and then shuts down
        all application components by calling the parent class close method.

        Returns:
            bool: True if the application was closed successfully, False otherwise.

        Note:
            This method should be called when the application is no longer needed
            to ensure proper resource cleanup and data persistence.
        """
        # Final cleanup of expired tool result files before shutdown
        self._cleanup_tool_results()
        return await super().close()

    async def compact_tool_result(
        self,
        messages: list[Msg],
        old_max_bytes: int = 3000,
        recent_max_bytes: int = 100 * 1024,
        retention_days: int = 3,
        recent_n: int = 1,
    ) -> list[Msg]:
        """
        Compact tool results by truncating large outputs and saving full content to files.

        This method processes a list of messages containing tool results and compacts
        any that exceed the configured threshold. Large tool outputs are truncated
        in the message while the full content is saved to separate files for later
        retrieval if needed.

        Args:
            messages (list[Msg]): List of messages potentially containing tool results
                that may need compaction.
            old_max_bytes (int): Byte threshold for old (non-recent) messages. Default 3000.
            recent_max_bytes (int): Byte threshold for recent messages (trailing consecutive
                tool-result messages). Default 100KB (102400 bytes). Content exceeding this
                limit is saved to disk; the message retains the first 100KB with a
                read_file-style truncation notice and the saved file path.
            retention_days (int): Number of days to retain tool result files.
                Default 3.
            recent_n (int): Minimum number of most-recent tool-result messages to treat
                as "recent" (using recent_max_bytes). The actual recent window is the
                larger of this value and the trailing consecutive tool-result run.
                Default 1.

        Returns:
            list[Msg]: The processed list of messages with large tool results compacted.
                If an error occurs, returns the original unmodified messages.

        Note:
            - Recent tool results (trailing consecutive tool-result messages) are truncated
              to recent_max_bytes using read_file-style output with a file path hint.
            - Old tool results are truncated to old_max_bytes bytes.
            - Full content of truncated results is saved to tool_result_path.
            - Expired files are automatically cleaned up during this operation.
        """
        try:
            # Create compactor with instance configuration
            compactor = ToolResultCompactor(
                tool_result_dir=self.tool_result_path,
                retention_days=retention_days,
                old_max_bytes=old_max_bytes,
                recent_max_bytes=recent_max_bytes,
                recent_n=recent_n,
            )

            # Execute compaction and get processed messages
            result = await compactor.call(messages=messages, service_context=self.service_context)

            # Clean up any expired tool result files during compaction
            compactor.cleanup_expired_files()

            return result

        except Exception as e:
            # Log the error and return original messages to maintain functionality
            logger.exception(f"Error compacting tool results: {e}")
            return messages

    async def check_context(
        self,
        messages: list[Msg],
        memory_compact_threshold: int,
        memory_compact_reserve: int = 10000,
        as_token_counter: str | HuggingFaceTokenCounter = "default",
    ) -> tuple[list[Msg], list[Msg], bool]:
        """
        Check context size and determine if compaction is needed.

        Analyzes the provided messages to determine if they exceed the configured
        token threshold and splits them into two groups: messages that should be
        compacted and messages to keep in context.

        Args:
            messages (list[Msg]): List of messages to check for context overflow.
            memory_compact_threshold (int): Token count threshold for triggering
                compaction. Messages exceeding this threshold will be split.
            memory_compact_reserve (int): Token count to reserve for recent messages
                to keep in context. Defaults to 10000 tokens.
            as_token_counter (str | HuggingFaceTokenCounter): The token counter to use.

        Returns:
            tuple[list[Msg], list[Msg], bool]: A tuple containing:
                - messages_to_compact (list[Msg]): Older messages that should
                    be compacted/summarized.
                - messages_to_keep (list[Msg]): Recent messages to keep in context.
                - is_valid (bool): True if the split is valid (tool calls aligned),
                    False if splitting would break conversation integrity.

        Note:
            - Returns ([], messages, True) if no compaction is needed.
            - Ensures conversation pairs (user-assistant) are not split.
            - is_valid=False indicates tool_use and tool_result are misaligned.
        """
        try:
            checker = ContextChecker(
                memory_compact_threshold=memory_compact_threshold,
                memory_compact_reserve=memory_compact_reserve,
                as_token_counter=as_token_counter,
            )

            return await checker.call(
                messages=messages,
                service_context=self.service_context,
            )

        except Exception as e:
            logger.exception(f"Error checking context: {e}")
            return [], messages, False

    async def compact_memory(
        self,
        messages: list[Msg],
        as_llm: str | ChatModelBase = "default",
        as_llm_formatter: str | FormatterBase = "default",
        as_token_counter: str | HuggingFaceTokenCounter = "default",
        language: str = "zh",
        max_input_length: float = 128 * 1024,
        compact_ratio: float = 0.7,
        previous_summary: str = "",
        return_dict: bool = False,
        add_thinking_block: bool = True,
        extra_instruction: str = "",
    ) -> str | dict:
        """
        Compact a list of messages into a condensed summary.

        Uses the configured language model to generate a concise summary of the
        provided messages. This is useful for reducing context window usage while
        preserving important information from the conversation history.

        Args:
            messages (list[Msg]): List of messages to be compacted into a summary.
            as_llm (str | ChatModelBase): Language model identifier or instance
                to use for summarization. Defaults to "default".
            as_llm_formatter (str | FormatterBase): Formatter for the language model.
                Defaults to "default".
            as_token_counter (str | HuggingFaceTokenCounter): Token counter for
                measuring message length. Defaults to "default".
            language (str): Language for the summary output. "zh" for Chinese,
                any other value for English. Defaults to "zh".
            max_input_length (float): Maximum input length in tokens for the model.
                Defaults to 128K tokens.
            compact_ratio (float): Ratio used to calculate compaction threshold.
                Defaults to 0.7.
            previous_summary (str): Previous summary to incorporate into the new
                summary for continuity. Defaults to empty string.
            return_dict (bool): If True, returns a dict with user_message,
                history_compact, and is_valid. Defaults to False.
            add_thinking_block (bool): If True, adds a thinking block to the summary.
            extra_instruction (str): Optional additional instruction appended to the
                compaction prompt. Use this to guide what information to keep or
                remove. For example: "Remove debug logs and tool-call details. Keep
                requirements, decisions, and pending tasks." Defaults to empty string
                (no extra instruction, preserving default behavior).

        Returns:
            str | dict: The condensed summary string, or a dict containing
                user_message, history_compact, and is_valid if return_dict=True.
                Returns empty string or dict with empty values if an error occurred.
        """
        try:
            compactor = Compactor(
                memory_compact_threshold=self.calculate_memory_compact_threshold(max_input_length, compact_ratio),
                as_llm=as_llm,
                as_llm_formatter=as_llm_formatter,
                as_token_counter=as_token_counter,
                language=language if language == "zh" else "",
                return_dict=return_dict,
                add_thinking_block=add_thinking_block,
                extra_instruction=extra_instruction,
            )

            return await compactor.call(
                messages=messages,
                previous_summary=previous_summary,
                service_context=self.service_context,
            )

        except Exception as e:
            # Log error and return appropriate empty result
            logger.exception(f"Error compacting memory: {e}")
            if return_dict:
                return {"user_message": str(e), "history_compact": str(e), "is_valid": False}
            return ""

    async def summary_memory(
        self,
        messages: list[Msg],
        as_llm: str | ChatModelBase = "default",
        as_llm_formatter: str | FormatterBase = "default",
        as_token_counter: str | HuggingFaceTokenCounter = "default",
        toolkit: Toolkit | None = None,
        language: str = "zh",
        max_input_length: float = 128 * 1024,
        compact_ratio: float = 0.7,
        timezone: str | None = None,
        add_thinking_block: bool = True,
    ) -> str:
        """
        Generate a comprehensive summary of the given messages.

        Creates a detailed summary of the conversation history and persists it
        to the memory directory as structured files. Unlike compact_memory, this
        method produces more detailed summaries suitable for long-term storage.

        Args:
            messages (list[Msg]): List of messages to summarize.
            as_llm (str | ChatModelBase): Language model identifier or instance
                for summarization. Defaults to "default".
            as_llm_formatter (str | FormatterBase): Formatter for the language model.
                Defaults to "default".
            as_token_counter (str | HuggingFaceTokenCounter): Token counter for
                measuring message length. Defaults to "default".
            toolkit (Toolkit | None): Toolkit with file operations for persisting
                summaries. If None, creates a default toolkit with read/write/edit.
            language (str): Language for the summary output. "zh" for Chinese,
                any other value for English. Defaults to "zh".
            max_input_length (float): Maximum input length in tokens.
                Defaults to 128K tokens.
            compact_ratio (float): Ratio used to calculate compaction threshold.
                Defaults to 0.7.
            timezone (str | None): Timezone string for date formatting
                (e.g., "America/Chicago"). Defaults to system local time if None.

        Returns:
            str: The generated summary text, or an empty string if an error occurred.

        Note:
            This method may write summary files to the memory_path directory
            using the provided or default toolkit.
        """
        try:
            if toolkit is None:
                toolkit = Toolkit()
                file_io = FileIO(working_dir=str(self.working_path))
                toolkit.register_tool_function(file_io.read_file)
                toolkit.register_tool_function(file_io.write_file)
                toolkit.register_tool_function(file_io.edit_file)

            summarizer = Summarizer(
                working_dir=str(self.working_path),
                memory_dir=str(self.memory_path),
                memory_compact_threshold=self.calculate_memory_compact_threshold(max_input_length, compact_ratio),
                toolkit=toolkit,
                as_llm=as_llm,
                as_llm_formatter=as_llm_formatter,
                as_token_counter=as_token_counter,
                language=language if language == "zh" else "",
                timezone=timezone,
                add_thinking_block=add_thinking_block,
            )

            return await summarizer.call(messages=messages, service_context=self.service_context)

        except Exception as e:
            logger.exception(f"Error summarizing memory: {e}")
            return ""

    def add_async_summary_task(self, messages: list[Msg], **kwargs):
        """
        Add an asynchronous summary task for the given messages.

        Creates a background task to generate a summary of the provided messages
        without blocking the main execution flow. Completed tasks are automatically
        cleaned up from the task list.

        Args:
            messages (list[Msg]): List of messages to be summarized asynchronously.
            **kwargs: Additional keyword arguments passed to summary_memory().
                Supported arguments include:
                - as_llm: Language model identifier or instance
                - as_llm_formatter: Formatter for the language model
                - as_token_counter: Token counter instance
                - toolkit: Toolkit for file operations
                - language: Output language ("zh" or other)
                - max_input_length: Maximum input token length
                - compact_ratio: Compaction threshold ratio

        Note:
            - Completed/failed/canceled tasks are cleaned up before adding new ones
            - Task results and errors are logged automatically
            - Use await_summary_tasks() to wait for all pending tasks to complete
        """
        remaining_tasks = []
        for task in self.summary_tasks:
            if task.done():
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.error(f"Summary task failed: {exc}")
                else:
                    result = task.result()
                    logger.info(f"Summary task completed: {result}")
            else:
                remaining_tasks.append(task)
        self.summary_tasks = remaining_tasks

        task = asyncio.create_task(self.summary_memory(messages=messages, **kwargs))
        self.summary_tasks.append(task)

    @property
    def default_as_token_counter(self) -> HuggingFaceTokenCounter:
        """
        Get the default token counter for the memory.

        Returns:
            HuggingFaceTokenCounter: The default token counter instance.
        """
        return self.service_context.as_token_counters["default"]

    async def pre_reasoning_hook(
        self,
        messages: list[Msg],
        system_prompt: str = "",
        compressed_summary: str = "",
        as_llm: str | ChatModelBase = "default",
        as_llm_formatter: str | FormatterBase = "default",
        as_token_counter: str | HuggingFaceTokenCounter = "default",
        toolkit: Toolkit | None = None,
        language: str = "zh",
        max_input_length: float = 128 * 1024,
        compact_ratio: float = 0.7,
        memory_compact_reserve: int = 10000,
        enable_tool_result_compact: bool = True,
        tool_result_compact_keep_n: int = 3,
    ) -> tuple[list[Msg], str]:
        """
        Hook called before reasoning to manage memory and context.

        This method is designed to be called before each reasoning step to ensure
        the conversation context fits within model limits. It performs tool result
        compaction, checks context size, and triggers memory compaction if needed.

        Args:
            messages (list[Msg]): Current conversation messages to be processed.
            system_prompt (str): System prompt that will be included in the context.
                Used to calculate available space. Defaults to empty string.
            compressed_summary (str): Existing compressed summary from previous
                compactions. Defaults to empty string.
            as_llm (str | ChatModelBase): Language model for compaction operations.
                Defaults to "default".
            as_llm_formatter (str | FormatterBase): Formatter for the language model.
                Defaults to "default".
            as_token_counter (str | HuggingFaceTokenCounter): Token counter for
                measuring content length. Defaults to "default".
            toolkit (Toolkit | None): Toolkit for file operations in summarization.
                Defaults to None.
            language (str): Language for generated summaries. Defaults to "zh".
            max_input_length (float): Maximum context window size in tokens.
                Defaults to 128K tokens.
            compact_ratio (float): Ratio for calculating compaction threshold.
                Defaults to 0.7.
            memory_compact_reserve (int): Token count to reserve for new responses.
                Defaults to 10000 tokens.
            enable_tool_result_compact (bool): Whether to compact tool results.
                Defaults to True.
            tool_result_compact_keep_n (int): Number of recent messages to exclude
                from tool result compaction. Defaults to 3.

        Returns:
            tuple[list[Msg], str]: A tuple containing:
                - list[Msg]: Messages to keep in context (maybe reduced)
                - str: Updated compressed summary incorporating compacted messages

        Note:
            - Automatically triggers background summarization for compacted messages
            - Tool results in recent messages (keep_n) are not compacted
            - Returns original messages unchanged if no compaction is needed
        """
        msg_handler = AsMsgHandler(self.default_as_token_counter)

        system_token_count = await msg_handler.count_str_token(system_prompt)
        compressed_token_count = await msg_handler.count_str_token(compressed_summary)
        memory_compact_threshold = self.calculate_memory_compact_threshold(max_input_length, compact_ratio)
        left_compact_threshold = memory_compact_threshold - (system_token_count + compressed_token_count)
        logger.info(f"Left compact threshold: {left_compact_threshold}")

        if enable_tool_result_compact and tool_result_compact_keep_n > 0:
            compact_msgs = messages[:-tool_result_compact_keep_n]
            await self.compact_tool_result(compact_msgs)

        messages_to_compact, messages_to_keep, is_valid = await self.check_context(
            messages=messages,
            memory_compact_threshold=left_compact_threshold,
            memory_compact_reserve=memory_compact_reserve,
            as_token_counter=as_token_counter,
        )

        if not messages_to_compact:
            return messages, compressed_summary

        if not is_valid:
            logger.warning("Invalid messages to compact, skipping.")
            return messages, compressed_summary

        self.add_async_summary_task(
            messages=messages_to_compact,
            as_llm=as_llm,
            as_llm_formatter=as_llm_formatter,
            as_token_counter=as_token_counter,
            toolkit=toolkit,
            language=language,
            max_input_length=max_input_length,
            compact_ratio=compact_ratio,
        )

        compressed_summary = await self.compact_memory(
            messages=messages_to_compact,
            as_llm=as_llm,
            as_llm_formatter=as_llm_formatter,
            as_token_counter=as_token_counter,
            language=language,
            max_input_length=max_input_length,
            compact_ratio=compact_ratio,
            previous_summary=compressed_summary,
        )

        return messages_to_keep, compressed_summary

    async def await_summary_tasks(self) -> str:
        """
        Wait for all background summary tasks to complete and collect results.

        Blocks until all pending summary tasks in the task list have completed,
        canceled, or failed. Collects status information from each task and
        clears the task list after processing.

        Returns:
            str: A concatenated string of status messages for all tasks, including:
                - Completion confirmations with results
                - Cancellation notices
                - Error messages for failed tasks

        Note:
            - This method will block if any tasks are still running
            - All tasks are removed from summary_tasks after this call
            - Task exceptions are logged but do not raise to the caller
            - Use this before application shutdown to ensure all summaries complete
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
                        logger.error(f"Summary task failed: {exc}")
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

    async def memory_search(self, query: str, max_results: int = 5, min_score: float = 0.1) -> ToolResponse:
        """
        Mandatory recall step: semantically search MEMORY.md + memory/*.md
        (and optional session transcripts) before answering questions about
        prior work, decisions, dates, people, preferences, or todos; returns
        top snippets with path + lines.

        Args:
            query (str): The semantic search query to find relevant memory snippets.
            max_results (int): Maximum number of search results to return (optional), default 5.
            min_score (float): Minimum similarity score threshold for results (optional), default 0.1.

        Returns:
            ToolResponse: A ToolResponse containing the search results as text,
                or an error message if the query is empty.
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

        elif isinstance(max_results, str):
            try:
                max_results = min(max(int(max_results), 1), 100)
            except ValueError:
                max_results = 5
        else:
            max_results = 5

        # Validate and clamp min_score to valid range [0.001, 0.999]
        if isinstance(min_score, (int, float)):
            min_score = float(min(max(min_score, 0.001), 0.999))

        elif isinstance(min_score, str):
            try:
                min_score = float(min(max(float(min_score), 0.001), 0.999))
            except ValueError:
                min_score = 0.1

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

    def get_in_memory_memory(self, as_token_counter: HuggingFaceTokenCounter | None = None):
        """
        Create and return an in-memory memory instance.

        Factory method to create a ReMeInMemoryMemory instance configured with
        the specified token counter. This memory instance stores messages in RAM
        during the session, and automatically persists them to dialog_path when
        messages are compressed or cleared.

        Args:
            as_token_counter (HuggingFaceTokenCounter): Token counter for
                measuring content length in the memory.

        Returns:
            ReMeInMemoryMemory: A new in-memory memory instance ready for use.
                The instance is configured with self.dialog_path for persistence.

        Note:
            - Messages are stored in RAM during active session
            - When messages are compressed via mark_messages_compressed(), they
              are persisted to {dialog_path}/{date}.jsonl files
            - When clear_content() is called, all messages are persisted before
              clearing from memory
        """
        return ReMeInMemoryMemory(
            token_counter=as_token_counter or self.default_as_token_counter,
            dialog_path=str(self.dialog_path),
        )
