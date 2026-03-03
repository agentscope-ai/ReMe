"""ReMe Copaw application class."""

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
from .memory.file_based_copaw import Compactor, Summarizer, ToolResultCompactor
from .memory.tools import MemorySearch

logger = logging.getLogger(__name__)


class ReMeCopaw(Application):
    """ReMe Copaw application class."""

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
    ):

        self.working_path = Path(working_dir).absolute()
        self.working_path.mkdir(parents=True, exist_ok=True)

        self.memory_path = self.working_path / "memory"
        self.memory_path.mkdir(parents=True, exist_ok=True)

        self.tool_result_path = self.working_path / "tool_result"
        self.tool_result_path.mkdir(parents=True, exist_ok=True)

        self.chat_model: ChatModelBase = chat_model
        self.formatter: FormatterBase = formatter
        self.token_counter: HuggingFaceTokenCounter = token_counter
        self.toolkit: Toolkit = toolkit

        self.max_input_length: int | None = None
        self.memory_compact_threshold: int | None = None
        self.language: str | None = None

        self.vector_weight: float = vector_weight
        self.candidate_multiplier: float = candidate_multiplier

        self.update_params(
            max_input_length=max_input_length,
            memory_compact_ratio=memory_compact_ratio,
            language=language,
        )

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

        vector_enabled = bool(embedding_api_key) or bool(embedding_model_name)
        if vector_enabled:
            logger.info("Vector search enabled.")
        else:
            logger.warning(
                "Vector search disabled. Memory search functionality will be restricted. "
                "To enable, configure: EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL_NAME.",
            )
        fts_enabled = os.environ.get("FTS_ENABLED", "true").lower() == "true"

        memory_store_backend = os.environ.get("MEMORY_STORE_BACKEND", "auto")
        if memory_store_backend == "auto":
            memory_backend = "local" if platform.system() == "Windows" else "chroma"
        else:
            memory_backend = memory_store_backend

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

        self.summary_tasks: list[asyncio.Task] = []

    def update_params(
        self,
        max_input_length: int,
        memory_compact_ratio: float,
        language: str,
    ):
        """update each time"""
        self.max_input_length: int = max_input_length
        self.memory_compact_threshold = int(max_input_length * memory_compact_ratio * 0.9)
        if language == "zh":
            self.language = "zh"
        else:
            self.language = ""

    @staticmethod
    def _safe_str(key: str, default: str) -> str:
        """Safely get string from environment variable, return default if not set."""
        return os.environ.get(key, default)

    @staticmethod
    def _safe_int(key: str, default: int) -> int:
        """Safely get int from environment variable, return default on failure."""
        value = os.environ.get(key)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid int value '{value}' for key '{key}', using default {default}")
            return default

    def get_emb_envs(self):
        """Get embedding environment variables."""
        embedding_api_key = self._safe_str("EMBEDDING_API_KEY", "")
        embedding_base_url = self._safe_str("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        embedding_model_name = self._safe_str("EMBEDDING_MODEL_NAME", "")
        embedding_dimensions = self._safe_int("EMBEDDING_DIMENSIONS", 1024)
        embedding_cache_enabled = self._safe_str("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
        embedding_max_cache_size = self._safe_int("EMBEDDING_MAX_CACHE_SIZE", 2000)
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

    async def compact_tool_result(
        self,
        messages: list[Msg],
        tool_result_threshold: int = 10000,
        retention_days: int = 7,
    ) -> list[Msg]:
        """Compact tool results by truncating large outputs and saving full content to files."""
        try:
            compactor = ToolResultCompactor(
                tool_result_dir=self.tool_result_path,
                tool_result_threshold=tool_result_threshold,
                retention_days=retention_days,
            )
            compactor.context["messages"] = messages

            result = await compactor.execute()

            # Optionally cleanup expired files
            compactor.cleanup_expired_files()

            return result

        except Exception as e:
            logger.exception(f"Error compacting tool results: {e}")
            return messages

    async def compact_memory(self, messages: list[Msg], previous_summary: str = "") -> str:
        """Compact the given messages."""
        try:
            compactor = Compactor(
                memory_compact_threshold=self.memory_compact_threshold,
                chat_model=self.chat_model,
                formatter=self.formatter,
                token_counter=self.token_counter,
                language=self.language,
            )

            return await compactor.call(messages=messages, previous_summary=previous_summary)

        except Exception as e:
            logger.exception(f"Error compacting memory: {e}")
            return ""

    async def summary_memory(self, messages: list[Msg]) -> str:
        """Generate a summary of the given messages."""
        try:
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

            return await compactor.call(messages=messages)

        except Exception as e:
            logger.exception(f"Error summarizing memory: {e}")
            return ""

    async def await_summary_tasks(self) -> str:
        """Wait for all summary tasks to complete."""
        result = ""
        for task in self.summary_tasks:
            if task.done():
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    result += "Summary task was cancelled.\n"
                else:
                    exc = task.exception()
                    if exc is not None:
                        logger.exception(f"Summary task failed: {exc}")
                        result += f"Summary task failed: {exc}\n"
                    else:
                        task_result = task.result()
                        logger.info(f"Summary task completed: {task_result}")
                        result += f"Summary task completed: {task_result}\n"

            else:
                try:
                    task_result = await task
                    logger.info(f"Summary task completed: {task_result}")
                    result += f"Summary task completed: {task_result}\n"

                except Exception as e:
                    logger.exception(f"Summary task failed: {e}")
                    result += f"Summary task failed: {e}\n"

        self.summary_tasks.clear()
        return result

    def add_async_summary_task(self, messages: list[Msg]):
        """Add an async summary task for the given messages."""
        # Clean up completed summary tasks
        remaining_tasks = []
        for task in self.summary_tasks:
            if task.done():
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.exception(f"Summary task failed: {exc}")
                else:
                    result = task.result()
                    logger.info(f"Summary task completed: {result}")
            else:
                remaining_tasks.append(task)
        self.summary_tasks = remaining_tasks

        task = asyncio.create_task(self.summary_memory(messages=messages))
        self.summary_tasks.append(task)

    async def memory_search(self, query: str, max_results: int = 5, min_score: float = 0.1) -> ToolResponse:
        """memory search"""
        if not query:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: No query provided.",
                    ),
                ],
            )

        if isinstance(max_results, int):
            max_results = min(max(max_results, 1), 100)
        else:
            max_results = 5

        if isinstance(min_score, float):
            min_score = min(max(min_score, 0.001), 0.999)
        else:
            min_score = 0.1

        search_tool = MemorySearch(
            vector_weight=self.vector_weight,
            candidate_multiplier=self.candidate_multiplier,
        )

        search_result = await search_tool.call(
            query=query,
            max_results=max_results,
            min_score=min_score,
            service_context=self.service_context,
        )

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=search_result,
                ),
            ],
        )
