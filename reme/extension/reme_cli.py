"""ReMe File System"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

from prompt_toolkit import PromptSession

from .config import ReMeConfigParser
from .core import Application
from .core.enumeration import ChunkEnum
from .core.op import BaseTool
from .core.schema import Message, StreamChunk
from .core.tools import (
    BashTool,
    EditTool,
    LsTool,
    ReadTool,
    WriteTool,
    ExecuteCode,
    DashscopeSearch,
    TavilySearch,
)
from .core.utils import execute_stream_task, play_horse_easter_egg
from .memory.cli import FbCli, FbCompactor, FbContextChecker, FbSummarizer
from .memory.tools import MemoryGet, MemorySearch


class ReMeCli(Application):
    """ReMe Cli"""

    def __init__(
        self,
        *args,
        working_dir: str = ".reme",
        config_path: str = "cli",
        enable_logo: bool = True,
        log_to_console: bool = True,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        default_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_file_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
        **kwargs,
    ):
        """Initialize ReMe with config."""
        working_path = Path(working_dir)
        working_path.mkdir(parents=True, exist_ok=True)
        memory_path = working_path / "memory"
        memory_path.mkdir(parents=True, exist_ok=True)
        self.working_dir: str = str(working_path.absolute())

        default_file_watcher_config = default_file_watcher_config or {}
        if not default_file_watcher_config.get("watch_paths", None):
            default_file_watcher_config["watch_paths"] = [
                str(working_path / "MEMORY.md"),
                str(working_path / "memory.md"),
                str(memory_path),
            ]
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            working_dir=working_dir,
            config_path=config_path,
            enable_logo=enable_logo,
            log_to_console=log_to_console,
            parser=ReMeConfigParser,
            default_llm_config=default_llm_config,
            default_embedding_model_config=default_embedding_model_config,
            default_file_store_config=default_file_store_config,
            default_token_counter_config=default_token_counter_config,
            default_file_watcher_config=default_file_watcher_config,
            **kwargs,
        )

        self.service_config.metadata.setdefault("context_window_tokens", context_window_tokens)
        self.service_config.metadata.setdefault("reserve_tokens", reserve_tokens)
        self.service_config.metadata.setdefault("keep_recent_tokens", keep_recent_tokens)
        self.service_config.metadata.setdefault("vector_weight", vector_weight)
        self.service_config.metadata.setdefault("candidate_multiplier", candidate_multiplier)

        self.commands = {
            "/new": "Create a new conversation.",
            "/compact": "Compact messages into a summary.",
            "/exit": "Exit the application.",
            "/clear": "Clear the history.",
            "/help": "Show help.",
            "/horse": "A surprise.",
        }

    async def chat_with_remy(self, tool_result_max_size: int = 100, **kwargs):
        """Interactive CLI chat with Remy using simple streaming output."""
        language = self.service_config.language
        print(f"ReMe language={language or 'default'}")
        tools: list[BaseTool] = [
            MemorySearch(
                vector_weight=self.service_config.metadata["vector_weight"],
                candidate_multiplier=self.service_config.metadata["candidate_multiplier"],
            ),
            BashTool(cwd=self.working_dir),
            LsTool(cwd=self.working_dir),
            ReadTool(cwd=self.working_dir),
            EditTool(cwd=self.working_dir),
            WriteTool(cwd=self.working_dir),
            ExecuteCode(),
        ]
        tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
        dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
        if tavily_api_key:
            tools.append(TavilySearch(name="web_search", language=language))
            print("find tavily_api_key, append Tavily search tool")
        elif dashscope_api_key:
            tools.append(DashscopeSearch(name="web_search", language=language))
            print("find dashscope_api_key, append Dashscope search tool")
        else:
            print("No Tavily or Dashscope API key found, skip Tavily and Dashscope search tool")

        fb_cli = FbCli(
            tools=tools,
            context_window_tokens=self.service_config.metadata["context_window_tokens"],
            reserve_tokens=self.service_config.metadata["reserve_tokens"],
            keep_recent_tokens=self.service_config.metadata["keep_recent_tokens"],
            working_dir=self.working_dir,
            language=language,
            **kwargs,
        )
        session = PromptSession()

        # Print welcome banner
        print("\n========================================")
        print("  Welcome to Remy Chat!")
        print("========================================\n")

        async def chat(q: str) -> AsyncGenerator[StreamChunk, None]:
            """Execute chat query and yield streaming chunks."""
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(
                fb_cli.call(
                    query=q,
                    stream_queue=stream_queue,
                    service_context=self.service_context,
                ),
            )
            async for _chunk in execute_stream_task(
                stream_queue=stream_queue,
                task=task,
                task_name="cli",
                output_format="chunk",
            ):
                yield _chunk

        while True:
            try:
                # Get user input (async)
                user_input = await session.prompt_async("You: ")
                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input == "/exit":
                    break

                if user_input == "/new":
                    result = await fb_cli.new()
                    print(f"{result}\nConversation reset\n")
                    continue

                if user_input == "/compact":
                    result = await fb_cli.compact(force_compact=True)
                    print(f"{result}\nHistory compacted.\n")
                    continue

                if user_input == "/history":
                    result = fb_cli.format_history()
                    print(f"Formated History:\n{result}\n")
                    continue

                if user_input == "/clear":
                    fb_cli.messages.clear()
                    print("History cleared.\n")
                    continue

                if user_input == "/help":
                    print("\nCommands:")
                    for command, description in self.commands.items():
                        print(f"  {command}: {description}")
                    continue

                if user_input == "/horse":
                    play_horse_easter_egg()
                    continue

                # Stream processing state
                in_thinking = False
                in_answer = False

                try:
                    async for chunk in chat(user_input):
                        if chunk.chunk_type == ChunkEnum.THINK:
                            if not in_thinking:
                                print("\033[90mThinking: ", end="", flush=True)
                                in_thinking = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.ANSWER:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            if not in_answer:
                                print("\nRemy: ", end="", flush=True)
                                in_answer = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.TOOL:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            print(f"\033[36m  -> {chunk.chunk}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.TOOL_RESULT:
                            tool_name = chunk.metadata.get("tool_name", "unknown")
                            result = chunk.chunk
                            if len(result) > tool_result_max_size:
                                result = result[:tool_result_max_size] + f"... ({len(chunk.chunk)} chars total)"
                            print(f"\033[36m  -> Tool result for {tool_name}: {result.strip()}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.ERROR:
                            print(f"\n\033[91m[ERROR] {chunk.chunk}\033[0m")
                            # Also log the full error metadata if available
                            if chunk.metadata:
                                import traceback

                                traceback.print_exc()

                        elif chunk.chunk_type == ChunkEnum.DONE:
                            break

                except Exception as e:
                    print(f"\nStream error: {e}")

                # End current streaming line
                print("\n")
                print("----------------------------------------\n")

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()

        print("\nGoodbye!\n")

    async def context_check(self, messages: list[Message | dict]) -> dict:
        """Check if messages exceed context limits."""
        checker = FbContextChecker(
            context_window_tokens=self.service_config.metadata["context_window_tokens"],
            reserve_tokens=self.service_config.metadata["reserve_tokens"],
            keep_recent_tokens=self.service_config.metadata["keep_recent_tokens"],
        )
        return await checker.call(messages=messages, service_context=self.service_context)

    async def compact(
        self,
        messages_to_summarize: list[Message | dict] = None,
        turn_prefix_messages: list[Message | dict] = None,
        previous_summary: str = "",
        language: str = "zh",
        **kwargs,
    ) -> str | dict:
        """Compact messages into a summary."""
        compactor = FbCompactor(language=language, **kwargs)
        return await compactor.call(
            messages_to_summarize=messages_to_summarize or [],
            turn_prefix_messages=turn_prefix_messages or [],
            previous_summary=previous_summary,
            service_context=self.service_context,
        )

    async def summary(
        self,
        messages: list[Message | dict],
        date: str,
        version: str = "default",
        language: str = "zh",
        **kwargs,
    ) -> str | dict:
        """Generate a summary of the given messages."""
        summarizer = FbSummarizer(
            tools=[
                BashTool(cwd=self.working_dir),
                LsTool(cwd=self.working_dir),
                ReadTool(cwd=self.working_dir),
                WriteTool(cwd=self.working_dir),
                EditTool(cwd=self.working_dir),
            ],
            working_dir=self.working_dir,
            language=language,
            version=version,
            **kwargs,
        )
        return await summarizer.call(messages=messages, date=date, service_context=self.service_context)

    async def memory_search(self, query: str, max_results: int = 5, min_score: float = 0.1) -> str:
        """
        Mandatory recall step: semantically search MEMORY.md + memory/*.md (and optional session transcripts)
        before answering questions about prior work, decisions, dates, people, preferences, or todos;
        returns top snippets with path + lines.

        Args:
            query: The semantic search query to find relevant memory snippets
            max_results: Maximum number of search results to return (optional), default is 5
            min_score: Minimum similarity score threshold for results (optional), default is 0.1

        Returns:
            Search results as formatted string
        """
        search_tool = MemorySearch(
            vector_weight=self.service_config.metadata["vector_weight"],
            candidate_multiplier=self.service_config.metadata["candidate_multiplier"],
        )
        return await search_tool.call(
            query=query,
            max_results=max_results,
            min_score=min_score,
            service_context=self.service_context,
        )

    async def memory_get(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        """
        Safe snippet read from MEMORY.md, memory/*.md with optional offset/limit;
        use after memory_search to pull only the needed lines and keep context small.

        Args:
            path: Path to the memory file to read (relative or absolute)
            offset: Starting line number (1-indexed, optional)
            limit: Number of lines to read from the starting line (optional)

        Returns:
            Memory file content as string
        """
        get_tool = MemoryGet(cwd=self.working_dir)
        return await get_tool.call(path=path, offset=offset, limit=limit, service_context=self.service_context)

    async def needs_compaction(self, messages: list[Message | dict]) -> bool:
        """Check if messages need compaction based on context window limits."""
        messages = [Message(**message) if isinstance(message, dict) else message for message in messages]
        checker = FbContextChecker(
            context_window_tokens=self.service_config.metadata["context_window_tokens"],
            reserve_tokens=self.service_config.metadata["reserve_tokens"],
        )
        result = await checker.call(messages=messages, service_context=self.service_context)
        return result["needs_compaction"]


async def async_main():
    """Main function for testing the ReMeFs CLI."""
    async with ReMeCli(*sys.argv[1:], log_to_console=False) as reme:
        await reme.chat_with_remy()


def main():
    """Main function for testing the ReMeFs CLI."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
