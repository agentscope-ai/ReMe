"""CLI component for interactive chat using agentscope-based memory tools."""

import asyncio
from datetime import datetime
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse
from agentscope.pipeline import stream_printing_messages
from loguru import logger

from ....core.op import BaseOp
from ....core.utils import format_messages
from .compactor import Compactor
from .context_checker import ContextChecker
from .summarizer import Summarizer
from ..tools import FileIO, MemorySearch


class CliAgent(BaseOp):
    """CLI agent for interactive chat with memory management."""

    def __init__(
        self,
        working_dir: str,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
        language: str = "zh",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.vector_weight: float = vector_weight
        self.candidate_multiplier: float = candidate_multiplier
        self.context_window_tokens: int = context_window_tokens
        self.reserve_tokens: int = reserve_tokens
        self.keep_recent_tokens: int = keep_recent_tokens
        self.language: str = language

        # Initialize message history
        self.messages: list[Msg] = []
        self.previous_summary: str = ""
        self.summary_tasks: list[asyncio.Task] = []

    def add_summary_task(self, messages: list[Msg]):
        """Add summary task to queue."""
        remaining_tasks = []
        for task in self.summary_tasks:
            if task.done():
                exc = task.exception()
                if exc is not None:
                    logger.exception(f"Summary task failed: {exc}")
                else:
                    result = task.result()
                    logger.info(f"Summary task completed: {result}")
            else:
                remaining_tasks.append(task)
        self.summary_tasks = remaining_tasks

        # Create a toolkit for the summarizer
        toolkit = self._create_file_toolkit()

        # Create summarizer instance
        memory_path = Path(self.working_dir) / "memory"
        summarizer = Summarizer(
            working_dir=self.working_dir,
            memory_dir=str(memory_path),
            memory_compact_threshold=int(self.context_window_tokens * 0.7),
            token_counter=self.as_token_counter,
            toolkit=toolkit,
            as_llm=self.as_llm,
            as_llm_formatter=self.as_llm_formatter,
            language=self.language if self.language == "zh" else "",
            console_enabled=False,  # We disable the terminal printing to avoid messy outputs
        )

        # Create summary task
        summary_task = asyncio.create_task(
            summarizer.call(
                messages=messages,
                service_context=self.service_context,
            ),
        )
        self.summary_tasks.append(summary_task)

    def _create_file_toolkit(self):
        """Create a toolkit with file operations."""

        toolkit = Toolkit()
        file_io = FileIO(working_dir=self.working_dir)
        toolkit.register_tool_function(file_io.read)
        toolkit.register_tool_function(file_io.write)
        toolkit.register_tool_function(file_io.edit)

        return toolkit

    async def new(self) -> str:
        """Reset conversation history using summary."""
        if not self.messages:
            self.messages.clear()
            self.previous_summary = ""
            return "No history to reset."

        self.add_summary_task(self.messages)

        self.messages.clear()
        self.previous_summary = ""
        return "History saved to memory files and reset."

    async def context_check(self) -> dict:
        """Check if messages exceed token limits."""
        # Create context checker
        checker = ContextChecker(
            memory_compact_threshold=self.context_window_tokens - self.reserve_tokens,
            memory_compact_reserve=self.reserve_tokens,
            token_counter=self.as_token_counter,
        )

        return await checker.call(
            messages=self.messages,
            service_context=self.service_context,
        )

    async def compact(self, force_compact: bool = False) -> str:
        """Compact history then reset."""
        if not self.messages:
            return "No history to compact."

        # Check and find cut point
        messages_to_compact, messages_to_keep, _ = await self.context_check()
        tokens_before = len(self.messages)

        if force_compact:
            messages_to_summarize = self.messages
            left_messages = []
        elif not messages_to_compact:
            return "History is within token limits, no compaction needed."
        else:
            messages_to_summarize = messages_to_compact
            left_messages = messages_to_keep

        # Create compactor
        compactor = Compactor(
            memory_compact_threshold=self.context_window_tokens - self.reserve_tokens,
            token_counter=self.as_token_counter,
            as_llm=self.as_llm,
            as_llm_formatter=self.as_llm_formatter,
            language=self.language if self.language == "zh" else "",
            console_enabled=False,  # We disable the terminal printing to avoid messy outputs
        )

        summary_content = await compactor.call(
            messages=messages_to_summarize,
            previous_summary=self.previous_summary,
            service_context=self.service_context,
        )

        self.add_summary_task(messages=messages_to_summarize)

        # Assemble final messages
        self.messages = left_messages
        self.previous_summary = summary_content

        return f"History compacted from {tokens_before} messages."

    def format_history(self) -> str:
        """Format history messages."""
        return format_messages(
            messages=self.messages,
            add_index=False,
            add_reasoning=False,
            strip_markdown_headers=False,
        )

    async def _build_messages(self, query: str) -> list[Msg]:
        """Build system prompt message."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

        # Create system prompt
        system_prompt = self.prompt_format(
            "system_prompt",
            workspace_dir=self.working_dir,
            current_time=current_time,
            has_previous_summary=bool(self.previous_summary),
            previous_summary=self.previous_summary or "",
        )

        logger.info(f"[{self.__class__.__name__}] system_prompt: {system_prompt}")

        # Build message list
        messages = [Msg(name="system", role="system", content=system_prompt)]
        messages.extend(self.messages)
        messages.append(Msg(name="user", role="user", content=query))

        return messages

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

    async def execute(self):
        """Execute the agent."""
        _ = await self.compact(force_compact=False)

        # Build messages for the agent
        query = self.context.query
        messages = await self._build_messages(query)

        toolkit = self._create_file_toolkit()
        # Register memory search tool
        toolkit.register_tool_function(self.memory_search)

        # Create the ReAct agent
        agent = ReActAgent(
            name="reme_cli_agent",
            model=self.as_llm,
            sys_prompt=messages[0].content,  # System prompt
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )

        # We disable the terminal printing to avoid messy outputs
        agent.set_console_output_enabled(False)

        self.messages = messages[1:]  # remove the first SYSTEM message

        # Stream processing state
        in_thinking = False
        in_answer = False

        # obtain the printing messages from the agent in a streaming way
        last_text_content = ""
        last_think_content = ""
        async for msg, last in stream_printing_messages(
            agents=[agent],
            coroutine_task=agent(self.messages),
        ):
            # print(msg, last)
            content_blocks = msg.get_content_blocks()
            for block in content_blocks:
                if block["type"] == "thinking":
                    if not in_thinking and len(block["thinking"]) > len(last_think_content):
                        print("\033[90m\nThinking: ", end="", flush=True)
                        in_thinking = True
                    print(block["thinking"][len(last_think_content) :], end="", flush=True)
                    last_think_content = block["thinking"]
                elif block["type"] == "text":
                    if in_thinking:
                        print("\033[0m")  # reset color after thinking
                        in_thinking = False
                    if not in_answer:
                        print("\nRemy: ", end="", flush=True)
                        in_answer = True
                    print(block["text"][len(last_text_content) :], end="", flush=True)
                    last_text_content = block["text"]
                elif block["type"] == "tool_use":
                    if in_thinking:
                        print("\033[0m")  # reset color after thinking
                        in_thinking = False
                    if last:
                        print(f"\033[36m  -> Executing Tool: name={block['name']}, input={block['input']}\033[0m")
                elif block["type"] == "tool_result":
                    if last:
                        last_think_content = ""  # reset for further thinking
                        print(f"\033[36m  -> Tool Result for `{block['name']}`: {block['output'][0]['text']}\033[0m")
                else:
                    print(f"Unknown block type: {block['type']}")
            if last:
                self.messages.append(msg)
