"""FbCli system prompt"""

import asyncio
from datetime import datetime
from pathlib import Path

from loguru import logger

from ...core.enumeration import Role, ChunkEnum
from ...core.op import BaseReactStream
from ...core.schema import Message, StreamChunk
from ...core.tools import BashTool, LsTool, ReadTool, WriteTool, EditTool
from ...core.utils import format_messages


class FbCli(BaseReactStream):
    """FbCli agent with system prompt."""

    def __init__(
        self,
        working_dir: str,
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.context_window_tokens: int = context_window_tokens
        self.reserve_tokens: int = reserve_tokens
        self.keep_recent_tokens: int = keep_recent_tokens

        self.messages: list[Message] = []
        self.previous_summary: str = ""
        self.summary_tasks: list[asyncio.Task] = []

    def add_summary_task(self, messages: list[Message]):
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

        from .fb_summarizer import FbSummarizer

        # Summarize current conversation and save to memory files
        current_date = datetime.now().strftime("%Y-%m-%d")
        summarizer = FbSummarizer(
            tools=[
                BashTool(cwd=self.working_dir),
                LsTool(cwd=self.working_dir),
                ReadTool(cwd=self.working_dir),
                WriteTool(cwd=self.working_dir),
                EditTool(cwd=self.working_dir),
            ],
            working_dir=self.working_dir,
            language=self.language,
        )

        summary_task = asyncio.create_task(
            summarizer.call(
                messages=messages,
                date=current_date,
                service_context=self.service_context,
            ),
        )
        self.summary_tasks.append(summary_task)

    async def new(self) -> str:
        """Reset conversation history using summary.

        Summarizes current messages to memory files and clears history.
        """
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
        # Import required modules
        from .fb_context_checker import FbContextChecker

        # Step 1: Check and find cut point
        checker = FbContextChecker(
            context_window_tokens=self.context_window_tokens,
            reserve_tokens=self.reserve_tokens,
            keep_recent_tokens=self.keep_recent_tokens,
        )
        return await checker.call(messages=self.messages, service_context=self.service_context)

    async def compact(self, force_compact: bool = False) -> str:
        """Compact history then reset.

        First compacts messages if they exceed token limits (generating a summary),
        then calls reset_history to save to files and clear.

        Args:
            force_compact: If True, force compaction of all messages into summary

        Returns:
            str: Summary of compaction result
        """
        if not self.messages:
            return "No history to compact."

        # Import required modules
        from .fb_compactor import FbCompactor

        # Step 1: Check and find cut point
        cut_result = await self.context_check()
        tokens_before = cut_result.get("token_count", 0)

        if force_compact:
            messages_to_summarize = self.messages
            turn_prefix_messages = []
            left_messages = []
        elif not cut_result.get("needs_compaction", False):
            return "History is within token limits, no compaction needed."
        else:
            messages_to_summarize = cut_result.get("messages_to_summarize", [])
            turn_prefix_messages = cut_result.get("turn_prefix_messages", [])
            left_messages = cut_result.get("left_messages", [])

        compactor = FbCompactor(language=self.language)
        summary_content = await compactor.call(
            messages_to_summarize=messages_to_summarize,
            turn_prefix_messages=turn_prefix_messages,
            previous_summary=self.previous_summary,
            service_context=self.service_context,
        )

        self.add_summary_task(messages=messages_to_summarize)

        # Step 4: Assemble final messages
        self.messages = left_messages
        self.previous_summary = summary_content

        return f"History compacted from {tokens_before} tokens."

    def format_history(self) -> str:
        """Format history messages."""
        return format_messages(
            messages=self.messages,
            add_index=False,
            add_reasoning=False,
            strip_markdown_headers=False,
        )

    async def build_messages(self) -> list[Message]:
        """Build system prompt message."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
        has_web_search = any(t.name == "web_search" for t in self.tools)

        system_prompt = self.prompt_format(
            "system_prompt",
            workspace_dir=self.working_dir,
            current_time=current_time,
            has_web_search=has_web_search,
            has_previous_summary=bool(self.previous_summary),
            previous_summary=self.previous_summary or "",
        )
        logger.info(f"[{self.__class__.__name__}] system_prompt: {system_prompt}")

        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            *self.messages,
            Message(role=Role.USER, content=self.context.query),
        ]

    async def execute(self):
        """Execute the agent."""
        _ = await self.compact(force_compact=False)

        messages = await self.build_messages()
        for i, message in enumerate(messages):
            role = message.name or message.role
            logger.info(f"[{self.__class__.__name__}] msg[{i}] role={role} {message.simple_dump(as_dict=False)}")

        t_tools, messages, success = await self.react(messages, self.tools)

        # Update self.messages: react() returns [SYSTEM, ...history...],
        # so we remove the first SYSTEM message
        self.messages = messages[1:]

        # Emit final done signal
        await self.context.add_stream_chunk(
            StreamChunk(
                chunk_type=ChunkEnum.DONE,
                chunk="",
                metadata={
                    "success": success,
                    "total_steps": len(t_tools),
                },
            ),
        )

        return {
            "answer": messages[-1].content if success else "",
            "success": success,
            "messages": messages,
            "tools": t_tools,
        }
