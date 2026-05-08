"""Summarizer module for memory summarization operations."""

import datetime
import json
import zoneinfo

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.token import HuggingFaceTokenCounter
from agentscope.tool import Toolkit
from loguru import logger

from ..component import BaseStep
from ..schema import AsMsgStat, AsBlockStat


class Summarizer(BaseStep):
    """Summarizer step for summarizing memory messages."""

    def __init__(
        self,
        working_dir: str,
        memory_dir: str,
        memory_compact_threshold: int,
        toolkit: Toolkit | None = None,
        console_enabled: bool = False,
        timezone: str | None = None,
        add_thinking_block: bool = True,
        as_token_counter: HuggingFaceTokenCounter | None = None,
        **kwargs,
    ):
        """Initialize the summarizer step.

        Args:
            working_dir: Working directory path.
            memory_dir: Memory directory path for storing summaries.
            memory_compact_threshold: Token threshold for memory compaction.
            toolkit: Optional toolkit for the agent.
            console_enabled: Whether to enable console output.
            timezone: Optional timezone string for date formatting.
            add_thinking_block: Whether to include thinking blocks in output.
            as_token_counter: Optional token counter instance.
            **kwargs: Additional keyword arguments passed to BaseStep.
        """
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        self.memory_dir: str = memory_dir
        self.memory_compact_threshold: int = memory_compact_threshold
        self.toolkit: Toolkit | None = toolkit
        self.console_enabled: bool = console_enabled
        self.timezone: str | None = timezone
        self.add_thinking_block: bool = add_thinking_block
        self._as_token_counter: HuggingFaceTokenCounter | None = as_token_counter

    def _get_current_datetime(self) -> datetime.datetime:
        """Get current datetime with timezone, fallback to local time if timezone is invalid."""
        if self.timezone:
            try:
                return datetime.datetime.now(zoneinfo.ZoneInfo(self.timezone))
            except Exception as e:
                self.logger.error(f"Invalid timezone: {self.timezone}, falling back to local time error={e}")
        return datetime.datetime.now()

    async def _count_str_token(self, text: str) -> int:
        """Count tokens in a string."""
        return await self.as_token_counter.count(messages=[], text=text)

    async def _format_tool_result_output(self, output: str | list[dict]) -> tuple[str, int]:
        """Convert tool result output to string."""
        if isinstance(output, str):
            return output, await self._count_str_token(output)

        textual_parts = []
        total_token_count = 0
        for block in output:
            try:
                if not isinstance(block, dict) or "type" not in block:
                    logger.warning(
                        f"Invalid block: {block}, expected a dict with 'type' key, skipped.",
                    )
                    continue

                block_type = block["type"]

                if block_type == "text":
                    textual_parts.append(block.get("text", ""))
                    total_token_count += await self._count_str_token(textual_parts[-1])

                elif block_type in ["image", "audio", "video"]:
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        data = source.get("data", "")
                        total_token_count += len(data) // 4 if data else 10
                    else:
                        url = source.get("url", "")
                        total_token_count += await self._count_str_token(url) if url else 10
                        textual_parts.append(f"[{block_type}] {url}")

                elif block_type == "file":
                    file_path = block.get("path", "") or block.get("url", "")
                    file_name = block.get("name", file_path)
                    textual_parts.append(f"[file] {file_name}: {file_path}")
                    total_token_count += await self._count_str_token(file_path)

                else:
                    logger.warning(
                        f"Unsupported block type '{block_type}' in tool result, skipped.",
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to process block {block}: {e}, skipped.",
                )

        return "\n".join(textual_parts), total_token_count

    async def _stat_message(self, message: Msg) -> AsMsgStat:
        """Analyze a message and generate block statistics."""
        blocks = []
        if isinstance(message.content, str):
            blocks.append(
                AsBlockStat(
                    block_type="text",
                    text=message.content,
                    token_count=await self._count_str_token(message.content),
                ),
            )
            return AsMsgStat(
                name=message.name or message.role,
                role=message.role,
                content=blocks,
                timestamp=message.timestamp or "",
                metadata=message.metadata or {},
            )

        for block in message.content:
            block_type = block.get("type", "unknown")

            if block_type == "text":
                text = block.get("text", "")
                token_count = await self._count_str_token(text)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text=text,
                        token_count=token_count,
                    ),
                )

            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                token_count = await self._count_str_token(thinking)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text=thinking,
                        token_count=token_count,
                    ),
                )

            elif block_type in ("image", "audio", "video"):
                source = block.get("source", {})
                url = source.get("url", "")
                if source.get("type") == "base64":
                    data = source.get("data", "")
                    token_count = len(data) // 4 if data else 10
                else:
                    token_count = await self._count_str_token(url) if url else 10
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        media_url=url,
                    ),
                )

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", "")
                try:
                    input_str = json.dumps(tool_input, ensure_ascii=False)
                except (TypeError, ValueError):
                    input_str = str(tool_input)
                token_count = await self._count_str_token(tool_name + input_str)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        tool_name=tool_name,
                        tool_input=input_str,
                    ),
                )

            elif block_type == "tool_result":
                tool_name = block.get("name", "")
                output = block.get("output", "")
                formatted_output, token_count = await self._format_tool_result_output(output)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        tool_name=tool_name,
                        tool_output=formatted_output,
                    ),
                )

            else:
                logger.warning(f"Unsupported block type {block_type}, skipped.")

        return AsMsgStat(
            name=message.name or message.role,
            role=message.role,
            content=blocks,
            timestamp=message.timestamp or "",
            metadata=message.metadata or {},
        )

    async def _count_msgs_token(self, messages: list[Msg]) -> int:
        """Count total token count of a list of messages."""
        total = 0
        for msg in messages:
            stat = await self._stat_message(msg)
            total += stat.total_tokens
        return total

    async def _format_msgs_to_str(
        self,
        messages: list[Msg],
        memory_compact_threshold: int,
        include_thinking: bool = True,
    ) -> str:
        """Format list of messages to a single formatted string.

        Messages are processed in reverse order (newest first) and older
        messages are skipped when token count exceeds memory_compact_threshold.

        Args:
            messages: List of Msg objects to format.
            memory_compact_threshold: Maximum token count before skipping older messages.
            include_thinking: Whether to include thinking blocks in output.
        """
        if not messages:
            return ""

        formatted_parts: list[str] = []
        total_token_count = 0

        for i in range(len(messages) - 1, -1, -1):
            stat = await self._stat_message(messages[i])
            formatted_content = stat.format(include_thinking=include_thinking)
            content_token_count = await self._count_str_token(formatted_content)

            is_latest = i == len(messages) - 1
            if not is_latest and total_token_count + content_token_count > memory_compact_threshold:
                logger.info(
                    f"Skipping older messages: adding {content_token_count} tokens would exceed threshold "
                    f"{memory_compact_threshold} (current: {total_token_count})",
                )
                break

            if is_latest and content_token_count > memory_compact_threshold:
                logger.warning(
                    f"Latest message alone ({content_token_count} tokens) exceeds threshold "
                    f"{memory_compact_threshold}, including it anyway.",
                )

            formatted_parts.append(formatted_content)
            total_token_count += content_token_count

        formatted_parts.reverse()
        return "\n\n".join(formatted_parts)

    async def execute(self):
        """Execute the summarization step."""
        messages: list[Msg] = self.context.data.get("messages", [])

        if not messages:
            return ""

        before_token_count = await self._count_msgs_token(messages)
        history_formatted_str: str = await self._format_msgs_to_str(
            messages=messages,
            memory_compact_threshold=self.memory_compact_threshold,
            include_thinking=self.add_thinking_block,
        )
        after_token_count = await self._count_str_token(history_formatted_str)
        logger.info(f"Summarizer before_token_count={before_token_count} after_token_count={after_token_count}")

        if not history_formatted_str:
            logger.warning(f"No history to summarize. messages={messages}")
            return ""

        agent = ReActAgent(
            name="reme_summarizer",
            model=self.as_llm.model,
            sys_prompt="You are a helpful assistant.",
            formatter=self.as_llm_formatter.formatter,
            toolkit=self.toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_message: str = f"# conversation\n{history_formatted_str}\n\n" + self.prompt_format(
            "user_message",
            date=self._get_current_datetime().strftime("%Y-%m-%d"),
            working_dir=self.working_dir,
            memory_dir=self.memory_dir,
        )

        summary_msg: Msg = await agent.reply(
            Msg(
                name="reme",
                role="user",
                content=user_message,
            ),
        )
        for i, (msg, _) in enumerate(agent.memory.content):
            logger.info(f"Summarizer memory[{i}]: {msg.content}")

        history_summary: str = summary_msg.get_text_content()
        logger.info(f"Summarizer Result:\n{history_summary}")
        return history_summary
