"""Custom memory implementation with bugfixes and extensions."""

from agentscope.agent._react_agent import _MemoryMark  # noqa
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.token import HuggingFaceTokenCounter

from .utils import AsMsgHandler
from ...core.utils import get_std_logger

logger = get_std_logger()


class ReMeInMemoryMemory(InMemoryMemory):
    """Extended InMemoryMemory with bugfixes and summary support."""

    def __init__(self, token_counter: HuggingFaceTokenCounter):
        super().__init__()
        self._token_counter: HuggingFaceTokenCounter = token_counter
        self._msg_handler: AsMsgHandler = AsMsgHandler(token_counter)

    async def get_memory(
        self,
        mark: str | None = None,
        exclude_mark: str | None = _MemoryMark.COMPRESSED,
        prepend_summary: bool = True,
        **_kwargs,
    ) -> list[Msg]:
        """Get the messages from the memory by mark (if provided).

        Args:
            mark: Optional mark to filter messages
            exclude_mark: Optional mark to exclude messages
            prepend_summary: Whether to prepend compressed summary
            **_kwargs: Additional keyword arguments (ignored)

        Returns:
            List of filtered messages
        """
        if not (mark is None or isinstance(mark, str)):
            raise TypeError(f"The mark should be a string or None, but got {type(mark)}.")

        if not (exclude_mark is None or isinstance(exclude_mark, str)):
            raise TypeError(f"The exclude_mark should be a string or None, but got {type(exclude_mark)}.")

        # Filter messages based on mark
        filtered_content = [(msg, marks) for msg, marks in self.content if mark is None or mark in marks]

        # Further filter messages based on exclude_mark
        if exclude_mark is not None:
            filtered_content = [(msg, marks) for msg, marks in filtered_content if exclude_mark not in marks]

        if prepend_summary and self._compressed_summary:
            previous_summary = f"""
<previous-summary>
{self._compressed_summary}
</previous-summary>
The above is a summary of our previous conversation.
Use it as context to maintain continuity.
                    """.strip()

            return [
                Msg(
                    "user",
                    previous_summary,
                    "user",
                ),
                *[msg for msg, _ in filtered_content],
            ]

        return [msg for msg, _ in filtered_content]

    def get_compressed_summary(self) -> str:
        """Get the compressed summary of the memory."""
        return self._compressed_summary

    def state_dict(self) -> dict:
        """Get the state dictionary for serialization."""
        return {
            "content": [[msg.to_dict(), marks] for msg, marks in self.content],
            "_compressed_summary": self._compressed_summary,
        }

    # pylint: disable=attribute-defined-outside-init
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load the state dictionary for deserialization."""
        if strict and "content" not in state_dict:
            raise KeyError("The state_dict does not contain 'content' key required for InMemoryMemory.")

        self.content = []  # pylint: disable=attribute-defined-outside-init
        for item in state_dict.get("content", []):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                msg_dict, marks = item
                msg = Msg.from_dict(msg_dict)
                self.content.append((msg, marks))

            elif isinstance(item, dict):
                # For compatibility with older versions
                msg = Msg.from_dict(item)
                self.content.append((msg, []))

            else:
                raise ValueError("Invalid item format in state_dict for InMemoryMemory.")

        self._compressed_summary = state_dict.get("_compressed_summary", "")

    async def mark_messages_compressed(self, messages: list[Msg]) -> int:
        """Mark messages as compressed and return count."""
        return await self.update_messages_mark(
            new_mark=_MemoryMark.COMPRESSED,
            msg_ids=[msg.id for msg in messages],
        )

    def clear_compressed_summary(self):
        """Clear the compressed summary."""
        self._compressed_summary = ""  # pylint: disable=attribute-defined-outside-init

    def clear_content(self):
        """Clear the content."""
        self.content.clear()

    async def estimate_tokens(self, max_input_length: int) -> dict:
        """Estimate token usage for current memory.

        Args:
            max_input_length: Max input length for context usage calculation.

        Returns:
            Dict containing detailed token statistics:
            - total_messages: Number of messages
            - compressed_summary_tokens: Tokens in compressed summary
            - messages_tokens: Tokens in messages
            - estimated_tokens: Total estimated tokens
            - max_input_length: Max input length from config
            - context_usage_ratio: Usage percentage
            - messages_detail: List of per-message AsMsgStat objects
        """
        messages = await self.get_memory(
            exclude_mark=_MemoryMark.COMPRESSED,
            prepend_summary=False,
        )

        compressed_summary = self.get_compressed_summary()
        compressed_summary_tokens = self._msg_handler.count_str_token(compressed_summary)

        # Build per-message token details using AsMsgHandler
        messages_detail = [self._msg_handler.stat_message(msg) for msg in messages]

        # Calculate total message tokens from stats
        messages_tokens = sum(stat.total_tokens for stat in messages_detail)
        estimated_tokens = messages_tokens + compressed_summary_tokens

        # Calculate context usage ratio
        context_usage_ratio = (estimated_tokens / max_input_length * 100) if max_input_length > 0 else 0

        return {
            "total_messages": len(messages),
            "compressed_summary_tokens": compressed_summary_tokens,
            "messages_tokens": messages_tokens,
            "estimated_tokens": estimated_tokens,
            "max_input_length": max_input_length,
            "context_usage_ratio": context_usage_ratio,
            "messages_detail": messages_detail,
        }

    async def get_history_str(self, max_input_length: int) -> str:
        """Get formatted history string similar to /history command output.

        Args:
            max_input_length: Max input length for context usage calculation.

        Returns:
            Formatted string containing conversation history details
        """
        stats = await self.estimate_tokens(max_input_length)

        lines = []
        for i, msg_stat in enumerate(stats["messages_detail"], 1):
            blocks_info = ""
            if msg_stat.content:
                block_strs = [f"{b.block_type}(tokens={b.token_count})" for b in msg_stat.content]
                blocks_info = f"\n    content: [{', '.join(block_strs)}]"

            lines.append(
                f"[{i}] **{msg_stat.role}** "
                f"(total_tokens={msg_stat.total_tokens})"
                f"{blocks_info}\n    preview: {msg_stat.preview}",
            )

        return (
            f"**Conversation History**\n\n"
            f"- Total messages: {stats['total_messages']}\n"
            f"- Estimated tokens: {stats['estimated_tokens']}\n"
            f"- Max input length: {stats['max_input_length']}\n"
            f"- Context usage: {stats['context_usage_ratio']:.1f}%\n"
            f"- Compressed summary tokens: {stats['compressed_summary_tokens']}\n\n" + "\n\n".join(lines)
        )
