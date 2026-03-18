"""ContextChecker module for checking context size and splitting messages."""

from agentscope.message import Msg

from ..utils import AsMsgHandler
from ....core.op import BaseOp
from ....core.utils import get_logger

logger = get_logger()


class ContextChecker(BaseOp):
    """
    ContextChecker class for checking context size and splitting messages.

    This class analyzes conversation messages to determine if the context
    exceeds the specified token threshold and splits messages into two groups:
    those that should be compacted and those to keep in context.

    Attributes:
        memory_compact_threshold (int): Token count threshold for triggering compaction.
        memory_compact_reserve (int): Token count to reserve for recent messages.
    """

    def __init__(
        self,
        memory_compact_threshold: int,
        memory_compact_reserve: int = 10000,
        **kwargs,
    ):
        """
        Initialize the ContextChecker.

        Args:
            memory_compact_threshold (int): Token count threshold for triggering
                compaction. Messages exceeding this threshold will be split.
            memory_compact_reserve (int): Token count to reserve for recent messages
                to keep in context. Defaults to 10000 tokens.
            **kwargs: Additional keyword arguments passed to BaseOp.
        """
        super().__init__(**kwargs)
        self.memory_compact_threshold: int = memory_compact_threshold
        self.memory_compact_reserve: int = memory_compact_reserve
        assert self.memory_compact_threshold > self.memory_compact_reserve

    async def execute(self) -> tuple[list[Msg], list[Msg], bool]:
        """
        Execute context check and split messages.

        Retrieves messages from context and checks if they exceed the token
        threshold. If so, splits them into messages to compact and messages
        to keep.

        Context Parameters:
            messages (list[Msg]): List of conversation messages to check.
                Retrieved from self.context.get("messages", []).

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
        messages: list[Msg] = self.context.get("messages", [])

        if not messages:
            logger.info("ContextChecker: No messages to check.")
            return [], [], True

        msg_handler = AsMsgHandler(self.as_token_counter)
        messages_to_compact, messages_to_keep, is_valid = await msg_handler.context_check(
            messages=messages,
            memory_compact_threshold=self.memory_compact_threshold,
            memory_compact_reserve=self.memory_compact_reserve,
        )

        if messages_to_compact:
            logger.info(
                f"ContextChecker Result: "
                f"to_compact={len(messages_to_compact)}, "
                f"to_keep={len(messages_to_keep)}, "
                f"is_valid={is_valid}",
            )

        return messages_to_compact, messages_to_keep, is_valid
