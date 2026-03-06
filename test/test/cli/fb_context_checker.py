"""Context window limit checker for reactive agents."""

from loguru import logger

from ...core.enumeration import Role
from ...core.op import BaseOp
from ...core.schema import CutPointResult, Message


class FbContextChecker(BaseOp):
    """Check if context exceeds token limits and find cut point for compaction."""

    def __init__(
        self,
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
        **kwargs,
    ):
        """
        Initialize context checker.

        Args:
            context_window_tokens: Total context window size.
            reserve_tokens: Tokens to reserve for output and overhead.
            keep_recent_tokens: Tokens to keep in recent messages.
            **kwargs: Additional BaseReact arguments.
        """
        super().__init__(tools=[], **kwargs)
        self.context_window_tokens: int = context_window_tokens
        self.reserve_tokens: int = reserve_tokens
        self.keep_recent_tokens: int = keep_recent_tokens

    @staticmethod
    def _normalize_messages(messages: list[Message | dict]) -> list[Message]:
        """Convert dict messages to Message objects."""
        return [Message(**m) if isinstance(m, dict) else m for m in messages]

    @staticmethod
    def _is_user_message(message: Message) -> bool:
        """Check if message is user role."""
        return message.role is Role.USER

    def _find_turn_start_index(self, messages: list[Message], entry_index: int) -> int:
        """Find user message that starts the turn. Returns -1 if not found."""
        if not messages or entry_index < 0 or entry_index >= len(messages):
            return -1

        for i in range(entry_index, -1, -1):
            if self._is_user_message(messages[i]):
                return i
        return -1

    def _find_cut_point(
        self,
        messages: list[Message],
        token_count: int,
        threshold: int,
    ) -> CutPointResult:
        """
        Find cut point with split turn detection.

        Split turn: User → Assistant → [CUT] → Assistant → User
        Clean cut: User → [CUT] → Assistant → User
        """
        if not messages:
            return CutPointResult(
                needs_compaction=False,
                token_count=token_count,
                threshold=threshold,
            )

        accumulated_tokens = 0
        cut_index = 0

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            msg_tokens = self.token_counter.count_token([msg])
            accumulated_tokens += msg_tokens

            if accumulated_tokens >= self.keep_recent_tokens:
                cut_index = i
                logger.debug(f"Cut point at index {cut_index}, {accumulated_tokens} tokens")
                break

        if cut_index == 0:
            return CutPointResult(
                left_messages=messages,
                needs_compaction=True,
                token_count=token_count,
                threshold=threshold,
                accumulated_tokens=accumulated_tokens,
            )

        cut_message = messages[cut_index]
        is_user_cut = self._is_user_message(cut_message)

        if is_user_cut:
            return CutPointResult(
                messages_to_summarize=messages[:cut_index],
                left_messages=messages[cut_index:],
                cut_index=cut_index,
                needs_compaction=True,
                token_count=token_count,
                threshold=threshold,
                accumulated_tokens=accumulated_tokens,
            )

        turn_start_index = self._find_turn_start_index(messages, cut_index)

        if turn_start_index == -1:
            logger.warning("Split turn detected but no turn start found, treating as clean cut")
            return CutPointResult(
                messages_to_summarize=messages[:cut_index],
                left_messages=messages[cut_index:],
                cut_index=cut_index,
                needs_compaction=True,
                token_count=token_count,
                threshold=threshold,
                accumulated_tokens=accumulated_tokens,
            )

        return CutPointResult(
            messages_to_summarize=messages[:turn_start_index],
            turn_prefix_messages=messages[turn_start_index:cut_index],
            left_messages=messages[cut_index:],
            is_split_turn=True,
            cut_index=cut_index,
            needs_compaction=True,
            token_count=token_count,
            threshold=threshold,
            accumulated_tokens=accumulated_tokens,
        )

    async def execute(self):
        """
        Execute context check and find cut point.

        Returns:
            dict: CutPointResult.model_dump() with cut point information.
        """
        messages = self.context.messages
        normalized_messages = self._normalize_messages(messages)
        token_count: int = self.token_counter.count_token(normalized_messages)
        threshold = self.context_window_tokens - self.reserve_tokens

        needs_compaction = token_count >= threshold

        if not needs_compaction:
            logger.info(f"Token count {token_count} below threshold ({threshold}), no compaction needed")
            cut_result = CutPointResult(
                needs_compaction=False,
                token_count=token_count,
                threshold=threshold,
                left_messages=normalized_messages,
            )
            return cut_result.model_dump()

        logger.info(f"Compaction needed, token count: {token_count}, threshold: {threshold}")
        cut_result = self._find_cut_point(normalized_messages, token_count, threshold)
        return cut_result.model_dump()
