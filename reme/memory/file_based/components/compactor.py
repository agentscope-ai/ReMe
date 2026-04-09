"""Compactor module for memory compaction operations."""

from agentscope.agent import ReActAgent
from agentscope.message import Msg

from ..utils import AsMsgHandler
from ....core.op import BaseOp
from ....core.utils import get_logger

logger = get_logger()


def _is_valid_summary(content: str) -> bool:
    """Check if the summary content is valid.

    Args:
        content: The summary content to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not content or not content.strip():
        return False
    if "##" not in content:
        return False
    return True


class Compactor(BaseOp):
    """Compactor class for compacting memory messages."""

    def __init__(
        self,
        memory_compact_threshold: int,
        console_enabled: bool = False,
        return_dict: bool = False,
        add_thinking_block: bool = True,
        extra_instruction: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory_compact_threshold: int = memory_compact_threshold
        self.console_enabled: bool = console_enabled
        self.return_dict: bool = return_dict
        self.add_thinking_block: bool = add_thinking_block
        self.extra_instruction: str = extra_instruction

    # pylint: disable=too-many-return-statements
    async def execute(self):
        messages: list[Msg] = self.context.get("messages", [])
        previous_summary: str = self.context.get("previous_summary", "")

        if not messages:
            if self.return_dict:
                return {"user_message": "", "history_compact": "", "is_valid": False}
            return ""

        msg_handler = AsMsgHandler(self.as_token_counter)
        before_token_count = await msg_handler.count_msgs_token(messages)
        history_formatted_str: str = await msg_handler.format_msgs_to_str(
            messages=messages,
            memory_compact_threshold=self.memory_compact_threshold,
            include_thinking=self.add_thinking_block,
        )
        after_token_count = await msg_handler.count_str_token(history_formatted_str)
        logger.info(f"Compactor before_token_count={before_token_count} after_token_count={after_token_count}")

        if not history_formatted_str:
            logger.warning(f"No history to compact. messages={messages}")
            if self.return_dict:
                return {"user_message": "", "history_compact": "", "is_valid": False}
            return ""

        agent = ReActAgent(
            name="reme_compactor",
            model=self.as_llm,
            sys_prompt=self.get_prompt("system_prompt"),
            formatter=self.as_llm_formatter,
        )
        agent.set_console_output_enabled(self.console_enabled)

        if previous_summary:
            user_message: str = (
                f"# conversation\n{history_formatted_str}\n\n"
                f"# previous-summary\n{previous_summary}\n\n" + self.get_prompt("update_user_message")
            )
        else:
            user_message: str = f"# conversation\n{history_formatted_str}\n\n" + self.get_prompt("initial_user_message")

        if self.extra_instruction:
            user_message += f"\n\n# extra-instruction\n{self.extra_instruction}"
        logger.info(f"Compactor sys_prompt={agent.sys_prompt} user_message={user_message}")

        compact_msg: Msg = await agent.reply(
            Msg(
                name="reme",
                role="user",
                content=user_message,
            ),
        )

        history_compact: str = compact_msg.get_text_content()
        is_valid: bool = _is_valid_summary(history_compact)

        if not is_valid:
            logger.warning(f"Invalid summary result: {history_compact[:200]}...")
            if self.return_dict:
                return {"user_message": user_message, "history_compact": history_compact, "is_valid": False}
            return ""

        logger.info(f"Compactor Result:\n{history_compact}")

        if self.return_dict:
            return {"user_message": user_message, "history_compact": history_compact, "is_valid": True}
        return history_compact
