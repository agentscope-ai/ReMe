"""Compactor module for memory compaction operations."""

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.token import HuggingFaceTokenCounter
from loguru import logger

from ..utils import AsMsgHandler
from ....core.op import BaseOp


class Compactor(BaseOp):
    """Compactor class for compacting memory messages."""

    def __init__(
        self,
        memory_compact_threshold: int,
        token_counter: HuggingFaceTokenCounter,
        console_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory_compact_threshold: int = memory_compact_threshold

        self.msg_handler = AsMsgHandler(token_counter=token_counter)
        self.console_enabled: bool = console_enabled

    async def execute(self):
        messages: list[Msg] = self.context.get("messages", [])
        previous_summary: str = self.context.get("previous_summary", "")

        if not messages:
            return ""

        before_token_count = self.msg_handler.count_msgs_token(messages)
        history_formatted_str: str = self.msg_handler.format_msgs_to_str(
            messages=messages,
            memory_compact_threshold=self.memory_compact_threshold,
        )
        after_token_count = self.msg_handler.count_str_token(history_formatted_str)
        logger.info(f"Compactor before_token_count={before_token_count} after_token_count={after_token_count}")

        if not history_formatted_str:
            logger.warning(f"No history to compact. messages={messages}")
            return ""

        agent = ReActAgent(
            name="reme_compactor",
            model=self.as_llm,
            sys_prompt=self.get_prompt("system_prompt"),
            formatter=self.as_llm_formatter,
        )
        agent.set_console_output_enabled(self.console_enabled)

        if previous_summary:
            prefix: str = self.get_prompt("update_user_message_prefix")
            suffix: str = self.get_prompt("update_user_message_suffix")
            user_message: str = (
                f"<conversation>\n{history_formatted_str}\n</conversation>\n\n"
                f"{prefix}\n\n"
                f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
                f"{suffix}"
            )
        else:
            user_message: str = f"<conversation>\n{history_formatted_str}\n</conversation>\n\n" + self.get_prompt(
                "initial_user_message",
            )
        logger.info(f"Compactor sys_prompt={agent.sys_prompt} user_message={user_message}")

        compact_msg: Msg = await agent.reply(
            Msg(
                name="reme",
                role="user",
                content=user_message,
            ),
        )

        history_compact: str = compact_msg.get_text_content()
        logger.info(f"Compactor Result:\n{history_compact}")
        return history_compact
