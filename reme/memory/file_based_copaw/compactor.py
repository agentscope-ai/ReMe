"""Compactor module for memory compaction operations."""

import logging

from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.token import HuggingFaceTokenCounter

from .memory_formatter import MemoryFormatter
from ...core.op import BaseOp

logger = logging.getLogger(__name__)


class Compactor(BaseOp):
    """Compactor class for compacting memory messages."""

    def __init__(
        self,
        memory_compact_threshold: int,
        chat_model: ChatModelBase,
        formatter: FormatterBase,
        token_counter: HuggingFaceTokenCounter,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory_compact_threshold: int = memory_compact_threshold

        self.chat_model: ChatModelBase = chat_model
        self.formatter: FormatterBase = formatter
        self.as_token_counter: HuggingFaceTokenCounter = token_counter

    async def execute(self):
        messages: list[Msg] = self.context.get("messages", [])
        previous_summary: str = self.context.get("previous_summary", "")

        if not messages:
            return ""

        formatter = MemoryFormatter(
            token_counter=self.as_token_counter,
            memory_compact_threshold=self.memory_compact_threshold,
        )
        history_formatted_str: str = formatter.format(messages)

        if not history_formatted_str:
            logger.warning(f"No history to compact. messages={messages}")
            return ""

        agent = ReActAgent(
            name="reme_compactor",
            model=self.chat_model,
            sys_prompt=self.get_prompt("system_prompt"),
            formatter=self.formatter,
        )

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
