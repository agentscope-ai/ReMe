"""Summarizer module for memory summarization operations."""

import datetime

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.token import HuggingFaceTokenCounter
from agentscope.tool import Toolkit
from loguru import logger

from ..utils import AsMsgHandler
from ....core.op import BaseOp


class Summarizer(BaseOp):
    """Summarizer class for summarizing memory messages."""

    def __init__(
        self,
        working_dir: str,
        memory_dir: str,
        memory_compact_threshold: int,
        token_counter: HuggingFaceTokenCounter,
        toolkit: Toolkit,
        console_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        self.memory_dir: str = memory_dir
        self.memory_compact_threshold: int = memory_compact_threshold

        self.msg_handler = AsMsgHandler(token_counter=token_counter)
        self.toolkit: Toolkit = toolkit
        self.console_enabled: bool = console_enabled

    async def execute(self):
        messages: list[Msg] = self.context.get("messages", [])

        if not messages:
            return ""

        before_token_count = self.msg_handler.count_msgs_token(messages)
        history_formatted_str: str = self.msg_handler.format_msgs_to_str(
            messages=messages,
            memory_compact_threshold=self.memory_compact_threshold,
        )
        after_token_count = self.msg_handler.count_str_token(history_formatted_str)
        logger.info(f"Summarizer before_token_count={before_token_count} after_token_count={after_token_count}")

        if not history_formatted_str:
            logger.warning(f"No history to summarize. messages={messages}")
            return ""

        agent = ReActAgent(
            name="reme_summarizer",
            model=self.as_llm,
            sys_prompt="You are a helpful assistant.",
            formatter=self.as_llm_formatter,
            toolkit=self.toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_message: str = f"<conversation>\n{history_formatted_str}\n</conversation>\n" + self.prompt_format(
            "user_message",
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
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

        history_summary: str = summary_msg.get_text_content()
        logger.info(f"Summarizer Result:\n{history_summary}")
        return history_summary
