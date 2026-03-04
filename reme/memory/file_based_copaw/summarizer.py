"""Summarizer module for memory summarization operations."""

import datetime
import logging

from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.token import HuggingFaceTokenCounter
from agentscope.tool import Toolkit

from .memory_formatter import MemoryFormatter
from .file_io import FileIO
from ...core.op import BaseOp

logger = logging.getLogger(__name__)


class Summarizer(BaseOp):
    """Summarizer class for summarizing memory messages."""

    def __init__(
        self,
        working_dir: str,
        memory_dir: str,
        memory_compact_threshold: int,
        chat_model: ChatModelBase,
        formatter: FormatterBase,
        token_counter: HuggingFaceTokenCounter,
        toolkit: Toolkit | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        self.memory_dir: str = memory_dir
        self.memory_compact_threshold: int = memory_compact_threshold

        self.chat_model: ChatModelBase = chat_model
        self.formatter: FormatterBase = formatter
        self.as_token_counter: HuggingFaceTokenCounter = token_counter
        if toolkit is not None:
            self.toolkit: Toolkit = toolkit
        else:
            self.toolkit = Toolkit()
            file_io = FileIO(working_dir=self.working_dir)
            self.toolkit.register_tool_function(file_io.read)
            self.toolkit.register_tool_function(file_io.write)
            self.toolkit.register_tool_function(file_io.edit)

    async def execute(self):
        messages: list[Msg] = self.context.get("messages", [])

        if not messages:
            return ""

        formatter = MemoryFormatter(
            token_counter=self.as_token_counter,
            memory_compact_threshold=self.memory_compact_threshold,
        )
        history_formatted_str: str = formatter.format(messages)

        if not history_formatted_str:
            logger.warning(f"No history to summarize. messages={messages}")
            return ""

        agent = ReActAgent(
            name="reme_summarizer",
            model=self.chat_model,
            sys_prompt="You are a helpful assistant.",
            formatter=self.formatter,
            toolkit=self.toolkit,
        )

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
