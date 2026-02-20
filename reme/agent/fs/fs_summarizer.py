"""Personal memory retriever agent for retrieving personal memories through vector search."""

import datetime

from loguru import logger

from ...core.enumeration import Role
from ...core.op import BaseReact
from ...core.schema import Message
from ...core.utils import format_messages


class FsSummarizer(BaseReact):
    """Retrieve personal memories through vector search and history reading."""

    def __init__(self, working_dir: str, memory_dir: str = "memory", version: str = "v1", **kwargs):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        self.memory_dir: str = memory_dir
        self.version: str = version

    async def build_messages(self) -> list[Message]:
        messages: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        date_str: str = self.context.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))

        if self.version == "default":
            messages.append(
                Message(
                    role=Role.USER,
                    content=self.prompt_format(
                        "user_message_default",
                        working_dir=self.working_dir,
                        date=date_str,
                        memory_dir=self.memory_dir,
                    ),
                ),
            )

        elif self.version == "v1":
            conversation = format_messages(messages, add_index=False)
            messages = [
                Message(
                    role=Role.USER,
                    content=f"<conversation>\n{conversation}\n</conversation>\n"
                    + self.prompt_format(
                        "user_message_default",
                        conversation=format_messages(messages, add_index=False),
                        working_dir=self.working_dir,
                        date=date_str,
                        memory_dir=self.memory_dir,
                    ),
                ),
            ]

        else:
            messages.extend(
                [
                    Message(role=Role.SYSTEM, content=self.get_prompt("system_prompt")),
                    Message(
                        role=Role.USER,
                        content=self.prompt_format(
                            "user_message",
                            date=date_str,
                            memory_dir=self.memory_dir,
                        ),
                    ),
                ],
            )
        return messages

    async def execute(self):
        result = await super().execute()
        answer = result["answer"]
        logger.info(f"[{self.__class__.__name__}] answer={answer}")
        return result
