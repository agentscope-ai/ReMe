"""Personal memory retriever agent for retrieving personal memories through vector search."""

import datetime

from loguru import logger

from ...core.enumeration import Role
from ...core.op import BaseReact
from ...core.schema import Message
from ...core.utils import format_messages


class FbSummarizer(BaseReact):
    """Retrieve personal memories through vector search and history reading."""

    def __init__(
        self,
        working_dir: str,
        memory_dir: str = "memory",
        version: str = "default",
        return_prompt: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        self.memory_dir: str = memory_dir
        self.version: str = version
        self.return_prompt = return_prompt

    async def build_messages(self) -> list[Message]:
        messages: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        date_str: str = self.context.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))

        if self.version == "default":
            conversation = format_messages(messages, add_index=False)
            messages = [
                Message(
                    role=Role.USER,
                    content=f"<conversation>\n{conversation}\n</conversation>\n"
                    + self.prompt_format(
                        "user_message_default",
                        conversation=conversation,
                        working_dir=self.working_dir,
                        date=date_str,
                        memory_dir=self.memory_dir,
                    ),
                ),
            ]

        elif self.version == "v1":
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

        else:
            messages.extend(
                [
                    Message(role=Role.SYSTEM, content=self.get_prompt("system_prompt_deprecated")),
                    Message(
                        role=Role.USER,
                        content=self.prompt_format(
                            "user_message_deprecated",
                            date=date_str,
                            memory_dir=self.memory_dir,
                        ),
                    ),
                ],
            )
        return messages

    async def execute(self):
        if self.return_prompt:
            result = {}
            messages: list[Message] = await self.build_messages()
            result["prompt"] = messages[-1].content
            return result
        else:
            result = await super().execute()
            answer = str(result["answer"])
            logger.info(f"[{self.__class__.__name__}] answer={answer}")
        return result
