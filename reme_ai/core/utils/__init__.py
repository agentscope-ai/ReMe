from typing import List

from ..schema import Message


def format_messages(messages: List[Message | dict], **kwargs) -> str:
    result_list = []
    for i, message in enumerate(messages):
        if isinstance(message, dict):
            message = Message(**message)

        result_list.append(message.format_message(i, **kwargs))

    return "\n".join(result_list)
