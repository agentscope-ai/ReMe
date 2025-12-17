import datetime
from typing import List

from .base_memory_agent_op import BaseMemoryAgentOp
from .. import C
from ..enumeration import Role, MemoryType
from ..schema import Message


@C.register_op()
class SimpleSummaryAgentOp(BaseMemoryAgentOp):
    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages(self) -> List[Message]:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary_context: str
        if "query" in self.input_dict:
            summary_context = self.input_dict["query"]

        elif "messages" in self.input_dict:
            input_messages = self.input_dict["messages"]
            input_messages = [Message(**x) for x in input_messages if isinstance(x, dict)]
            summary_context = "\n".join([x.format_message(
                add_time_created=True,
                use_name_first=True,
                add_reasoning_content=True,
                add_tool_calls=True,
            ) for x in input_messages])

        else:
            raise ValueError("input_dict must contain either 'query' or 'messages'")

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.get_prompt("system_prompt").format(
                    now_time=now_time,
                    summary_context=summary_context,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]
        return messages
