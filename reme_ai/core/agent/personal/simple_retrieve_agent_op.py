import datetime
from typing import List

from ..base_memory_agent_op import BaseMemoryAgentOp
from ... import C
from ...enumeration import Role, MemoryType
from ...schema import Message


@C.register_op()
class SimpleRetrieveAgentOp(BaseMemoryAgentOp):
    """Agent for retrieving memories from the vector store.

    This agent analyzes conversation context to determine if retrieval is needed,
    then uses vector_retrieve_memory and read_history tools to find relevant memories.
    """

    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages(self) -> List[Message]:
        """Build the initial messages for the retrieve agent.

        Returns:
            List[Message]: List containing the system prompt message.
        """
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        retrieve_context: str
        if "query" in self.input_dict:
            retrieve_context = self.input_dict["query"]

        elif "messages" in self.input_dict:
            input_messages = self.input_dict["messages"]
            input_messages = [Message(**x) for x in input_messages if isinstance(x, dict)]
            input_messages = [x for x in input_messages if x.role is not Role.SYSTEM]
            retrieve_context = "\n".join([x.format_message(
                add_time_created=True,
                use_name_first=True,
                add_reasoning_content=True,
                add_tool_calls=True,
            ) for x in input_messages])

        else:
            raise ValueError("input_dict must contain either 'query' or 'messages'")

        # Build system prompt with context
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            retrieve_context=retrieve_context,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
        )

        # Build user message
        user_message = self.get_prompt("user_message")

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]
        return messages
