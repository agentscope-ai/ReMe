from typing import List

from flowllm.core.schema import Message as FlowMessage, ToolCall as FlowToolCall
from pydantic import BaseModel, Field


class Message(FlowMessage):

    def format_message(
            self,
            i: int | None = None,
            add_time_created: bool = False,
            use_name_first: bool = False,
            add_reasoning_content: bool = True,
            add_tool_calls: bool = True
    ) -> str:
        content = ""
        if i is not None:
            content += f"round{i} "

        if add_time_created:
            content += f"[{self.time_created}] "

        if use_name_first:
            content += f"{self.name or self.role.value}:\n"
        else:
            content += f"{self.role.value}:\n"

        if add_reasoning_content and self.reasoning_content:
            content += self.reasoning_content + "\n"

        if self.content:
            content += self.content + "\n"

        if add_tool_calls and self.tool_calls:
            for tool_call in self.tool_calls:
                content += f" - tool_call={tool_call.name} params={tool_call.arguments}\n"

        return content.strip()


class ToolCall(FlowToolCall):
    pass

class Trajectory(BaseModel):

    task_id: str = Field(default="")
    messages: List[Message] = Field(default_factory=list)
    score: float = Field(default=0.0)
    metadata: dict = Field(default_factory=dict)

from .memory_node import MemoryNode