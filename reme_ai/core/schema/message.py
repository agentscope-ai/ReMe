import datetime
import json
from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tool_call import ToolCall
from ..enumeration import Role


class ContentBlock(BaseModel):
    """
    examples:
    {
        "type": "image_url",
        "image_url": {
            "url": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"
        },
    }
    {
        "type": "video",
        "video": [
            "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/xzsgiz/football1.jpg",
            "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/tdescd/football2.jpg",
            "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/zefdja/football3.jpg",
            "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/aedbqh/football4.jpg",
        ],
    }
    {
        "type": "text",
        "text": "How do you solve this problem?"
    }
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="")
    content: str | dict | list = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def init_block(cls, data: dict):
        """Initialize content block by extracting content based on type field."""
        result = data.copy()
        content_type = data.get("type", "")
        if content_type and content_type in data:
            result["content"] = data[content_type]
        return result

    def simple_dump(self) -> dict:
        """Convert ContentBlock to a simple dictionary format."""
        result = {
            "type": self.type,
            self.type: self.content,
            **self.model_extra,
        }

        return result


class Message(BaseModel):
    """Represents a message in a conversation with role, content, and optional tool calls."""

    name: str | None = Field(default=None)
    role: Role = Field(default=Role.USER)
    content: str | List[ContentBlock] = Field(default="")
    reasoning_content: str = Field(default="")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_call_id: str = Field(default="")
    time_created: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: dict = Field(default_factory=dict)

    def dump_content(self) -> str | list[dict]:
        """Dump message content to string or list of dicts."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return [x.simple_dump() for x in self.content]
        else:
            raise ValueError(f"Invalid content type: {type(self.content)}")

    def simple_dump(self, add_reasoning: bool = True) -> dict:
        """Convert Message to a simple dictionary format for API serialization."""
        result: dict = {"role": self.role.value, "content": self.dump_content()}

        if add_reasoning:
            result["reasoning_content"] = self.reasoning_content

        if self.tool_calls:
            result["tool_calls"] = [x.simple_output_dump() for x in self.tool_calls]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        return result

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
            if isinstance(self.content, str):
                content += self.content + "\n"
            else:
                for block in self.content:
                    if isinstance(block.content, str):
                        content += block.content
                    else:
                        content += json.dumps(block.content, ensure_ascii=False)
                    content += "\n"

        if add_tool_calls and self.tool_calls:
            for tool_call in self.tool_calls:
                content += f" - tool_call={tool_call.name} params={tool_call.arguments}\n"

        return content.strip()


class Trajectory(BaseModel):
    """Represents a conversation trajectory with messages and optional scoring."""

    task_id: str = Field(default="")
    messages: List[Message] = Field(default_factory=list)
    score: float = Field(default=0.0)
    metadata: dict = Field(default_factory=dict)
