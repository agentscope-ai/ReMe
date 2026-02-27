"""Data models for multi-modal conversation history and LLM interaction trajectories."""

import datetime
import json
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tool_call import ToolCall
from ..enumeration import Role


class ContentBlock(BaseModel):
    """
    Individual unit of multi-modal content like text, images, or video.
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
    def init_block(cls, data: dict) -> dict:
        """Dynamically maps the type-specific key to the content field."""
        content_type = data.get("type", "")
        if content_type and content_type in data:
            data["content"] = data[content_type]
        return data

    def simple_dump(self) -> dict:
        """Serializes the block into an API-compatible dictionary format."""
        return {
            "type": self.type,
            self.type: self.content,
            **self.model_extra,
        }


class Message(BaseModel):
    """Data model for a single dialogue entry including roles and tool interactions."""

    name: str | None = Field(default=None)
    role: Role = Field(default=Role.USER)
    content: str | list[ContentBlock] = Field(default="")
    reasoning_content: str = Field(default="")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str = Field(default="")
    time_created: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: dict = Field(default_factory=dict)

    def dump_content(self) -> str | list[dict]:
        """Returns content as a raw string or a list of serialized blocks."""
        if isinstance(self.content, str):
            return self.content
        return [block.simple_dump() for block in self.content]

    def get_text_content(self) -> str:
        """Extract plain text content from message, handling both str and list[ContentBlock]."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            block.content if isinstance(block.content, str) else str(block.content) for block in self.content
        )

    def simple_dump(
        self,
        add_name: bool = False,
        add_reasoning: bool = True,
        add_time_created: bool = False,
        add_metadata: bool = False,
        enable_argument_dict: bool = False,
        as_dict: bool = True,
    ) -> dict | str:
        """Transforms the message into a simplified dictionary for standard APIs."""
        result = {}
        if add_name and self.name:
            result["name"] = self.name

        result["role"] = self.role.value
        result["content"] = self.dump_content()

        if add_reasoning and self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content

        if self.tool_calls:
            result["tool_calls"] = [
                tc.simple_output_dump(
                    as_dict=True,
                    enable_argument_dict=enable_argument_dict,
                )
                for tc in self.tool_calls
            ]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if add_time_created:
            result["time_created"] = self.time_created

        if add_metadata:
            result["metadata"] = self.metadata

        return result if as_dict else json.dumps(result, ensure_ascii=False)

    def format_message(
        self,
        index: int | None = None,
        add_time: bool = False,
        use_name: bool = False,
        add_reasoning: bool = True,
        add_tools: bool = True,
        strip_markdown_headers: bool = False,
    ) -> str:
        """Generates a human-readable string representation of the message."""
        prefix = f"round{index} " if index is not None else ""
        time_str = f"[{self.time_created}] " if add_time else ""
        header = f"{self.name or self.role.value if use_name else self.role.value}:"

        lines = [f"{prefix}{time_str}{header}"]

        def strip_md_func(line):
            if strip_markdown_headers:
                line = re.sub(r"\n##+ +", "\n", line)
            return line

        if add_reasoning and self.reasoning_content:
            lines.append(self.reasoning_content)

        if isinstance(self.content, str):
            lines.append(strip_md_func(self.content))

        elif isinstance(self.content, list):
            for block in self.content:
                text = (
                    block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False)
                )
                text = str(text)
                lines.append(strip_md_func(text))

        if add_tools and self.tool_calls:
            for tc in self.tool_calls:
                lines.append(f" - tool_call={tc.name} params={tc.arguments}")

        return " ".join(lines).strip()


class Trajectory(BaseModel):
    """Sequence of messages representing a full conversation session and its evaluation."""

    task_id: str = Field(default="")
    messages: list[Message] = Field(default_factory=list)
    score: float = Field(default=0.0)
    metadata: dict = Field(default_factory=dict)
