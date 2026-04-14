"""ReMe OpenAI chat message formatter."""

import json
from typing import Any

from agentscope.formatter import OpenAIChatFormatter
from agentscope.formatter._openai_formatter import (
    _format_openai_image_block,
    _to_openai_audio_data,
)
from agentscope.message import Msg, TextBlock, ImageBlock, URLSource


def _format_openai_video_block(video_block: dict) -> dict[str, Any]:
    """Format a video block for OpenAI API.

    Args:
        video_block: The video block containing a URL or base64 source.

    Returns:
        A dict with OpenAI-compatible video_url content.
    """
    source = video_block["source"]
    if source["type"] == "url":
        url = source["url"]
    elif source["type"] == "base64":
        url = f"data:{source['media_type']};base64,{source['data']}"
    else:
        raise ValueError(f"Unsupported video source type: {source['type']}")

    return {"type": "video_url", "video_url": {"url": url}}


class ReMeOpenAIChatFormatter(OpenAIChatFormatter):
    """Extends OpenAIChatFormatter with tool result image promotion and reasoning content support."""

    async def _format(
            self,
            msgs: list[Msg],
    ) -> list[dict[str, Any]]:
        """Format messages into OpenAI API format.

        Handles text, thinking (reasoning_content), tool_use, tool_result,
        image, audio, and video content blocks.

        Args:
            msgs: List of Msg objects.

        Returns:
            List of dicts with "role", "name", "content" and optional "tool_calls" or "reasoning_content".
        """
        self.assert_list_of_msgs(msgs)

        messages: list[dict] = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            content_blocks = []
            tool_calls = []
            reasoning_content_blocks = []

            for block in msg.get_content_blocks():
                typ = block.get("type")

                if typ == "text":
                    content_blocks.append({**block})

                elif typ == "thinking":
                    reasoning_content_blocks.append({**block})

                elif typ == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    })

                elif typ == "tool_result":
                    textual_output, multimodal_data = self.convert_tool_result_to_string(block["output"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("id"),
                        "content": textual_output,
                        "name": block.get("name"),
                    })

                    # Promote tool result images into a follow-up user message
                    promoted_blocks = []
                    for url, multimodal_block in multimodal_data:
                        if multimodal_block["type"] == "image" and self.promote_tool_result_images:
                            promoted_blocks.extend([
                                TextBlock(type="text", text=f"\n- The image from '{url}': "),
                                ImageBlock(type="image", source=URLSource(type="url", url=url)),
                            ])

                    if promoted_blocks:
                        promoted_blocks = [
                            TextBlock(
                                type="text",
                                text="<system-info>The following are the image contents from the tool "
                                     f"result of '{block['name']}':",
                            ),
                            *promoted_blocks,
                            TextBlock(type="text", text="</system-info>"),
                        ]
                        msgs.insert(
                            i + 1,
                            Msg(name="user", content=promoted_blocks, role="user"),
                        )

                elif typ == "image":
                    content_blocks.append(_format_openai_image_block(block))

                elif typ == "audio":
                    # Skip assistant audio output
                    if msg.role == "assistant":
                        continue
                    content_blocks.append({
                        "type": "input_audio",
                        "input_audio": _to_openai_audio_data(block["source"]),
                    })

                elif typ == "video":
                    # Skip assistant video output
                    if msg.role == "assistant":
                        continue
                    content_blocks.append(_format_openai_video_block(block))

                else:
                    ...
                    # logger.warning("Unsupported block type %s, skipped.", typ)

            msg_openai = {
                "role": msg.role,
                "name": msg.name,
                "content": content_blocks or None,
            }

            if tool_calls:
                msg_openai["tool_calls"] = tool_calls

            if reasoning_content_blocks:
                reasoning_msg = "\n".join(
                    r.get("thinking", "") for r in reasoning_content_blocks
                )
                if reasoning_msg:
                    msg_openai["reasoning_content"] = reasoning_msg

            if msg_openai["content"] or msg_openai.get("tool_calls"):
                messages.append(msg_openai)

            i += 1

        return messages
