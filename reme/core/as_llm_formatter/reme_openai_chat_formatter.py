"""ReMeOpenAIChatFormatter"""

import json
from typing import Any

from agentscope.formatter import OpenAIChatFormatter
from agentscope.formatter._openai_formatter import (
    _format_openai_image_block,
    _to_openai_audio_data,
)
from agentscope.message import Msg, TextBlock, ImageBlock, URLSource
from loguru import logger


def _format_openai_video_block(video_block: dict) -> dict[str, Any]:
    """Format a video block for OpenAI API.

    Args:
        video_block: The video block to format.

    Returns:
        A dictionary with video content in OpenAI format.
    """
    source = video_block["source"]
    if source["type"] == "url":
        url = source["url"]
    elif source["type"] == "base64":
        data = source["data"]
        media_type = source["media_type"]
        url = f"data:{media_type};base64,{data}"
    else:
        raise ValueError(f"Unsupported video source type: {source['type']}")

    return {
        "type": "video_url",
        "video_url": {
            "url": url,
        },
    }


class ReMeOpenAIChatFormatter(OpenAIChatFormatter):
    """ReMeOpenAIChatFormatter"""

    async def _format(
        self,
        msgs: list[Msg],
    ) -> list[dict[str, Any]]:
        """Format message objects into OpenAI API required format.

        Args:
            msgs (`list[Msg]`):
                The list of Msg objects to format.

        Returns:
            `list[dict[str, Any]]`:
                A list of dictionaries, where each dictionary has "name",
                "role", and "content" keys.
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
                    # Collect thinking blocks for reasoning_content field
                    # This is compatible with models like DeepSeek that support
                    # extended thinking via reasoning_content field
                    reasoning_content_blocks.append({**block})

                elif typ == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(
                                    block.get("input", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        },
                    )

                elif typ == "tool_result":
                    (
                        textual_output,
                        multimodal_data,
                    ) = self.convert_tool_result_to_string(block["output"])

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("id"),
                            "content": (textual_output),  # type: ignore[arg-type]
                            "name": block.get("name"),
                        },
                    )

                    # Then, handle the multimodal data if any
                    promoted_blocks: list = []
                    for url, multimodal_block in multimodal_data:
                        if multimodal_block["type"] == "image" and self.promote_tool_result_images:
                            promoted_blocks.extend(
                                [
                                    TextBlock(
                                        type="text",
                                        text=f"\n- The image from '{url}': ",
                                    ),
                                    ImageBlock(
                                        type="image",
                                        source=URLSource(
                                            type="url",
                                            url=url,
                                        ),
                                    ),
                                ],
                            )

                    if promoted_blocks:
                        # Insert promoted blocks as new user message(s)
                        promoted_blocks = [
                            TextBlock(
                                type="text",
                                text="<system-info>The following are "
                                "the image contents from the tool "
                                f"result of '{block['name']}':",
                            ),
                            *promoted_blocks,
                            TextBlock(
                                type="text",
                                text="</system-info>",
                            ),
                        ]

                        msgs.insert(
                            i + 1,
                            Msg(
                                name="user",
                                content=promoted_blocks,
                                role="user",
                            ),
                        )

                elif typ == "image":
                    content_blocks.append(
                        _format_openai_image_block(
                            block,  # type: ignore[arg-type]
                        ),
                    )

                elif typ == "audio":
                    # Filter out audio content when the multimodal model
                    # outputs both text and audio, to prevent errors in
                    # subsequent model calls
                    if msg.role == "assistant":
                        continue
                    input_audio = _to_openai_audio_data(block["source"])
                    content_blocks.append(
                        {
                            "type": "input_audio",
                            "input_audio": input_audio,
                        },
                    )

                elif typ == "video":
                    # Filter out video content when the multimodal model
                    # outputs both text and video, to prevent errors in
                    # subsequent model calls
                    if msg.role == "assistant":
                        continue
                    content_blocks.append(
                        _format_openai_video_block(block),
                    )

                else:
                    logger.warning(
                        "Unsupported block type %s in the message, skipped.",
                        typ,
                    )

            msg_openai = {
                "role": msg.role,
                "name": msg.name,
                "content": content_blocks or None,
            }

            if tool_calls:
                msg_openai["tool_calls"] = tool_calls

            # Add reasoning_content for thinking blocks (compatible with DeepSeek, etc.)
            if reasoning_content_blocks:
                reasoning_msg = "\n".join(reasoning.get("thinking", "") for reasoning in reasoning_content_blocks)
                if reasoning_msg:
                    msg_openai["reasoning_content"] = reasoning_msg

            # When both content and tool_calls are None, skipped
            if msg_openai["content"] or msg_openai.get("tool_calls"):
                messages.append(msg_openai)

            # Move to next message
            i += 1

        return messages
