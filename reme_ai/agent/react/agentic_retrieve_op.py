"""Async React agent operator tailored for retrieval workflows."""

from typing import Dict, List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall, Message
from flowllm.gallery.agent import ReactAgentOp


@C.register_op()
class AgenticRetrieveOp(ReactAgentOp):
    """React agent that exposes RAG-friendly tools and context policies."""

    file_path: str = __file__

    def __init__(
        self,
        llm: str = "qwen3_30b_instruct",
        max_steps: int = 5,
        add_think_tool: bool = False,
        **kwargs,
    ):
        super().__init__(llm=llm, max_steps=max_steps, add_think_tool=add_think_tool, **kwargs)

    def build_tool_call(self) -> ToolCall:
        """Expose metadata describing how to invoke the agent."""
        return ToolCall(
            **{
                "description": "A React agent that answers user queries.",
                "input_schema": {
                    "messages": {
                        "type": "array",
                        "description": "messages",
                        "required": True,
                    },
                    "context_manage_mode": {
                        "type": "string",
                        "description": "Context management mode: 'compact' (only compacts tool messages), 'compress' "
                        "(only LLM-based compression), 'auto' (compaction first then compression if "
                        "needed). Defaults to 'auto'.",
                        "required": True,
                        "enum": ["compact", "compress", "auto"],
                    },
                    "max_total_tokens": {
                        "type": "integer",
                        "description": "Maximum token threshold for triggering compression/compaction. For compaction "
                        "this is total tokens; for compression this excludes keep_recent_count and "
                        "system messages. Defaults to 20000.",
                        "required": False,
                    },
                    "max_tool_message_tokens": {
                        "type": "integer",
                        "description": "Maximum token count per tool message before compaction applies. Exceeding "
                        "messages store full content externally with a preview in context. Defaults "
                        "to 2000.",
                        "required": False,
                    },
                    "group_token_threshold": {
                        "type": "integer",
                        "description": "Maximum tokens per compression group for LLM-based compression. None/0 "
                        "compresses all messages together. Oversized messages form their own group. "
                        "Used in 'compress' or 'auto' mode.",
                        "required": False,
                    },
                    "keep_recent_count": {
                        "type": "integer",
                        "description": "Number of recent messages preserved without compression/compaction. Defaults "
                        "to 1 for compaction and 2 for compression.",
                        "required": False,
                    },
                    "store_dir": {
                        "type": "string",
                        "description": "Directory for storing offloaded contents. Required for compaction/compression "
                        "to save full tool messages and compressed groups.",
                        "required": False,
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Chat session identifier for naming stored files. Defaults to auto-generated "
                        "UUID if omitted.",
                        "required": False,
                    },
                },
            },
        )

    async def build_tool_op_dict(self) -> dict:
        """Collect available tool operators from the execution context."""
        from reme_ai.context.file_tool import GrepOp, ReadFileOp

        grep_op = GrepOp(language=self.language)
        read_file_op = ReadFileOp(language=self.language)
        tool_op_dict: Dict[str, BaseAsyncToolOp] = {
            grep_op.tool_call.name: grep_op,
            read_file_op.tool_call.name: read_file_op,
        }

        return tool_op_dict

    async def build_messages(self) -> List[Message]:
        """Build the initial message history for the LLM."""
        return self.context.messages

    async def before_chat(self, messages: List[Message]):
        """Run context offload to trim prior messages before invoking the agent."""
        from reme_ai.context.offload import ContextOffloadOp

        op = ContextOffloadOp()
        await op.async_call(**self.input_dict)
        messages = op.context.response.answer
        messages = [Message(**x) for x in messages]
        return messages
