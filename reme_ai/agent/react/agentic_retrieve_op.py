"""Async React agent operator tailored for retrieval workflows."""

from typing import Dict, List

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall, Message
from loguru import logger


@C.register_op()
class AgenticRetrieveOp(BaseAsyncToolOp):
    """React agent that exposes RAG-friendly tools and context policies."""

    file_path: str = __file__

    def __init__(
        self,
        llm: str = "qwen3_30b_instruct",
        max_steps: int = 5,
        **kwargs,
    ):
        """Initialize the agent runtime configuration."""
        super().__init__(llm=llm, **kwargs)
        self.max_steps: int = max_steps

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

    async def async_execute(self):
        """Main execution loop that alternates LLM calls and tool invocations."""
        from reme_ai.context.file_tool import GrepOp, ReadFileOp
        from reme_ai.context.offload import ContextOffloadOp
        from reme_ai.context.file_tool import BatchWriteFileOp

        grep_op = GrepOp(language=self.language)
        read_file_op = ReadFileOp(language=self.language)
        tool_op_dict: Dict[str, BaseAsyncToolOp] = {
            grep_op.tool_call.name: grep_op,
            read_file_op.tool_call.name: read_file_op,
        }

        messages = [Message(**x) for x in self.context.messages]
        context_kwargs = self.input_dict.copy()
        context_kwargs.pop("messages", None)

        for i in range(self.max_steps):
            op = ContextOffloadOp() >> BatchWriteFileOp()
            await op.async_call(messages=[x.simple_dump() for x in messages], **context_kwargs)
            messages = [Message(**x) for x in op.context.response.answer]

            assistant_message: Message = await self.llm.achat(
                messages=messages,
                tools=[op.tool_call for op in tool_op_dict.values()],
            )
            messages.append(assistant_message)
            logger.info(f"round{i + 1}.assistant={assistant_message.model_dump_json()}")

            if not assistant_message.tool_calls:
                break

            op_list: List[BaseAsyncToolOp] = []
            for j, tool_call in enumerate(assistant_message.tool_calls):
                if tool_call.name not in tool_op_dict:
                    logger.exception(f"unknown tool_call.name={tool_call.name}")
                    continue

                logger.info(f"round{i + 1}.{j} submit tool_calls={tool_call.name} argument={tool_call.argument_dict}")

                op_copy: BaseAsyncToolOp = tool_op_dict[tool_call.name].copy()
                op_copy.tool_call.id = tool_call.id
                op_list.append(op_copy)
                self.submit_async_task(op.async_call, **tool_call.argument_dict)

            await self.join_async_task()

            for j, op in enumerate(op_list):
                tool_result = str(op.output)
                tool_message = Message(role=Role.TOOL, content=tool_result, tool_call_id=op.tool_call.id)
                messages.append(tool_message)
                logger.info(f"round{i + 1}.{j} join tool_result={tool_result[:200]}...\n\n")

        self.set_output(messages[-1].content)
        self.context.response.messages = messages
