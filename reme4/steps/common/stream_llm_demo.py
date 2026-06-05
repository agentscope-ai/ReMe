"""Demo step that drives an Agent via the agent_wrapper component with streaming output."""

import json

from agentscope.event import (
    TextBlockDeltaEvent,
    ThinkingBlockDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolResultTextDeltaEvent,
    ModelCallEndEvent,
    ReplyStartEvent,
)
from agentscope.message import Msg
from agentscope.tool import FunctionTool, Toolkit

from ..base_step import BaseStep
from ...components import R
from ...enumeration import ChunkEnum


def add(a: float, b: float) -> str:
    """Add two numbers and return the sum.

    Args:
        a: first addend
        b: second addend
    """
    return str(a + b)


@R.register("stream_llm_demo_step")
class StreamLLMDemoStep(BaseStep):
    """Drive an Agent powered by the ``agent_wrapper`` component with streaming output.

    When streaming is enabled on the context, text/thinking/tool events are
    pushed chunk-by-chunk via ``self.context.add_stream_string``.
    When streaming is not enabled, falls back to non-streaming reply.

    Inputs (from RuntimeContext):
        query      (str, required): user message content.
        sys_prompt (str, optional): system prompt for the agent.
        use_add_tool (bool, optional): register the ``add`` tool when True.

    Output (written to context.response.answer):
        The agent's final reply text.
    """

    DEFAULT_SYS_PROMPT = "You are a helpful assistant. Provide clear and detailed responses."

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        sys_prompt: str = self.context.get("sys_prompt") or self.DEFAULT_SYS_PROMPT
        use_add_tool: bool = bool(self.context.get("use_add_tool", False))

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        toolkit = Toolkit(tools=[FunctionTool(add)]) if use_add_tool else Toolkit()

        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "toolkit": toolkit,
        }

        if self.context.stream:
            text = await self._stream_reply(query, **wrapper_kwargs)
        else:
            _, msg = await self.agent_wrapper.reply(query, **wrapper_kwargs)
            text = (msg.get_text_content() or "").strip()

        self.logger.debug(f"[{self.name}] response: {text!r}")

        self.context.response.success = True
        self.context.response.answer = text
        self.context.response.metadata.update(
            {
                "query": query,
                "sys_prompt": sys_prompt,
                "use_add_tool": use_add_tool,
                "response": text,
            },
        )
        return self.context.response

    async def _stream_reply(self, query: str, **wrapper_kwargs) -> str:
        """Stream agent reply events to the context stream queue."""
        assert self.context is not None
        reply_msg: Msg | None = None

        async for event in self.agent_wrapper.reply_stream(query, **wrapper_kwargs):
            if isinstance(event, ReplyStartEvent):
                reply_msg = Msg(
                    id=event.reply_id,
                    name=event.name,
                    role=event.role,
                    content=[],
                )
            elif isinstance(event, TextBlockDeltaEvent):
                await self.context.add_stream_string(event.delta, ChunkEnum.CONTENT)
            elif isinstance(event, ThinkingBlockDeltaEvent):
                await self.context.add_stream_string(event.delta, ChunkEnum.THINK)
            elif isinstance(event, ToolCallStartEvent):
                payload = json.dumps({"name": event.tool_call_name, "id": event.tool_call_id})
                await self.context.add_stream_string(payload, ChunkEnum.TOOL_CALL)
            elif isinstance(event, ToolCallDeltaEvent):
                await self.context.add_stream_string(event.delta, ChunkEnum.TOOL_CALL)
            elif isinstance(event, ToolResultTextDeltaEvent):
                await self.context.add_stream_string(event.delta, ChunkEnum.TOOL_RESULT)
            elif isinstance(event, ModelCallEndEvent):
                usage = json.dumps(
                    {"input_tokens": event.input_tokens, "output_tokens": event.output_tokens},
                )
                await self.context.add_stream_string(usage, ChunkEnum.USAGE)

            if reply_msg is not None:
                reply_msg.append_event(event)

        if reply_msg is not None:
            return (reply_msg.get_text_content() or "").strip()
        return ""
