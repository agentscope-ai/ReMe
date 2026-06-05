"""Claude Code SDK backend for the unified agent wrapper."""

from typing import Any, TYPE_CHECKING

from .base_agent_wrapper import BaseAgentWrapper
from ..component_registry import R

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


@R.register("claude_code")
class CcAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by Claude Code SDK.

    Kwargs:
        system_prompt: System prompt for the agent.
        model: Claude model to use.
        permission_mode: Permission mode for tool execution.
        max_turns: Maximum conversation turns.
        tools: list[BaseJob] to register as agent wrapper tools.
    """

    @staticmethod
    def _make_tool(job: "BaseJob"):
        from claude_agent_sdk import SdkMcpTool

        async def run_job(args):
            response = await job(**args)
            return {
                "content": [{"type": "text", "text": str(response.answer)}],
                "is_error": not response.success,
            }

        return SdkMcpTool(
            name=job.name,
            description=job.description,
            input_schema=job.parameters,
            handler=run_job,
        )

    async def reply(self, inputs: Any, session_id: str | None = None, **kwargs) -> tuple[str, Any]:
        from claude_agent_sdk import query, ResultMessage, create_sdk_mcp_server
        from claude_agent_sdk.types import ClaudeAgentOptions

        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)

        opts = ClaudeAgentOptions()
        opts.system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        opts.model = kwargs["model"]
        opts.permission_mode = kwargs.get("permission_mode", None)
        opts.max_turns = kwargs.get("max_turns", 50)
        if session_id:
            opts.session_id = session_id
            opts.fork_session = True

        tools: list["BaseJob"] = kwargs.get("tools", [])
        if tools:
            sdk_tools = [self._make_tool(job) for job in tools]
            server = create_sdk_mcp_server(name="reme_tools", tools=sdk_tools)
            if isinstance(opts.mcp_servers, dict):
                opts.mcp_servers["reme"] = server
            else:
                opts.mcp_servers = {"reme": server}
            opts.allowed_tools.extend(job.name for job in tools)

        output_schema = kwargs.get("output_schema")
        if output_schema:
            opts.output_format = {"type": "json_schema", "schema": output_schema}

        if isinstance(inputs, str):
            prompt = inputs
        else:
            raise NotImplementedError("Only string input is supported for Claude Code.")

        last_msg = None
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, ResultMessage):
                last_msg = msg

        if last_msg is None:
            raise ValueError("No message received from Claude Code.")

        result = last_msg.structured_output if output_schema and last_msg.structured_output else last_msg
        return last_msg.session_id or "", result
