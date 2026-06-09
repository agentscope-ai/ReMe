"""Claude Code SDK backend for the unified agent wrapper."""

from typing import Any, TYPE_CHECKING

from .base_agent_wrapper import BaseAgentWrapper
from ..component_registry import R

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


@R.register("claude_code")
class CcAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by Claude Code SDK."""

    @staticmethod
    def _make_tool(job: "BaseJob"):
        from claude_agent_sdk import SdkMcpTool

        async def run_job(args):
            response = await job(**args)
            return {"content": [{"type": "text", "text": str(response.answer)}], "is_error": not response.success}

        return SdkMcpTool(name=job.name, description=job.description, input_schema=job.parameters, handler=run_job)

    async def reply(self, inputs: Any, **kwargs) -> tuple[str, Any]:
        from claude_agent_sdk import query, ResultMessage, create_sdk_mcp_server
        from claude_agent_sdk.types import ClaudeAgentOptions

        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)

        session_id: str = kwargs.pop("session_id", "")
        fork_session: bool = kwargs.pop("fork_session", False)

        sp = kwargs.get("system_prompt")
        if isinstance(sp, str):
            kwargs["system_prompt"] = {
                "type": "preset",
                "preset": "claude_code",
                "append": sp,
                "exclude_dynamic_sections": True,
            }
        kwargs.setdefault("setting_sources", [])

        opts = ClaudeAgentOptions()
        skip_keys = {"tools", "output_schema"}
        for k, v in kwargs.items():
            if k not in skip_keys and hasattr(opts, k):
                setattr(opts, k, v)

        if session_id:
            opts.resume = session_id
            if fork_session:
                opts.fork_session = True
        elif fork_session:
            # fork_session with no session_id to fork from is meaningless.
            raise ValueError("fork_session=True requires a non-empty session_id")

        tools: list["BaseJob"] = kwargs.get("tools", [])
        if tools:
            sdk_tools = [self._make_tool(job) for job in tools]
            server = create_sdk_mcp_server(name="reme_tools", tools=sdk_tools)
            opts.mcp_servers = (opts.mcp_servers if isinstance(opts.mcp_servers, dict) else {}) | {"reme": server}
            opts.allowed_tools.extend(job.name for job in tools)

        if output_schema := kwargs.get("output_schema"):
            schema_dict = (
                output_schema.model_json_schema() if hasattr(output_schema, "model_json_schema") else output_schema
            )
            opts.output_format = {"type": "json_schema", "schema": schema_dict}

        if not isinstance(inputs, str):
            raise NotImplementedError("Only string input is supported for Claude Code.")

        last_msg = None
        async for msg in query(prompt=inputs, options=opts):
            if isinstance(msg, ResultMessage):
                last_msg = msg

        if last_msg is None:
            raise ValueError("No message received from Claude Code.")
        
        if output_schema:
            structured = last_msg.structured_output or {}
            return last_msg.session_id or "", {"message": last_msg, "structured_output": structured}
        return last_msg.session_id or "", last_msg
