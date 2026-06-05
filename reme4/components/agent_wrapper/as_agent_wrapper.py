"""AgentScope backend for the unified agent wrapper."""

import datetime
import uuid
import zoneinfo
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agentscope.agent import Agent, ContextConfig, ModelConfig, ReActConfig
from agentscope.message import TextBlock, ToolResultState, UserMsg
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit

from .base_agent_wrapper import BaseAgentWrapper
from ..as_llm import BaseAsLLM
from ..component_registry import R
from ...utils import AsStateHandler

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


@R.register("agentscope")
class AsAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by AgentScope framework."""

    def __init__(self, as_llm: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.as_llm = self.bind(as_llm, BaseAsLLM, optional=False)

    @staticmethod
    def _make_tool(job: "BaseJob") -> FunctionTool:
        async def run_job(**kwargs) -> ToolChunk:
            response = await job(**kwargs)
            state = ToolResultState.SUCCESS if response.success else ToolResultState.ERROR
            return ToolChunk(content=[TextBlock(text=str(response.answer))], state=state)

        tool = FunctionTool(func=run_job, name=job.name, description=job.description)
        if job.parameters:
            tool.input_schema = job.parameters
        return tool

    def _build_agent(self, inputs: Any, **kwargs) -> tuple[Agent, Any]:
        """Build an Agent instance from kwargs. Returns (agent, processed_inputs)."""
        model = self.as_llm.model if self.as_llm else None
        if model is None:
            raise ValueError("AsAgentWrapper requires a bound as_llm component with a valid model.")

        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)

        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        tools: list["BaseJob"] = kwargs.get("tools", [])
        toolkit = kwargs.get("toolkit") or (
            Toolkit(tools=[self._make_tool(job) for job in tools]) if tools else Toolkit()
        )

        perm_mode = PermissionMode(kwargs.get("permission_mode", "bypass"))
        state = AgentState(permission_context=PermissionContext(mode=perm_mode))

        agent = Agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            toolkit=toolkit,
            state=state,
            model_config=ModelConfig(**(kwargs.get("model_config") or {})),
            context_config=ContextConfig(**(kwargs.get("context_config") or {})),
            react_config=ReActConfig(**(kwargs.get("react_config") or {})),
        )

        if isinstance(inputs, str):
            inputs = UserMsg(name="user", content=inputs)

        return agent, inputs

    def _session_path(self, session_id: str) -> Path:
        tz_name = self.app_context.app_config.timezone if self.app_context else None
        if tz_name:
            try:
                dt = datetime.datetime.now(zoneinfo.ZoneInfo(tz_name))
            except Exception:
                dt = datetime.datetime.now()
        else:
            dt = datetime.datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        resource = self.app_context.app_config.resource_dir if self.app_context else "resource"
        return self.vault_path / resource / date_str / f"session_reme_{session_id}.jsonl"

    async def reply(self, inputs: Any, **kwargs) -> tuple[str, Any]:
        session_id: str = kwargs.pop("session_id", "")
        fork_session: bool = kwargs.pop("fork_session", False)

        agent, inputs = self._build_agent(inputs, **kwargs)

        if session_id:
            path = self._session_path(session_id)
            if path.exists():
                loaded = await AsStateHandler(path).load()
                agent.state.session_id = loaded.session_id
                agent.state.summary = loaded.summary
                agent.state.context = loaded.context
                agent.state.reply_id = loaded.reply_id
                agent.state.cur_iter = loaded.cur_iter

        await agent.observe(inputs)
        await agent.reply()
        last_msg = agent.state.context[-1]

        if session_id:
            if fork_session:
                new_sid = uuid.uuid4().hex
                agent.state.session_id = new_sid
                save_path = self._session_path(new_sid)
            else:
                save_path = self._session_path(session_id)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            await AsStateHandler(save_path).dump(agent.state)

        output_schema: dict | None = kwargs.get("output_schema")
        if output_schema is not None:
            assert self.as_llm is not None, "AsAgentWrapper requires a bound as_llm component with a valid model."
            model = self.as_llm.model
            assert model is not None, "AsAgentWrapper requires a bound as_llm component with a valid model."
            res = await model.generate_structured_output(
                messages=agent.state.context,
                structured_model=output_schema,
            )
            return agent.state.session_id, {"message": last_msg, "structured_output": res.content}

        return agent.state.session_id, last_msg

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[Any, None]:
        """Stream agent events via AgentScope's reply_stream API."""
        agent, inputs = self._build_agent(inputs, **kwargs)

        async for event in agent.reply_stream(inputs):
            yield event
