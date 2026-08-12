"""Read-only subagent for evidence-backed diagnosis of ReMe candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from agentscope.agent import Agent, ReActConfig
from agentscope.message import Msg, UserMsg
from agentscope.model import ChatModelBase
from agentscope.permission import PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Glob, Grep, Read, ToolBase, Toolkit

from ..utils import (
    CODE_REPOSITORY,
    create_git_inspection_tools,
    create_memory_inspection_tools,
    create_validation_catalog_tool,
    load_agent_prompt,
    resolve_workspace,
)


def create_diagnostic_agent(
    model: ChatModelBase,
    workspace: str | Path | None = None,
    *,
    state: AgentState | None = None,
    max_iters: int = 30,
    extra_tools: Sequence[ToolBase] = (),
) -> Agent:
    """Create a read-only diagnostic agent bound to one Meta-ReMe workspace."""

    workspace_path = resolve_workspace(workspace)
    if not isinstance(max_iters, int) or isinstance(max_iters, bool) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")
    prompt = load_agent_prompt(__file__)
    state = state or AgentState()
    state.permission_context.mode = PermissionMode.EXPLORE
    tools: list[ToolBase] = [
        Read(),
        Grep(),
        Glob(),
        create_validation_catalog_tool(workspace_path),
        *create_git_inspection_tools(workspace_path),
        *create_memory_inspection_tools(workspace_path),
        *extra_tools,
    ]
    return Agent(
        name=prompt.name,
        system_prompt=prompt.render_system(
            workspace=workspace_path,
            code_repository=workspace_path / CODE_REPOSITORY,
        ),
        model=model,
        toolkit=Toolkit(tools=tools),
        state=state,
        react_config=ReActConfig(max_iters=max_iters),
    )


async def run_diagnostic_agent(
    agent: Agent,
    *,
    request: str,
    trajectory_path: str | Path,
    validation_paths: Sequence[str] = (),
) -> Msg:
    """Ask a configured diagnostic agent to produce one diagnosis report."""

    prompt = load_agent_prompt(__file__)
    message = prompt.render_task(
        request=request.strip() or "Diagnose the current candidate and prioritize score-limiting issues.",
        trajectory_path=Path(trajectory_path).resolve(),
        validation_paths="\n".join(f"- {Path(path).resolve()}" for path in validation_paths)
        or "- all workspace validations",
    )
    return await agent.reply(UserMsg(name="meta_reme_orchestrator", content=message))


def message_text(message: Msg) -> str:
    """Extract the plain-text portion of an AgentScope message."""

    return message.get_text_content() or ""
