"""Main AgentScope agent that executes the Meta-ReMe optimization loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentscope.agent import Agent, ContextConfig, ReActConfig
from agentscope.message import Msg, UserMsg
from agentscope.model import ChatModelBase
from agentscope.permission import AdditionalWorkingDirectory, PermissionBehavior, PermissionMode, PermissionRule
from agentscope.state import AgentState
from agentscope.tool import Bash, Edit, Glob, Grep, Read, Toolkit, Write

from ..tools.diagnostic_tool import create_diagnostic_subagent_tool
from ..tools.validation_tool import create_validation_tool
from ..utils import (
    CODE_REPOSITORY,
    create_memory_inspection_tools,
    create_validation_catalog_tool,
    load_agent_prompt,
    resolve_workspace,
)


class OptimizerAgent(Agent):
    """Optimizer agent with opt-in AgentScope context compression."""

    def __init__(self, *args: Any, compression_enabled: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compression_enabled = compression_enabled

    async def compress_context(self, *args: Any, **kwargs: Any) -> None:
        """Compress context only when explicitly enabled by Meta-ReMe configuration."""

        if self.compression_enabled:
            await super().compress_context(*args, **kwargs)


def create_optimizer_agent(
    model: ChatModelBase,
    workspace: str | Path | None = None,
    *,
    diagnostic_model: ChatModelBase | None = None,
    state: AgentState | None = None,
    validation_concurrency: int = 1,
    max_iters: int = 80,
) -> Agent:
    """Create the coding agent that searches for a higher-scoring candidate."""

    workspace_path = resolve_workspace(workspace)
    repository = workspace_path / CODE_REPOSITORY
    if not isinstance(max_iters, int) or isinstance(max_iters, bool) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")
    prompt = load_agent_prompt(__file__)
    validation = create_validation_tool(workspace_path, validation_concurrency)
    diagnosis = create_diagnostic_subagent_tool(diagnostic_model or model, workspace_path)
    catalog = create_validation_catalog_tool(workspace_path)
    memory_tools = create_memory_inspection_tools(workspace_path)
    state = state or AgentState()
    state.permission_context.mode = PermissionMode.DONT_ASK
    state.permission_context.working_directories[str(repository)] = AdditionalWorkingDirectory(
        path=str(repository),
        source="meta-reme optimizer",
    )
    for tool in (validation, diagnosis, catalog, *memory_tools):
        state.permission_context.allow_rules[tool.name] = [
            PermissionRule(
                tool_name=tool.name,
                rule_content=None,
                behavior=PermissionBehavior.ALLOW,
                source="meta-reme optimizer workflow",
            ),
        ]
    workspace_pattern = f"{workspace_path}/**"
    for tool_name in ("Read", "Grep", "Glob"):
        state.permission_context.allow_rules[tool_name] = [
            PermissionRule(
                tool_name=tool_name,
                rule_content=workspace_pattern,
                behavior=PermissionBehavior.ALLOW,
                source="meta-reme optimizer read-only inspection",
            ),
        ]
    repository_pattern = f"{repository}/**"
    for tool_name in ("Edit", "Write"):
        state.permission_context.allow_rules[tool_name] = [
            PermissionRule(
                tool_name=tool_name,
                rule_content=repository_pattern,
                behavior=PermissionBehavior.ALLOW,
                source="meta-reme optimizer candidate repository",
            ),
        ]
    state.permission_context.allow_rules["Bash"] = [
        PermissionRule(
            tool_name="Bash",
            rule_content=command,
            behavior=PermissionBehavior.ALLOW,
            source="meta-reme optimizer candidate workflow",
        )
        for command in (
            f"find {workspace_path}",
            "git status",
            "git log",
            "git diff",
            "git show",
            "git branch",
            "git rev-parse",
            "git add",
            "git commit",
            "git switch",
            "git checkout -b",
            "python3 -m json.tool",
            "pytest tests/",
            "pre-commit run",
        )
    ]
    return OptimizerAgent(
        name=prompt.name,
        system_prompt=prompt.render_system(workspace=workspace_path, code_repository=repository),
        model=model,
        toolkit=Toolkit(
            tools=[
                Read(),
                Grep(),
                Glob(),
                Edit(),
                Write(),
                Bash(cwd=repository),
                catalog,
                *memory_tools,
                validation,
                diagnosis,
            ],
        ),
        state=state,
        context_config=ContextConfig(
            trigger_ratio=prompt.context_compression.trigger_ratio,
            reserve_ratio=prompt.context_compression.reserve_ratio,
        ),
        react_config=ReActConfig(max_iters=max_iters),
        compression_enabled=prompt.context_compression.enabled,
    )


async def run_optimizer_agent(agent: Agent, objective: str | None = None) -> Msg:
    """Start one complete code-test-diagnose-branch-optimize loop."""

    prompt = load_agent_prompt(__file__)
    return await agent.reply(
        UserMsg(
            name="meta_reme_orchestrator",
            content=prompt.render_task(
                objective=(objective or "Improve mean_query_score on the installed benchmark cases.").strip(),
            ),
        ),
    )
