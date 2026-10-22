"""Ownership restructure: WorkspaceContext owns state; bindings stay explicit."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio

import pytest

from reme.application import Application
from reme.components import ApplicationContext
from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.component_registry import R
from reme.components.job import BaseJob
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
from reme.steps.base_step import BaseStep
from reme.steps.evolve.dream.integrate import DreamIntegrateStep


class _AliasReporterStep(BaseStep):
    """Answer with the bound workspace alias."""

    async def execute(self):
        bound = self.context.workspace
        self.context.response.answer = bound.alias if bound is not None else ""


def _context(tmp_path):
    registry = R.copy()
    registry.register(_AliasReporterStep, "alias_reporter_step")
    return ApplicationContext(
        registry=registry,
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
    )


def _reporter_job(context, name, **kwargs):
    return BaseJob(
        name=name,
        app_context=context,
        parameters={"type": "object", "properties": {}},
        steps=[{"backend": "alias_reporter_step"}],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_default_binding_reaches_real_agent_tool_call(tmp_path):
    """A default-bound caller keeps its binding through a real FunctionTool call."""
    context = _context(tmp_path)
    job = _reporter_job(context, "report", workspace="beta")  # child configured for beta
    job.step_specs = [job._resolve_step({"backend": "alias_reporter_step"})]

    # The wrapper tool boundary injects the caller's resolved (default) binding.
    tool = AsAgentWrapper._make_tool(job, None, None, workspace="alpha")
    chunk = await tool._func()
    text = "".join(block.text for block in chunk.content if hasattr(block, "text"))
    assert text == "alpha"

    # Model-supplied workspace arguments are rejected at the same boundary.
    with pytest.raises(ValueError, match="cannot be provided by the model"):
        await tool._func(workspace="beta")


def test_call_cwd_resolves_relative_against_bound_workspace(tmp_path):
    context = _context(tmp_path)
    absolute = tmp_path / "elsewhere"

    wrapper = CodexAgentWrapper(app_context=context, cwd="relative/sub")
    assert wrapper._call_cwd({"workspace": "beta"}) == tmp_path / "beta" / "relative" / "sub"
    assert wrapper._call_cwd({}) == tmp_path / "alpha" / "relative" / "sub"

    absolute_wrapper = CodexAgentWrapper(app_context=context, cwd=str(absolute))
    assert absolute_wrapper._call_cwd({"workspace": "beta"}) == absolute

    default_wrapper = CodexAgentWrapper(app_context=context)
    assert default_wrapper._call_cwd({"workspace": "beta"}) == tmp_path / "beta"
    assert default_wrapper._call_cwd({}) != tmp_path / "beta"  # project path, not a workspace


def test_dream_integration_lock_is_workspace_scoped(tmp_path):
    context = _context(tmp_path)
    step = DreamIntegrateStep(app_context=context)

    step.context = RuntimeContext(workspace=context.workspaces["alpha"])
    lock_alpha = step._integration_lock()
    step.context = RuntimeContext(workspace=context.workspaces["beta"])
    lock_beta = step._integration_lock()
    step.context = RuntimeContext(workspace=context.workspaces["alpha"])
    assert step._integration_lock() is lock_alpha
    assert lock_alpha is not lock_beta


# ----- Real watch loops -------------------------------------------------------

_WATCH_COMPONENTS = {
    "tokenizer": {"default": {"backend": "regex"}},
    "file_store": {"default": {"backend": "local", "embedding_store": "", "tag_index": ""}},
    "keyword_index": {"default": {"backend": "bm25"}},
    "file_graph": {"default": {"backend": "local"}},
    "file_chunker": {"markdown": {"backend": "markdown", "supported_extensions": ["md"]}},
}


def _watch_job(alias):
    return {
        "backend": "background",
        "workspace": alias,
        "enable_serve": False,
        "watch_dirs": ["daily_dir"],
        "watch_suffixes": ["md"],
        "parameters": {"type": "object", "properties": {}},
        "steps": [
            {
                "backend": "watch_changes_step",
                "debounce": 200,
                "step": 100,
                "poll_delay_ms": 200,
                "dispatch_steps": [{"backend": "update_index_step", "persist": False}],
            },
        ],
    }


@pytest.mark.asyncio
async def test_real_watch_loops_index_own_workspace_and_clean_up(tmp_path):
    app = Application(
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components=_WATCH_COMPONENTS,
        jobs={"watch_alpha": _watch_job("alpha"), "watch_beta": _watch_job("beta")},
    )
    note = tmp_path / "beta" / "daily" / "watch-me.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    async with app:
        store_beta = app.context.get_component(ComponentEnum.FILE_STORE, "default", "beta")
        store_alpha = app.context.get_component(ComponentEnum.FILE_STORE, "default", "alpha")
        for i in range(30):  # up to ~15s for watcher startup + debounce + dispatch
            note.write_text(f"# beta watch\nreal watch event content v{i}\n", encoding="utf-8")
            await asyncio.sleep(0.5)
            if await store_beta.get_nodes():
                break
        nodes_beta = await store_beta.get_nodes()
        nodes_alpha = await store_alpha.get_nodes()

    assert [n.path for n in nodes_beta] == ["daily/watch-me.md"]
    assert nodes_alpha == []
    # Background loops stopped with the application; nothing keeps running.
    for job in app.context.jobs.values():
        assert job._task is None or job._task.done()
