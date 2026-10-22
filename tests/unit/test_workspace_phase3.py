"""Phase 3: nested propagation, session namespacing, and background binding."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from reme.application import Application
from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.component_registry import R
from reme.components.job import BackgroundJob, BaseJob, CronJob
from reme.components.runtime_context import RuntimeContext
from reme.steps.base_step import BaseStep
from reme.steps.evolve.base_auto_resource import _compute_agent_session_id
from reme.steps.index.draft import AddDraftStep


class _AliasReporterStep(BaseStep):
    """Answer with the bound workspace alias."""

    async def execute(self):
        bound = self.context.workspace
        self.context.response.answer = bound.alias if bound is not None else ""


class _NestedCallerStep(BaseStep):
    """Call the report job as a nested job and relay its answer."""

    async def execute(self):
        response = await self.run_job("report")
        self.context.response.answer = f"parent->{response.answer}"


def _registry():
    registry = R.copy()
    registry.register(_AliasReporterStep, "alias_reporter_step")
    registry.register(_NestedCallerStep, "nested_caller_step")
    return registry


def _context(tmp_path):
    return ApplicationContext(
        registry=_registry(),
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
    )


def _job(context, name, backend="alias_reporter_step", job_cls=BaseJob, **kwargs):
    return job_cls(
        name=name,
        app_context=context,
        parameters={"type": "object", "properties": {}},
        steps=[{"backend": backend}],
        **kwargs,
    )


# ----- Nested propagation ----------------------------------------------------


@pytest.mark.asyncio
async def test_nested_run_job_inherits_workspace_binding(tmp_path):
    context = _context(tmp_path)
    parent = _job(context, "parent", backend="nested_caller_step")
    child = _job(context, "report")
    context.jobs = {"parent": parent, "report": child}
    await parent.start()
    await child.start()

    response = await parent(workspace="beta")
    assert response.answer == "parent->beta"
    # Default workspace stays the implicit binding for unscoped calls.
    response = await parent()
    assert response.answer == "parent->alpha"
    # Explicit nested selection wins over inheritance.
    caller = _NestedCallerStep(app_context=context)
    caller.context = RuntimeContext(workspace=context.workspaces["alpha"])
    response = await caller.run_job("report", workspace="beta")
    assert response.answer == "beta"


def test_workspace_forwarded_through_injected_channel_only():
    # The calling step's binding joins the server-owned injected kwargs...
    injected = BaseAgentWrapper._resolve_injected_job_kwargs(
        {"injected_job_kwargs": {"_allowed_paths": ["daily/x.md"]}, "workspace": "beta"},
    )
    assert injected == {"_allowed_paths": ["daily/x.md"], "workspace": "beta"}
    # ...and model-supplied tool arguments can never override it.
    with pytest.raises(ValueError, match="cannot be provided by the model"):
        BaseAgentWrapper._merge_injected_job_kwargs({"workspace": "alpha"}, injected)


# ----- Session/tool-context namespacing --------------------------------------


def test_agent_session_uuid_namespaced_per_workspace():
    path = "resource/2026-01-01/report.pdf"
    legacy = _compute_agent_session_id(path)
    alpha = _compute_agent_session_id(path, workspace_alias="alpha")
    beta = _compute_agent_session_id(path, workspace_alias="beta")
    # Default/empty alias keeps the historical UUID for backward compatibility.
    assert _compute_agent_session_id(path, workspace_alias="") == legacy
    assert len({legacy, alpha, beta}) == 3


@pytest.mark.asyncio
async def test_tool_context_state_namespaced_per_workspace(tmp_path):
    context = _context(tmp_path)
    step = AddDraftStep(app_context=context)
    for alias in ("alpha", "beta"):
        step.context = RuntimeContext(
            workspace=context.workspaces[alias],
            tool_context_id="ctx-1",
            text=f"draft-{alias}",
        )
        await step.execute()

    stores = context.metadata["tool_contexts"]
    # The default workspace keeps the legacy plain key; other aliases use
    # unambiguous tuple keys.
    assert set(stores) == {"ctx-1", ("beta", "ctx-1")}
    assert stores["ctx-1"]["drafts"] == ["draft-alpha"]
    assert stores[("beta", "ctx-1")]["drafts"] == ["draft-beta"]


@pytest.mark.asyncio
async def test_codex_thread_tool_contexts_namespaced_per_workspace(tmp_path):
    wrapper = CodexAgentWrapper(
        app_context=ApplicationContext(
            workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
        ),
    )

    class FakeCodex:
        """Minimal Codex double returning a fixed thread id."""

        async def thread_start(self, **_kwargs):
            return SimpleNamespace(id="thread-1")

        async def thread_resume(self, _thread_id, **_kwargs):
            return SimpleNamespace(id="thread-1")

    codex = FakeCodex()
    await wrapper._open_thread(codex, {"tool_context_id": "ctx-a"})
    # Same thread id under a different workspace is a different guard entry.
    await wrapper._open_thread(codex, {"session_id": "thread-1", "workspace": "beta", "tool_context_id": "ctx-b"})
    # Same workspace resume with changed constraints still raises.
    with pytest.raises(ValueError, match="cannot change"):
        await wrapper._open_thread(codex, {"session_id": "thread-1", "tool_context_id": "ctx-b"})

    assert sorted(wrapper._thread_tool_contexts) == ["beta/thread-1", "thread-1"]


# ----- Background/cron binding -----------------------------------------------


@pytest.mark.asyncio
async def test_background_job_binds_configured_workspace(tmp_path):
    context = _context(tmp_path)
    job = _job(context, "bg", job_cls=BackgroundJob, workspace="beta")
    job.step_specs = [job._resolve_step({"backend": "alias_reporter_step"})]
    assert (await job()).answer == "beta"

    default_job = _job(context, "bg-default", job_cls=BackgroundJob)
    default_job.step_specs = [default_job._resolve_step({"backend": "alias_reporter_step"})]
    assert (await default_job()).answer == "alpha"

    with pytest.raises(ValueError, match="Unknown workspace 'nope'"):
        await _job(context, "bg-bad", job_cls=BackgroundJob, workspace="nope")()


@pytest.mark.asyncio
async def test_cron_job_binds_configured_workspace(tmp_path):
    context = _context(tmp_path)
    job = _job(context, "cron", job_cls=CronJob, cron="0 0 * * *", workspace="beta")
    job.step_specs = [job._resolve_step({"backend": "alias_reporter_step"})]
    assert (await job._execute_steps()).answer == "beta"

    default_job = _job(context, "cron-default", job_cls=CronJob, cron="0 0 * * *")
    default_job.step_specs = [default_job._resolve_step({"backend": "alias_reporter_step"})]
    assert (await default_job._execute_steps()).answer == "alpha"


@pytest.mark.asyncio
async def test_per_alias_watch_variant_jobs_write_own_workspace(tmp_path):
    """Two watch-style variants with explicit aliases act on their own workspace."""
    app = Application(
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components={
            "tokenizer": {"default": {"backend": "regex"}},
            "file_store": {"default": {"backend": "local", "embedding_store": "", "tag_index": ""}},
            "keyword_index": {"default": {"backend": "bm25"}},
            "file_graph": {"default": {"backend": "local"}},
        },
        jobs={
            "watch_alpha": {
                "backend": "background",
                "workspace": "alpha",
                "enable_serve": False,
                "parameters": {"type": "object", "properties": {}},
                "path": "daily/marker.md",
                "content": "alpha-watch",
                "steps": [{"backend": "write_step"}],
            },
            "watch_beta": {
                "backend": "background",
                "workspace": "beta",
                "enable_serve": False,
                "parameters": {"type": "object", "properties": {}},
                "path": "daily/marker.md",
                "content": "beta-watch",
                "steps": [{"backend": "write_step"}],
            },
        },
    )
    async with app:
        await asyncio.sleep(0.5)

    alpha = (tmp_path / "alpha" / "daily" / "marker.md").read_text(encoding="utf-8")
    beta = (tmp_path / "beta" / "daily" / "marker.md").read_text(encoding="utf-8")
    assert "alpha-watch" in alpha and "beta-watch" not in alpha
    assert "beta-watch" in beta and "alpha-watch" not in beta


# ----- MCP boundary and Codex bridge -----------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_boundary_forwards_injected_workspace(tmp_path):
    from reme.components.service.mcp_tools import add_mcp_job

    context = _context(tmp_path)
    job = _job(context, "report")
    job.step_specs = [job._resolve_step({"backend": "alias_reporter_step"})]

    tools = []
    server = SimpleNamespace(add_tool=tools.append)
    assert add_mcp_job(server, job, injected_job_kwargs={"workspace": "beta"}, tool_error_on_failure=False)

    # The server-injected binding reaches the job boundary and wins over callers.
    assert await tools[0].fn() == "beta"
    with pytest.raises(Exception, match="cannot be provided by the caller"):
        await tools[0].fn(workspace="alpha")


@pytest.mark.asyncio
async def test_codex_bridge_uses_bound_workspace_path(tmp_path):
    import json

    context = _context(tmp_path)
    context.jobs = {"search_job": _job(context, "search_job")}
    wrapper = CodexAgentWrapper(app_context=context)

    config = wrapper._mcp_server_config({"job_tools": ["search_job"], "workspace": "beta"})
    assert config["args"][config["args"].index("--workspace") + 1] == str(tmp_path / "beta")
    raw = config["args"][config["args"].index("--injected-job-kwargs") + 1]
    assert json.loads(raw) == {"workspace": "beta"}

    default_config = wrapper._mcp_server_config({"job_tools": ["search_job"]})
    assert default_config["args"][default_config["args"].index("--workspace") + 1] == str(tmp_path / "alpha")

    with pytest.raises(ValueError, match="Unknown workspace 'nope'"):
        wrapper._mcp_server_config({"job_tools": ["search_job"], "workspace": "nope"})

    await wrapper.close()


# ----- Review follow-ups: explicit bindings, session isolation, key collisions


@pytest.mark.asyncio
async def test_nested_call_default_binding_beats_child_config(tmp_path):
    """An explicit default binding propagates; the child's configured alias must not win."""
    context = _context(tmp_path)
    parent = _job(context, "parent", backend="nested_caller_step")
    child = _job(context, "report", workspace="beta")  # job-level configured binding
    context.jobs = {"parent": parent, "report": child}
    await parent.start()
    await child.start()

    # Top-level call to the configured child still honors its configuration.
    assert (await child()).answer == "beta"
    # Nested under an alpha-bound parent, the parent binding wins.
    response = await parent(workspace="alpha")
    assert response.answer == "parent->alpha"
    response = await parent(workspace="beta")
    assert response.answer == "parent->beta"


@pytest.mark.asyncio
async def test_tool_context_keys_cannot_collide_across_workspaces(tmp_path):
    context = _context(tmp_path)
    step = AddDraftStep(app_context=context)
    # Default-workspace id that spells like another alias's namespaced key.
    step.context = RuntimeContext(workspace=context.workspaces["alpha"], tool_context_id="beta/ctx-1", text="d-alpha")
    await step.execute()
    step.context = RuntimeContext(workspace=context.workspaces["beta"], tool_context_id="ctx-1", text="d-beta")
    await step.execute()

    stores = context.metadata["tool_contexts"]
    assert stores["beta/ctx-1"]["drafts"] == ["d-alpha"]
    assert stores[("beta", "ctx-1")]["drafts"] == ["d-beta"]


@pytest.mark.asyncio
async def test_agentscope_sessions_isolated_per_workspace(tmp_path):
    from uuid import uuid4

    from agentscope.permission import PermissionMode

    from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper

    context = _context(tmp_path)
    wrapper = AsAgentWrapper(app_context=context, as_llm="")
    wrapper.as_llm = SimpleNamespace(model=object())

    session_id = str(uuid4())
    state = await wrapper._load_state({"session_id": session_id, "workspace": "beta"}, PermissionMode.DEFAULT)
    state.summary = "beta summary"
    await wrapper._dump_state(state, {"workspace": "beta"})

    session_file = tmp_path / "beta" / "mem_session" / "agentscope" / f"{session_id}.jsonl"
    assert session_file.is_file()
    assert not (tmp_path / "alpha" / "mem_session" / "agentscope" / f"{session_id}.jsonl").exists()

    # Same id under another workspace must not read beta's state.
    with pytest.raises(FileNotFoundError):
        await wrapper._load_state({"resume": session_id, "workspace": "alpha"}, PermissionMode.DEFAULT)
    resumed = await wrapper._load_state({"resume": session_id, "workspace": "beta"}, PermissionMode.DEFAULT)
    assert resumed.summary == "beta summary"
    # Default workspace is untouched by the beta session.
    with pytest.raises(FileNotFoundError):
        await wrapper._load_state({"resume": session_id}, PermissionMode.DEFAULT)


def test_cc_options_follow_bound_workspace(tmp_path):
    from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper

    context = _context(tmp_path)
    wrapper = CcAgentWrapper(app_context=context)

    opts = wrapper._build_options("hi", workspace="beta")
    assert opts.env["CLAUDE_CONFIG_DIR"].startswith(str(tmp_path / "beta"))
    assert opts.cwd == tmp_path / "beta"

    default_opts = wrapper._build_options("hi")
    assert default_opts.env["CLAUDE_CONFIG_DIR"].startswith(str(tmp_path / "alpha"))


@pytest.mark.asyncio
async def test_codex_thread_cwd_follows_bound_workspace(tmp_path):
    context = _context(tmp_path)
    wrapper = CodexAgentWrapper(app_context=context)
    observed = {}

    class FakeCodex:
        """Capture thread_start parameters without an app-server."""

        async def thread_start(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(id="thread-1")

    await wrapper._open_thread(FakeCodex(), {"workspace": "beta"})
    assert observed["cwd"] == str(tmp_path / "beta")

    observed.clear()
    await wrapper._open_thread(FakeCodex(), {})
    assert observed["cwd"] != str(tmp_path / "beta")


# ----- Codex bridge: real child-process execution ----------------------------


@pytest.mark.asyncio
async def test_codex_bridge_executes_non_default_alias(tmp_path):
    """The bridge child rebuilds the full workspace mapping and binds via the alias."""
    import json
    import sys

    from fastmcp import Client
    from fastmcp.client import StdioTransport

    snapshot = tmp_path / "bridge.json"
    snapshot.write_text(
        json.dumps(
            {
                "workspace_dir": {"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
                "enable_logo": False,
                "log_to_console": False,
                "log_to_file": False,
                "service": {"backend": "mcp"},
                "components": {
                    "tokenizer": {"default": {"backend": "regex"}},
                    "file_store": {"default": {"backend": "local", "embedding_store": "", "tag_index": ""}},
                    "keyword_index": {"default": {"backend": "bm25"}},
                    "file_graph": {"default": {"backend": "local"}},
                },
                "jobs": {
                    "write_note": {
                        "backend": "base",
                        "description": "write a note",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["path", "content"],
                        },
                        "steps": [{"backend": "write_step"}],
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-m",
            "reme.components.agent_wrapper.codex_mcp_server",
            "--config",
            str(snapshot),
            "--workspace",
            str(tmp_path / "beta"),
            "--workspace-alias",
            "beta",
            "--job",
            "write_note",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        keep_alive=False,
    )

    async with Client(transport, timeout=None, init_timeout=30) as client:
        await asyncio.wait_for(
            client.call_tool("write_note", {"path": "daily/bridge.md", "content": "via-beta"}),
            timeout=30,
        )

    assert "via-beta" in (tmp_path / "beta" / "daily" / "bridge.md").read_text(encoding="utf-8")
    assert not (tmp_path / "alpha" / "daily" / "bridge.md").exists()


@pytest.mark.asyncio
async def test_codex_bridge_rejects_alias_without_mapping_snapshot(tmp_path):
    import json
    import sys

    from fastmcp import Client
    from fastmcp.client import StdioTransport

    snapshot = tmp_path / "bridge-single.json"
    snapshot.write_text(
        json.dumps(
            {
                "workspace_dir": str(tmp_path / "only"),
                "enable_logo": False,
                "log_to_console": False,
                "log_to_file": False,
                "service": {"backend": "mcp"},
                "jobs": {
                    "write_note": {
                        "backend": "base",
                        "parameters": {"type": "object", "properties": {}},
                        "steps": [{"backend": "write_step"}],
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-m",
            "reme.components.agent_wrapper.codex_mcp_server",
            "--config",
            str(snapshot),
            "--workspace",
            str(tmp_path / "only"),
            "--workspace-alias",
            "beta",
            "--job",
            "write_note",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        keep_alive=False,
    )

    # The child process fails fast with a ValueError; the client sees the closed connection.
    with pytest.raises(Exception, match="Connection closed"):
        async with Client(transport, timeout=None, init_timeout=30):
            pass
