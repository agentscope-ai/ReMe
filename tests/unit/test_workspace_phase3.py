"""Phase 3: nested propagation, session namespacing, and background binding."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from reme.application import Application
from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.component_registry import R
from reme.components.job import BackgroundJob, BaseJob, CronJob
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
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

    # State is owned by each WorkspaceContext; identical ids stay independent.
    assert context.workspaces["alpha"].tool_contexts["ctx-1"]["drafts"] == ["draft-alpha"]
    assert context.workspaces["beta"].tool_contexts["ctx-1"]["drafts"] == ["draft-beta"]
    assert "tool_contexts" not in context.metadata


@pytest.mark.asyncio
async def test_codex_thread_tool_contexts_namespaced_per_workspace(tmp_path):
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    wrapper = CodexAgentWrapper(app_context=context)

    class FakeCodex:
        """Minimal Codex double echoing the requested thread id."""

        def __init__(self):
            self.next_id = "thread-1"

        async def thread_start(self, **_kwargs):
            return SimpleNamespace(id=self.next_id)

        async def thread_resume(self, thread_id, **_kwargs):
            return SimpleNamespace(id=thread_id)

    codex = FakeCodex()
    await wrapper._open_thread(codex, {"tool_context_id": "ctx-a"})
    # A different thread under another workspace gets its own guard entry.
    codex.next_id = "thread-2"
    await wrapper._open_thread(codex, {"workspace": "beta", "tool_context_id": "ctx-b"})
    # Same workspace resume with changed constraints still raises.
    with pytest.raises(ValueError, match="cannot change"):
        await wrapper._open_thread(codex, {"session_id": "thread-1", "tool_context_id": "ctx-b"})
    # A thread owned by one workspace cannot be resumed from another.
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await wrapper._open_thread(codex, {"session_id": "thread-1", "workspace": "beta"})

    alpha_state = context.workspaces["alpha"].agent_state(wrapper.name)
    beta_state = context.workspaces["beta"].agent_state(wrapper.name)
    assert set(alpha_state.thread_tool_contexts) == {"thread-1"}
    assert set(beta_state.thread_tool_contexts) == {"thread-2"}
    assert not wrapper._thread_tool_contexts
    assert context.workspaces["alpha"].codex_thread_state.owned_thread_ids == {"thread-1"}
    assert context.workspaces["beta"].codex_thread_state.owned_thread_ids == {"thread-2"}


@pytest.mark.asyncio
async def test_codex_thread_ownership_persists_across_restart(tmp_path):
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})

    class FakeCodex:
        """Minimal Codex double echoing the requested thread id."""

        async def thread_start(self, **_kwargs):
            return SimpleNamespace(id="thread-1")

        async def thread_resume(self, thread_id, **_kwargs):
            return SimpleNamespace(id=thread_id)

        async def thread_fork(self, thread_id, **_kwargs):
            return SimpleNamespace(id=f"{thread_id}-fork")

    codex = FakeCodex()
    wrapper = CodexAgentWrapper(app_context=context)
    await wrapper._open_thread(codex, {"workspace": "alpha"})
    ownership = tmp_path / "alpha" / "mem_session" / "codex" / "owned_threads.json"
    assert ownership.is_file()

    # A fresh wrapper (restart) reloads the ownership records and still rejects.
    restarted = CodexAgentWrapper(app_context=context)
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await restarted._open_thread(codex, {"session_id": "thread-1", "workspace": "beta"})
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await restarted._open_thread(codex, {"session_id": "thread-1", "workspace": "beta", "fork_session": True})
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await restarted.compact_session("thread-1", workspace="beta")
    # The owning workspace keeps working normally.
    thread = await restarted._open_thread(codex, {"session_id": "thread-1", "workspace": "alpha"})
    assert thread.id == "thread-1"
    # Threads recorded nowhere pass through and let the SDK decide.
    await restarted._open_thread(codex, {"session_id": "thread-elsewhere", "workspace": "beta"})


@pytest.mark.asyncio
async def test_codex_thread_ownership_is_shared_and_atomic_across_wrappers(tmp_path):
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    wrapper_a = CodexAgentWrapper(name="codex-a", app_context=context)
    wrapper_b = CodexAgentWrapper(name="codex-b", app_context=context)

    assert await wrapper_b._owned_thread_ids("alpha") == set()
    await wrapper_a._record_thread_ownership("alpha", "alpha-thread")
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await wrapper_b._check_thread_ownership("alpha-thread", "beta")

    await wrapper_b._record_thread_ownership("alpha", "second-alpha-thread")
    ownership = tmp_path / "alpha" / "mem_session" / "codex" / "owned_threads.json"
    assert set(json.loads(ownership.read_text(encoding="utf-8"))["thread_ids"]) == {
        "alpha-thread",
        "second-alpha-thread",
    }
    assert not list(ownership.parent.glob("*.tmp"))

    restarted_context = ApplicationContext(
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
    )
    restarted = CodexAgentWrapper(name="codex-restarted", app_context=restarted_context)
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await restarted._check_thread_ownership("alpha-thread", "beta")


@pytest.mark.asyncio
async def test_codex_resume_and_compact_claim_external_threads(tmp_path, monkeypatch):
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    wrapper_a = CodexAgentWrapper(name="codex-a", app_context=context, release_threads=False)
    wrapper_b = CodexAgentWrapper(name="codex-b", app_context=context, release_threads=False)

    class FakeCodex:
        """Resume an existing external thread."""

        async def thread_resume(self, thread_id, **_kwargs):
            return SimpleNamespace(id=thread_id)

    await wrapper_a._open_thread(FakeCodex(), {"session_id": "external-thread", "workspace": "alpha"})
    assert context.workspaces["alpha"].codex_thread_state.owned_thread_ids == {"external-thread"}
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await wrapper_b._open_thread(FakeCodex(), {"session_id": "external-thread", "workspace": "beta"})

    class CompactThread:
        """Minimal native compaction double."""

        id = "compact-thread"

        async def compact(self):
            return None

    compact_codex = SimpleNamespace(thread_resume=lambda _thread_id: None)

    async def resume_compact(_thread_id):
        return CompactThread()

    compact_codex.thread_resume = resume_compact

    async def get_compact_codex():
        return compact_codex

    monkeypatch.setattr(wrapper_a, "_get_codex", get_compact_codex)
    await wrapper_a.compact_session("compact-thread", workspace="alpha")
    assert "compact-thread" in context.workspaces["alpha"].codex_thread_state.owned_thread_ids

    restarted_context = ApplicationContext(
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
    )
    restarted = CodexAgentWrapper(name="codex-restarted", app_context=restarted_context)
    with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
        await restarted._open_thread(FakeCodex(), {"session_id": "external-thread", "workspace": "beta"})


@pytest.mark.asyncio
async def test_codex_concurrent_first_resume_has_one_owner(tmp_path):
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    wrapper_a = CodexAgentWrapper(name="codex-a", app_context=context)
    wrapper_b = CodexAgentWrapper(name="codex-b", app_context=context)

    class FakeCodex:
        """Yield between resume and ownership registration to force interleaving."""

        async def thread_resume(self, thread_id, **_kwargs):
            await asyncio.sleep(0)
            return SimpleNamespace(id=thread_id)

    results = await asyncio.gather(
        wrapper_a._open_thread(FakeCodex(), {"session_id": "shared-external", "workspace": "alpha"}),
        wrapper_b._open_thread(FakeCodex(), {"session_id": "shared-external", "workspace": "beta"}),
        return_exceptions=True,
    )
    assert len([result for result in results if not isinstance(result, Exception)]) == 1
    assert len([result for result in results if isinstance(result, ValueError)]) == 1

    owners = [
        alias
        for alias, workspace in context.workspaces.items()
        if "shared-external" in (workspace.codex_thread_state.owned_thread_ids or set())
    ]
    assert len(owners) == 1
    other = "beta" if owners[0] == "alpha" else "alpha"
    restarted_context = ApplicationContext(
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
    )
    restarted = CodexAgentWrapper(name="codex-restarted", app_context=restarted_context)
    with pytest.raises(ValueError, match="belongs to workspace"):
        await restarted._open_thread(FakeCodex(), {"session_id": "shared-external", "workspace": other})


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
async def test_codex_bridge_uses_resident_mcp_endpoint(tmp_path):
    from reme.components.service import HttpService

    context = _context(tmp_path)
    context.service = HttpService(app_context=context, tool_error_on_failure=True)
    context.jobs = {"search_job": _job(context, "search_job")}
    wrapper = CodexAgentWrapper(app_context=context)

    config = wrapper._mcp_server_config({"job_tools": ["search_job"], "workspace": "beta"})
    assert config["args"][config["args"].index("--endpoint") + 1] == "http://127.0.0.1:2333/mcp"
    raw = config["args"][config["args"].index("--injected-job-kwargs") + 1]
    assert json.loads(raw) == {"workspace": "beta"}
    assert "--config" not in config["args"] and "--workspace" not in config["args"]
    assert "--token" not in config["args"]

    default_config = wrapper._mcp_server_config({"job_tools": ["search_job"]})
    assert "--injected-job-kwargs" not in default_config["args"]
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

    assert context.workspaces["alpha"].tool_contexts["beta/ctx-1"]["drafts"] == ["d-alpha"]
    assert context.workspaces["beta"].tool_contexts["ctx-1"]["drafts"] == ["d-beta"]


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

        next_id = 1

        async def thread_start(self, **kwargs):
            observed.update(kwargs)
            self.next_id += 1
            return SimpleNamespace(id=f"thread-{self.next_id - 1}")

    codex = FakeCodex()
    await wrapper._open_thread(codex, {"workspace": "beta"})
    assert observed["cwd"] == str(tmp_path / "beta")

    observed.clear()
    await wrapper._open_thread(codex, {})
    assert observed["cwd"] != str(tmp_path / "beta")


# ----- Codex bridge: real child-process execution ----------------------------


def _codex_test_app(tmp_path, workspace_dir=None):
    if workspace_dir is None:
        workspace_dir = {"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")}
    return Application(
        workspace_dir=workspace_dir,
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={
            "backend": "http",
            "host": "127.0.0.1",
            "port": 0,
            "web_enabled": False,
            "tool_error_on_failure": True,
        },
        components={
            "tokenizer": {"default": {"backend": "regex"}},
            "file_store": {"default": {"backend": "local", "embedding_store": "", "tag_index": ""}},
            "keyword_index": {"default": {"backend": "bm25"}},
            "file_graph": {"default": {"backend": "local"}},
        },
        jobs={
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
            "read_note": {
                "backend": "base",
                "description": "read a note",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "steps": [{"backend": "read_step"}],
            },
        },
    )


async def _start_resident_http(app):
    import uvicorn

    service = app.context.service
    service.build_service(app)
    service.add_jobs(app)
    config = uvicorn.Config(service.service, host=service.host, port=0, log_level="error", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    assert server.started
    service.port = server.servers[0].sockets[0].getsockname()[1]
    return server, task


async def _stop_resident_http(server, task):
    server.should_exit = True
    await asyncio.wait_for(task, timeout=30)


@pytest.mark.asyncio
async def test_codex_bridge_reads_resident_parent_state(tmp_path):
    """The stdio bridge reuses the resident MCP service and its live workspace state."""
    from fastmcp import Client
    from fastmcp.client import StdioTransport

    app = _codex_test_app(tmp_path)
    wrapper = CodexAgentWrapper(app_context=app.context)
    async with app:
        server, task = await _start_resident_http(app)
        try:
            await app.run_job("write_note", path="daily/resident.md", content="parent-resident-beta", workspace="beta")
            from reme.components.client import MCPClient

            resident_client = MCPClient(transport="streamable-http", host="127.0.0.1", port=app.context.service.port)
            await resident_client.start()
            assert {entry["name"] for entry in await resident_client.list_actions()} == {"read_note", "write_note"}
            await resident_client.close()
            server_config = wrapper._mcp_server_config(
                {"job_tools": ["read_note", "write_note"], "workspace": "beta"},
            )
            transport = StdioTransport(
                command=server_config["command"],
                args=server_config["args"],
                cwd=server_config["cwd"],
                keep_alive=False,
                env=dict(os.environ),
            )
            async with Client(transport, timeout=None, init_timeout=30) as client:
                tools = await client.list_tools()
                assert {tool.name for tool in tools} == {"read_note", "write_note"}
                read = await asyncio.wait_for(client.call_tool("read_note", {"path": "daily/resident.md"}), timeout=30)
                with pytest.raises(Exception):
                    await client.call_tool("read_note", {"path": "daily/resident.md", "workspace": "alpha"})
                await asyncio.wait_for(
                    client.call_tool("write_note", {"path": "daily/bridge.md", "content": "via-bridge-beta"}),
                    timeout=30,
                )
                failure = await client.call_tool(
                    "write_note",
                    {"path": "../escape.md", "content": "must-not-write"},
                    raise_on_error=False,
                )
                assert failure.is_error
            assert not (tmp_path / "escape.md").exists()
            assert "parent-resident-beta" in str(read.content[0].text)
            assert "via-bridge-beta" in (tmp_path / "beta" / "daily" / "bridge.md").read_text(encoding="utf-8")
            assert not (tmp_path / "alpha" / "daily" / "bridge.md").exists()
            store_alpha = app.context.get_component(ComponentEnum.FILE_STORE, "default", "alpha")
            assert all("bridge" not in node.path for node in await store_alpha.get_nodes())

            invalid_config = wrapper._mcp_server_config({"job_tools": ["read_note"], "workspace": "nope"})
            invalid_transport = StdioTransport(
                command=invalid_config["command"],
                args=invalid_config["args"],
                cwd=invalid_config["cwd"],
                keep_alive=False,
                env=dict(os.environ),
            )
            async with Client(invalid_transport, timeout=None, init_timeout=30) as client:
                with pytest.raises(Exception, match="Unknown workspace 'nope'"):
                    await asyncio.wait_for(client.call_tool("read_note", {"path": "daily/resident.md"}), timeout=30)
        finally:
            await _stop_resident_http(server, task)
    await wrapper.close()


@pytest.mark.asyncio
async def test_codex_reply_defaults_workspace_for_real_job_tools(tmp_path, monkeypatch):
    from fastmcp import Client
    from fastmcp.client import StdioTransport

    @dataclass
    class TurnResult:
        """Serializable fake turn result."""

        final_response: str
        usage: object = None

    class FakeThread:
        """Execute the configured job tool through the real STDIO bridge."""

        def __init__(self, thread_id, config):
            self.id = thread_id
            self.config = config

        async def run(self, _inputs, **_kwargs):
            server = next(iter(self.config["mcp_servers"].values()))
            transport = StdioTransport(
                command=server["command"],
                args=server["args"],
                cwd=server["cwd"],
                keep_alive=False,
                env=dict(os.environ),
            )
            arguments = {"path": f"daily/reply-{self.id}.md", "content": self.id}
            if self.id == "thread-override":
                arguments["workspace"] = "beta"
            async with Client(transport, timeout=None, init_timeout=30) as client:
                await client.call_tool("write_note", arguments)
            return TurnResult(final_response="done")

    class FakeCodex:
        """Create threads with deterministic IDs."""

        def __init__(self):
            self.next_id = 0

        async def thread_start(self, **kwargs):
            self.next_id += 1
            thread_id = "thread-override" if self.next_id == 3 else f"thread-{self.next_id}"
            return FakeThread(thread_id, kwargs["config"])

    app = _codex_test_app(tmp_path)
    app.context.jobs["write_note"].kwargs["workspace"] = "beta"
    wrapper = CodexAgentWrapper(app_context=app.context, release_threads=False)
    codex = FakeCodex()

    async def get_codex():
        return codex

    monkeypatch.setattr(wrapper, "_get_codex", get_codex)
    async with app:
        server, task = await _start_resident_http(app)
        try:
            await wrapper.reply("default", job_tools=["write_note"])
            assert (tmp_path / "alpha" / "daily" / "reply-thread-1.md").is_file()
            assert not (tmp_path / "beta" / "daily" / "reply-thread-1.md").exists()

            await wrapper.reply("explicit", job_tools=["write_note"], workspace="alpha")
            assert (tmp_path / "alpha" / "daily" / "reply-thread-2.md").is_file()
            assert not (tmp_path / "beta" / "daily" / "reply-thread-2.md").exists()

            with pytest.raises(Exception):
                await wrapper.reply("override", job_tools=["write_note"])
            assert not (tmp_path / "beta" / "daily" / "reply-thread-override.md").exists()
        finally:
            await _stop_resident_http(server, task)
    await wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("named_alias", [False, True])
async def test_codex_legacy_bridge_preserves_single_workspace_binding(tmp_path, named_alias):
    from fastmcp import Client
    from fastmcp.client import StdioTransport

    alias = "alice" if named_alias else "default"
    root = tmp_path / alias
    workspace_dir = {alias: str(root)} if named_alias else str(root)
    app = _codex_test_app(tmp_path, workspace_dir=workspace_dir)
    wrapper = CodexAgentWrapper(app_context=app.context)
    server_config = wrapper._mcp_server_config({"job_tools": ["write_note"], "workspace": alias})
    transport = StdioTransport(
        command=server_config["command"],
        args=server_config["args"],
        cwd=server_config["cwd"],
        keep_alive=False,
        env=dict(os.environ),
    )

    async with Client(transport, timeout=None, init_timeout=30) as client:
        await asyncio.wait_for(
            client.call_tool("write_note", {"path": "daily/legacy.md", "content": alias}),
            timeout=30,
        )
    assert alias in (root / "daily" / "legacy.md").read_text(encoding="utf-8")
    await wrapper.close()


@pytest.mark.asyncio
async def test_codex_multi_workspace_rejects_unsupported_job_transport(tmp_path):
    from reme.components.service import CliService, HttpService

    context = _context(tmp_path)
    context.service = CliService(app_context=context)
    context.jobs = {"search_job": _job(context, "search_job")}
    wrapper = CodexAgentWrapper(app_context=context)

    with pytest.raises(RuntimeError, match="resident HTTP MCP service"):
        wrapper._mcp_server_config({"job_tools": ["search_job"], "workspace": "beta"})
    with pytest.raises(ValueError, match="only supported for single-workspace"):
        wrapper._mcp_server_config(
            {"job_tools": ["search_job"], "workspace": "beta", "mcp_config": "custom.yaml"},
        )
    context.service = HttpService(app_context=context, tool_error_on_failure=False)
    with pytest.raises(ValueError, match="tool_error_on_failure=true"):
        wrapper._mcp_server_config({"job_tools": ["search_job"], "workspace": "beta"})
    await wrapper.close()
