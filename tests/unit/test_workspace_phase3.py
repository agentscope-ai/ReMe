"""Phase 3: nested propagation, session namespacing, and background binding."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
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

from .codex_workspace_test_support import workspace_server


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
            "codex_app_server": {"default": {"backend": "sdk", "auth_mode": "oauth"}},
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


# ----- MCP boundary ----------------------------------------------------------


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


# ----- Shared Codex server with real workspace jobs ---------------------------


def _codex_test_app(tmp_path, workspace_dir=None):
    if workspace_dir is None:
        workspace_dir = {"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")}
    return Application(
        workspace_dir=workspace_dir,
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components={
            "codex_app_server": {"default": {"backend": "sdk", "auth_mode": "oauth"}},
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


@pytest.mark.asyncio
async def test_codex_shared_server_preserves_workspace_guards_and_cwd(tmp_path, monkeypatch):
    context = _context(tmp_path)
    server = workspace_server(context, monkeypatch)
    first = CodexAgentWrapper(name="first", app_context=context)
    second = CodexAgentWrapper(name="second", app_context=context)
    try:
        alpha = await first.reply("alpha", tool_context_id="ctx-alpha")
        beta = await second.reply("beta", workspace="beta", tool_context_id="ctx-beta")
        sdk = server._codex
        assert alpha["session_id"] != beta["session_id"]
        assert sdk.opened[0][2]["cwd"] == str(first.project_path)
        assert sdk.opened[1][2]["cwd"] == str(tmp_path / "beta")
        assert sdk.threads[beta["session_id"]].options[0]["cwd"] == str(tmp_path / "beta")
        with pytest.raises(ValueError, match="cannot change"):
            await second.reply("changed", resume=alpha["session_id"], tool_context_id="ctx-beta")
        for operation in ("resume", "fork", "compact"):
            with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
                if operation == "compact":
                    await second.compact_session(alpha["session_id"], workspace="beta")
                else:
                    await second.reply(
                        "wrong workspace",
                        resume=alpha["session_id"],
                        workspace="beta",
                        fork_session=operation == "fork",
                    )
        assert context.workspaces["alpha"].codex_thread_state.owned_thread_ids == {alpha["session_id"]}
        assert context.workspaces["beta"].codex_thread_state.owned_thread_ids == {beta["session_id"]}
    finally:
        await first.close()
        await second.close()
        await server.close()


@pytest.mark.asyncio
async def test_codex_ownership_survives_shared_server_and_context_restart(tmp_path, monkeypatch):
    context = _context(tmp_path)
    server = workspace_server(context, monkeypatch)
    wrapper = CodexAgentWrapper(app_context=context)
    try:
        await wrapper.reply("external", resume="external", workspace="alpha")
        await wrapper.compact_session("compact", workspace="alpha")
    finally:
        await wrapper.close()
        await server.close()
    restarted = _context(tmp_path)
    server = workspace_server(restarted, monkeypatch)
    wrapper = CodexAgentWrapper(app_context=restarted)
    try:
        for thread_id in ("external", "compact"):
            with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
                await wrapper.reply("wrong workspace", resume=thread_id, workspace="beta")
        assert server._codex is None
        assert (await wrapper.reply("correct", resume="external", workspace="alpha"))["result"] == "correct"
    finally:
        await wrapper.close()
        await server.close()


@pytest.mark.asyncio
async def test_codex_concurrent_first_resume_has_one_persisted_owner(tmp_path, monkeypatch):
    context = _context(tmp_path)
    server = workspace_server(context, monkeypatch)
    wrappers = [CodexAgentWrapper(name=name, app_context=context) for name in ("first", "second")]
    try:
        results = await asyncio.gather(
            wrappers[0].reply("alpha", resume="T", workspace="alpha"),
            wrappers[1].reply("beta", resume="T", workspace="beta"),
            return_exceptions=True,
        )
        assert sum(isinstance(result, dict) for result in results) == 1
        assert sum(isinstance(result, ValueError) for result in results) == 1
        owners = [
            alias for alias, ws in context.workspaces.items() if "T" in (ws.codex_thread_state.owned_thread_ids or ())
        ]
        assert len(owners) == 1
        assert len(server._codex.opened) == 1
        other = "beta" if owners[0] == "alpha" else "alpha"
        restarted = _context(tmp_path)
        new_server = workspace_server(restarted, monkeypatch)
        new_wrapper = CodexAgentWrapper(app_context=restarted)
        try:
            with pytest.raises(ValueError, match="belongs to workspace"):
                await new_wrapper.reply("wrong", resume="T", workspace=other)
            assert new_server._codex is None
        finally:
            await new_wrapper.close()
            await new_server.close()
    finally:
        for wrapper in wrappers:
            await wrapper.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["multi", "single_string", "single_mapping"])
@pytest.mark.parametrize("explicit", [False, True])
async def test_codex_scoped_mcp_calls_original_workspace_jobs(tmp_path, monkeypatch, mode, explicit):
    """Managed HTTP tools preserve binding, parent state and failures for CLI hosts too."""
    from fastmcp import Client
    from fastmcp.client import StreamableHttpTransport

    from .codex_workspace_test_support import WorkspaceCodex

    monkeypatch.setattr("openai_codex.AsyncCodex", WorkspaceCodex)
    workspace_dir = (
        None
        if mode == "multi"
        else (str(tmp_path / "single") if mode == "single_string" else {"named": str(tmp_path / "single")})
    )
    app = _codex_test_app(tmp_path, workspace_dir=workspace_dir)
    async with app:
        server = app.context.shared_components[ComponentEnum.CODEX_APP_SERVER]["default"]
        wrapper = CodexAgentWrapper(app_context=app.context)
        alias = "beta" if mode == "multi" and explicit else next(iter(app.context.workspaces))
        root = app.context.workspaces[alias].workspace_path
        await app.run_job("write_note", path="daily/resident.md", content="parent-state", workspace=alias)
        sdk = await server._get_client()
        seen = []

        async def tools_from_turn(_thread, _inputs):
            config = sdk.opened[-1][2]["config"]["mcp_servers"]["reme_jobs"]
            transport = StreamableHttpTransport(url=config["url"], headers=config["http_headers"])
            async with Client(transport) as client:
                assert {tool.name for tool in await client.list_tools()} == {"read_note", "write_note"}
                read = await client.call_tool("read_note", {"path": "daily/resident.md"})
                assert "parent-state" in read.content[0].text
                with pytest.raises(Exception):
                    await client.call_tool("read_note", {"path": "daily/resident.md", "workspace": "other"})
                await client.call_tool("write_note", {"path": "daily/tool.md", "content": "bound-write"})
                failure = await client.call_tool(
                    "write_note",
                    {"path": "../escape.md", "content": "must-not-write"},
                    raise_on_error=False,
                )
                assert failure.is_error
                seen.append(read)

        sdk.run_hook = tools_from_turn
        try:
            kwargs = {"workspace": alias} if explicit else {}
            await wrapper.reply("use tools", job_tools=["read_note", "write_note"], **kwargs)
            assert seen
            assert (root / "daily/tool.md").read_text(encoding="utf-8") == "bound-write\n"
            assert not (root.parent / "escape.md").exists()
            for other_alias, workspace in app.context.workspaces.items():
                if other_alias != alias:
                    assert not (workspace.workspace_path / "daily/tool.md").exists()
                    assert not await workspace.get_component(ComponentEnum.FILE_STORE, "default").get_nodes()
        finally:
            await wrapper.close()


@pytest.mark.asyncio
async def test_codex_session_history_and_appends_follow_workspace_binding(tmp_path, monkeypatch):
    """New session endpoints cannot read or inject another workspace's thread."""
    from unittest.mock import AsyncMock

    from reme.components.codex_app_server import CodexAppServer
    from reme.steps.common.session import SessionAppendStep, SessionReadStep, SessionStatusStep

    from .test_agent_session_api import NativeSession

    context = _context(tmp_path)
    server = CodexAppServer(app_context=context)
    context.shared_components[ComponentEnum.CODEX_APP_SERVER] = {"default": server}
    native = NativeSession(server)
    monkeypatch.setattr(server, "_get_client", AsyncMock(return_value=native.client))
    wrapper = CodexAgentWrapper(app_context=context)
    configured_beta = CodexAgentWrapper(name="beta", app_context=context, workspace="beta")
    session_id = native.thread.id
    try:
        for step_type in (SessionAppendStep, SessionReadStep, SessionStatusStep):
            step = step_type(app_context=context, agent_wrapper=wrapper)
            runtime = RuntimeContext(
                workspace=context.workspaces["beta"],
                session_id=session_id,
                content="beta proactive message",
                message_id="beta-message",
            )
            await step(runtime)
            assert runtime.response.success
        beta_path = server._message_path(session_id, "beta")
        assert beta_path.is_relative_to(tmp_path / "beta") and beta_path.is_file()
        assert not server._message_path(session_id, "alpha").exists()
        assert (await configured_beta.read_session(session_id)).session_id == session_id
        fork_path = server.session_path / "sessions" / "fork.jsonl"
        fork_path.write_text("")
        fork_thread = SimpleNamespace(
            id="fork-beta",
            path=str(fork_path),
            turns=native.thread.turns,
            status=native.thread.status,
        )
        native.threads[fork_thread.id] = fork_thread
        fork_handle = SimpleNamespace(
            id=fork_thread.id,
            read=AsyncMock(return_value=SimpleNamespace(thread=fork_thread)),
        )
        native.client.thread_fork = AsyncMock(return_value=fork_handle)
        async with server.thread(session_id, fork=True, workspace="beta") as forked:
            assert forked.id == "fork-beta"
        fork_snapshot = await server.read_session("fork-beta", workspace="beta")
        assert any(message.content == "beta proactive message" for message in fork_snapshot.messages)
        assert server._message_path("fork-beta", "beta").is_file()
        assert not server._message_path("fork-beta", "alpha").exists()
        reads = native.client._client.thread_read.await_count
        for operation in ("read_session", "get_session_status", "append_session"):
            kwargs = {"content": "wrong", "message_id": "wrong"} if operation == "append_session" else {}
            with pytest.raises(ValueError, match="belongs to workspace 'beta'"):
                await getattr(wrapper, operation)(session_id, **kwargs)
        assert native.client._client.thread_read.await_count == reads
        assert len(native.injected) == 1
    finally:
        await wrapper.close()
        await configured_beta.close()
        await server.close()
