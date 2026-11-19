"""Agent runtime wiring, durable notification delivery and shared Chat ownership."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import json
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.components import LocalAgentRuntime, ApplicationContext, R, RuntimeContext
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.agent_inbox import AgentEventInbox
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import SessionSnapshot, SessionStatus, StreamChunk
from reme.schema.proactive_agent import (
    CreateProactiveAgentConfig,
    ManageProactiveAgentRequest,
    ProactiveAgentConfig,
    ProactiveRunResult,
)
from reme.steps.proactive.agent import (
    AgentChatStep,
    AgentNewSessionStep,
    AgentReminderStep,
    AgentSessionStep,
    ChatAgentStep,
    InboxChatStep,
    ListChatSessionsStep,
    NewChatSessionStep,
    ProactiveAgentStep,
    ProactiveQueryStep,
)
from reme.steps.proactive.tools.inbox import ReadInboxStep, SendChatMessageStep
from reme.steps.proactive.tools.manage_proactive_agent import ManageProactiveAgentStep


class FakeWrapper(BaseAgentWrapper):
    """Keep native sessions and inference entirely inside the test workspace."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.created = []
        self.turns = []

    async def create_session(self, **kwargs):
        self.created.append(kwargs)
        return f"session-{len(self.created)}"

    async def session_exists(self, session_id):
        return session_id in {f"session-{index + 1}" for index in range(len(self.created))}

    async def get_session_status(self, session_id):
        return SessionStatus(session_id=session_id, status="idle")

    async def read_session(self, session_id):
        return SessionSnapshot(session_id=session_id, state=await self.get_session_status(session_id))

    @asynccontextmanager
    async def session_guard(self, session_id, *, expected_revision):
        if (await self.read_session(session_id)).revision != expected_revision:
            raise RuntimeError("Session changed before committing decision")
        yield

    async def reply_stream(self, inputs, **kwargs):
        self.turns.append((inputs, kwargs))
        result = {
            "session_id": kwargs["resume"],
            "result": "Processed",
            "structured_output": None,
            "usage": None,
            "status": "completed",
            "error": None,
            "raw": {},
        }
        yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="Processed")
        yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, chunk=result, status="completed")


@asynccontextmanager
async def running_runtime(tmp_path, *, watch=False, agents=None, chat=None, language="en"):
    context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai", language=language)
    wrapper = FakeWrapper(app_context=context)
    inbox = AgentEventInbox(app_context=context)
    runtime = LocalAgentRuntime(app_context=context, agents=agents, chat=chat)
    context.components = {
        "agent_wrapper": {"default": wrapper},
        "agent_inbox": {"default": inbox},
        "agent_runtime": {"default": runtime, "selected": runtime},
    }
    async with wrapper, inbox, runtime:
        if not watch:
            runtime.watcher.cancel()
            await asyncio.gather(runtime.watcher, return_exceptions=True)
            runtime.watcher = None
        yield runtime, context, wrapper


async def create_agent(runtime):
    await runtime.create(
        {"agent_name": "monitor", "task_description": "Monitor experiments"},
    )
    await runtime.append(
        "monitor",
        {"id": "event-1", "event_ts": runtime.now(), "source": "experiment", "description": "Finished"},
    )


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_runtime_initializes_chat_memory_and_preserves_it_across_turns_and_restart(
    tmp_path,
    monkeypatch,
    language,
):
    initial = "Confirmed initial interests\n\nKeep this formatting.\n"
    async with running_runtime(tmp_path, chat={"initial_memory": initial}, language=language) as (
        runtime,
        context,
        wrapper,
    ):
        memory = runtime.agents.path("Remy") / "MEMORY.md"
        assert memory.read_text() == initial
        assert not wrapper.turns
        assert "initial_memory" not in json.loads((memory.parent / "config.json").read_text())
        content = "# Personal memory\nUpdated interests and an explicit forgetting request\n"
        runtime.files.write(memory, content)
        before = memory.stat().st_mtime_ns
        write = runtime.files.write

        def reject_turn_initialization(path, *args, **kwargs):
            if path == memory:
                raise AssertionError("Chat turns must not initialize or overwrite personal memory")
            return write(path, *args, **kwargs)

        monkeypatch.setattr(runtime.files, "write", reject_turn_initialization)
        await ChatAgentStep(app_context=context)(RuntimeContext(stream_queue=asyncio.Queue()), query="Hello")
        await NewChatSessionStep(app_context=context)()
        assert memory.read_text() == content and memory.stat().st_mtime_ns == before
    async with running_runtime(tmp_path, chat={"initial_memory": "Changed startup text"}, language=language) as (
        runtime,
        _,
        _,
    ):
        assert (runtime.agents.path("Remy") / "MEMORY.md").read_text() == content


@pytest.mark.parametrize("chat", [{"initial_memory": None}, {"initial_memory": 42}, {"unknown": "text"}, []])
def test_initial_chat_memory_configuration_is_validated(tmp_path, chat):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    with pytest.raises(ValueError):
        LocalAgentRuntime(app_context=context, chat=chat)
    assert not (tmp_path / "reme_agents").exists()


def completed_result(runtime, agent_name="monitor"):
    detail = "daily/experiment.md"
    runtime.files.write(runtime.files.path(detail), "# Experiment finished\n")
    path = runtime.agents.path(agent_name) / "reminder" / f"{runtime.now():%Y-%m-%d}" / f"{agent_name}_090000.md"
    runtime.files.write(path, "---\naction: P0\n---\nExperiment finished\n")
    reminder = path.relative_to(runtime.files.workspace).as_posix()
    return (
        ProactiveRunResult.model_validate(
            {"notifications": [{"priority": "P0", "message": "Experiment finished", "paths": [reminder, detail]}]},
        ),
        "session-1",
    )


def inbox_path(runtime, read=False):
    """Locate the receiver path where the Runtime must never write inbox logs."""
    return runtime.agents.path("Remy") / "inbox" / ("inbox_read.jsonl" if read else "inbox_unread.jsonl")


def inbox_history(runtime):
    """Inspect canonical receipts independently of the fork-only read tool."""
    outbox = runtime.outbox
    refs = sorted(outbox.committed, key=lambda ref: (outbox.records[ref].created_at, ref.id))
    return [{**outbox.records[ref].model_dump(mode="json"), "from_agent_name": outbox.owners[ref]} for ref in refs]


@pytest.mark.parametrize("name,backend", [("proactive", "codex"), ("proactive_cc", "claude_code")])
def test_configuration_registers_runtime_and_agent_steps_without_legacy_aliases(name, backend):
    config = resolve_app_config(config=name, log_config=False)
    components = config["components"]
    wrapper = components["agent_wrapper"]["default"]
    assert components["agent_runtime"]["default"]["agent_wrapper"] == "default"
    assert wrapper["backend"] == backend
    assert wrapper["sandbox"] == "workspace-write"
    if backend == "claude_code":
        assert wrapper["anthropic_relay"] == "default"
        assert components["anthropic_relay"]["default"]["backend"] == "bailian"
        overrides = components["codex_app_server"]["default"]["config_overrides"]
        env = wrapper["env"]
        assert f"model_context_window={env['CLAUDE_CODE_MAX_CONTEXT_TOKENS']}" in overrides
        assert f"model_auto_compact_token_limit={int(env['CLAUDE_CODE_AUTO_COMPACT_WINDOW']) - 33_000}" in overrides
        assert {"tools", "allowed_tools", "disallowed_tools"}.isdisjoint(wrapper)
        assert json.loads(wrapper["settings"])["permissions"]["blockWritesOutsideWorkingDirectories"] is True
    assert "agent_runtime" in config["components"] and "proactive" not in config["components"]
    assert R.get(ComponentEnum.AGENT_RUNTIME, "local") is LocalAgentRuntime
    assert config["components"]["agent_inbox"]["default"]["backend"] == "event"
    assert R.get(ComponentEnum.AGENT_INBOX, "event") is AgentEventInbox
    assert R.get("proactive", "local") is None
    assert R.get(ComponentEnum.STEP, "chat_agent_step") is ChatAgentStep
    assert R.get(ComponentEnum.STEP, "proactive_agent_step") is ProactiveAgentStep
    assert R.get(ComponentEnum.STEP, "proactive_chat_step") is None
    for job_name, job in config["jobs"].items():
        if job_name in {"finance_news_cron", "stock_quotes_cron", "btc_quotes_cron"}:
            continue
        for step in job.get("steps", []):
            assert R.get(ComponentEnum.STEP, step["backend"]) is not None
    assert not config["jobs"]["run_proactive_agent"]["enable_serve"]
    assert not config["jobs"]["process_chat_reminders"]["enable_serve"]
    assert config["jobs"]["proactive_loop"]["steps"] == [{"backend": "agent_loop_step"}]
    assert "agent_loop" not in config["jobs"]
    for job_name in ("inbox_loop", "agent_event_list", "agent_mailbox"):
        assert job_name not in config["jobs"]
        assert R.get(ComponentEnum.STEP, f"{job_name}_step") is None
    assert config["jobs"]["read_chat_reminders"]["parameters"]["properties"] == {}
    schema = CreateProactiveAgentConfig.model_json_schema()
    properties = config["jobs"]["manage_proactive_agent"]["parameters"]["properties"]
    assert properties.keys() == ManageProactiveAgentRequest.model_json_schema()["properties"].keys()
    assert (
        not {"name", "agent_id", "instructions", "description", "trigger_type", "config", "query", "max_batch_events"}
        & properties.keys()
    )
    assert {"agent_name", "task_description"} <= set(schema["required"])
    assert "agent_type" not in schema["properties"] and "temporary" not in schema["properties"]
    assert "type" not in schema["properties"]
    assert "task_description" not in ProactiveAgentConfig.model_json_schema()["properties"]


@pytest.mark.parametrize("agent_type", ["cron", "proactive"])
def test_agent_configuration_rejects_legacy_type(agent_type):
    values = {"name": "monitor", "description": "Monitor", "type": agent_type}
    if agent_type == "cron":
        values["cron"] = "0 * * * *"
    with pytest.raises(ValueError, match="Extra inputs"):
        ProactiveAgentConfig(**values)


@pytest.mark.parametrize("agent_type", ["cron", "proactive"])
@pytest.mark.asyncio
async def test_formal_agents_inherit_web_search_from_shared_codex_server(tmp_path, agent_type):
    config = resolve_app_config(config="proactive", log_config=False)
    overrides = config["components"]["codex_app_server"]["default"]["config_overrides"]
    assert {"project_doc_max_bytes=0", "sandbox_workspace_write.network_access=false", 'web_search="live"'} <= set(
        overrides,
    )
    async with running_runtime(tmp_path) as (runtime, context, _):
        values = {"agent_name": "monitor", "task_description": "Monitor"}
        if agent_type == "cron":
            values["cron"] = "0 * * * *"
        await runtime.create(values)
        agent = await runtime.definition("monitor")
        step_config = config["jobs"]["run_proactive_agent"]["steps"][0]
        step = ProactiveAgentStep(app_context=context, **step_config)
        options = step.session_options(agent)
        assert options["job_tools"] == []
        assert "read_only_paths" not in options
        assert "sandbox" not in options  # The configured run inherits the selected wrapper's sandbox.
        assert options["approval_mode"] == "deny_all"
        assert options["config"] == {"skills.include_instructions": False}
        chat_tools = config["jobs"]["ask_agent"]["steps"][0]["job_tools"]
        chat = ChatAgentStep(app_context=context, job_tools=chat_tools).session_options()
        assert chat["job_tools"] == ["manage_proactive_agent", "todo", "idea"]
        assert "sandbox" not in chat
        assert "config" not in chat


def test_network_policy_has_no_per_agent_override():
    values = {"agent_name": "monitor"}
    for field in ("enable_web", "web_search_enable"):
        with pytest.raises(ValueError, match="Extra inputs"):
            ProactiveAgentConfig(**values, **{field: True})
    config = resolve_app_config(config="proactive", log_config=False)
    jobs = config["jobs"]
    assert "web_extract" not in jobs
    assert "enable_web" not in jobs["manage_proactive_agent"]["parameters"]["properties"]
    assert "agent_memory" not in jobs
    assert R.get(ComponentEnum.STEP, "agent_memory_step") is None


@pytest.mark.asyncio
async def test_chat_tool_changes_fork_existing_history_and_then_reuse_session(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        previous = await runtime.sessions.ensure(
            "Remy",
            wrapper,
            {"job_tools": ["manage_proactive_agent", "web_extract"]},
            runtime.now(),
        )
        tools = resolve_app_config(config="proactive", log_config=False)["jobs"]["ask_agent"]["steps"][0]["job_tools"]
        for _ in range(2):
            await ChatAgentStep(app_context=context, job_tools=tools)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                query="Hello",
            )
        assert len(wrapper.created) == 2
        assert wrapper.created[-1]["resume"] == previous and wrapper.created[-1]["fork_session"]
        options = wrapper.turns[-1][1]
        assert options["job_tools"] == tools
        assert "config" not in options
        assert (await runtime.sessions.list("Remy"))["total"] == 2


@pytest.mark.asyncio
async def test_chat_tools_are_selected_only_from_step_configuration(tmp_path):
    async with running_runtime(tmp_path) as (_, context, wrapper):
        for tools in ([], ["manage_proactive_agent"]):
            kwargs = {"job_tools": tools} if tools else {}
            await ChatAgentStep(app_context=context, **kwargs)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                query="Hello",
            )
            assert wrapper.turns[-1][1]["job_tools"] == tools


@pytest.mark.asyncio
async def test_trigger_updates_create_fresh_sessions(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        tool = ManageProactiveAgentStep(app_context=context)
        previous = None
        for agent_type in ("proactive", "cron", "proactive"):
            await tool(
                action="update",
                agent_name="monitor",
                cron="0 * * * *" if agent_type == "cron" else None,
            )
            agent = await runtime.definition("monitor")
            assert agent.agent_type == agent_type
            step = ProactiveAgentStep(app_context=context)
            now = runtime.now()
            current = await runtime.sessions.ensure(
                "monitor",
                wrapper,
                step.session_options(agent),
                now,
                new=True,
            )
            if previous is not None:
                assert current != previous
            assert not {"resume", "fork_session"} & wrapper.created[-1].keys()
            previous = current
        saved = json.loads((runtime.agents.path("monitor") / "config.json").read_text())
        assert "agent_type" not in saved and saved["cron"] is None


@pytest.mark.asyncio
async def test_named_runtime_reference_and_durable_definition_layout(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        step = ManageProactiveAgentStep(app_context=context, agent_runtime="selected")
        response = await step(
            action="create",
            agent_name="monitor",
            task_description="Monitor",
        )
        assert response.success and response.answer["agent_name"] == "monitor"
        assert runtime.agents.path("monitor") == tmp_path / "reme_agents/monitor"
        assert runtime.agent_wrapper is wrapper
        assert runtime.inbox is context.components["agent_inbox"]["default"]
        assert (runtime.agents.path("monitor") / "config.json").is_file()
        saved = json.loads((runtime.agents.path("monitor") / "config.json").read_text())
        assert "agent_type" not in saved and "type" not in saved
        assert "task_description" not in saved
        assert not list(tmp_path.rglob("state.json"))
        assert not list(tmp_path.rglob("agents.json"))
        assert not wrapper.turns
    assert runtime.closed and not runtime.is_started
    assert not runtime.inbox.is_started and not wrapper.is_started


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_management_exposes_task_path_and_action_specific_settings(tmp_path, language):
    async with running_runtime(tmp_path, language=language) as (_, context, _):
        tool = ManageProactiveAgentStep(app_context=context)
        response = await tool(
            action="create",
            agent_name="monitor",
            task_description="Only report failures",
        )
        created = response.answer
        path = tmp_path / created["task_path"]
        assert path == tmp_path / "reme_agents" / "monitor" / "TASK.md"
        assert created["agent_name"] == "monitor" and created["task_path"] == path.relative_to(tmp_path).as_posix()
        assert path.read_text() == "Only report failures"
        path.write_text("Only report completion\n")
        status = (await tool(action="status", agent_name="monitor")).answer
        assert status["task_path"] == created["task_path"]
        assert status["cron"] is None and "temporary" not in status
        assert "task_description" not in status
        listing = (await tool(action="list")).answer
        assert listing == {"items": [status], "total": 1}


@pytest.mark.asyncio
async def test_list_and_status_return_task_paths_without_loading_or_changing_contents(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "TASK.md"
        content = "实验记录\n" * 1000
        runtime.files.write(path, content)
        read = runtime.files.read

        def reject_task_read(candidate):
            assert candidate != path
            return read(candidate)

        monkeypatch.setattr(runtime.files, "read", reject_task_read)
        status = await runtime.status("monitor")
        assert status["task_path"] == path.relative_to(tmp_path).as_posix()
        assert "task_description" not in status
        listing = (await ManageProactiveAgentStep(app_context=context)(action="list")).answer
        assert listing["items"][0]["task_path"] == status["task_path"]
        assert path.read_text() == content


@pytest.mark.asyncio
async def test_update_changes_execution_settings_without_replacing_task_description(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "TASK.md"
        runtime.files.write(path, "Only record results\n\nExperiment 42\n")
        before = path.read_bytes(), path.stat().st_mtime_ns
        status = await runtime.update("monitor", {"min_trigger_interval_seconds": 30, "enabled": False})
        assert status["config"]["min_trigger_interval_seconds"] == 30
        assert not status["config"]["enabled"]
        updated = runtime.agents.update("monitor", {"notify_window": {"start": "09:00", "end": "10:00"}}, runtime.now())
        assert updated.notify_window.allows(runtime.now().replace(hour=9, minute=30))
        assert "task_description" not in status
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
        config = (path.parent / "config.json").read_bytes()
        with pytest.raises(ValueError, match="task_description"):
            await runtime.update("monitor", {"task_description": "Restore the old task"})
        assert (path.parent / "config.json").read_bytes() == config
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.asyncio
async def test_configuration_updates_do_not_require_or_recreate_task_description(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        task_file = runtime.agents.path("monitor") / "TASK.md"
        task_file.unlink()
        updated = await runtime.update("monitor", {"enabled": False})
        assert not updated["config"]["enabled"]
        assert updated["task_path"] == "reme_agents/monitor/TASK.md"
        assert "task_description" not in updated
        assert not task_file.exists()
        assert (await runtime.list_agents())["total"] == 1


@pytest.mark.asyncio
async def test_round_executor_receives_only_current_agent_and_event_batch(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        batches = []

        async def execute(agent, events):
            assert agent.agent_name == "monitor"
            batches.append(events)
            return ProactiveRunResult(notifications=[]), "session-1"

        result = await runtime.run("monitor", execute)
        assert result["processed"] == 1
        assert len(batches) == 1 and batches[0][0]["id"] == "event-1"
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 0


@pytest.mark.asyncio
async def test_restoring_a_deleted_agent_retains_edited_task_description(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "TASK.md"
        content = "Only report completion\n\nConfirmed result\n"
        runtime.files.write(path, content)
        await runtime.delete("monitor")
        restored = await runtime.create(
            {"agent_name": "monitor", "task_description": "Initial text"},
        )
        assert restored["restored"]
        assert path.read_text() == content


@pytest.mark.asyncio
async def test_initial_agents_use_creation_once_and_preserve_updates_and_deletions_on_restart(tmp_path):
    initial = [
        {"agent_name": "monitor", "task_description": "Initial monitoring task"},
        {"agent_name": "timer", "task_description": "Initial digest task", "cron": "0 8 * * *"},
    ]
    async with running_runtime(tmp_path, agents=initial) as (runtime, _, _):
        path = runtime.agents.path("monitor") / "TASK.md"
        assert path.read_text() == "Initial monitoring task"
        assert (runtime.agents.path("timer") / "TASK.md").read_text() == "Initial digest task"
        runtime.files.write(path, "Updated by the user\n")
        await runtime.update("monitor", {"min_trigger_interval_seconds": 60, "enabled": False})
        await runtime.delete("timer")
    async with running_runtime(tmp_path, agents=initial) as (runtime, _, _):
        status = await runtime.status("monitor")
        assert (tmp_path / status["task_path"]).read_text() == "Updated by the user\n"
        assert status["config"]["min_trigger_interval_seconds"] == 60
        assert not status["config"]["enabled"]
        assert (await runtime.list_agents())["total"] == 1
        assert runtime.agents.get("timer", include_deleted=True).deleted_at is not None


@pytest.mark.asyncio
async def test_failed_initial_task_write_never_publishes_an_agent(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        write = runtime.files.write

        def fail_task_write(path, *args, **kwargs):
            if path.name == "TASK.md":
                raise OSError("Task write failed")
            return write(path, *args, **kwargs)

        monkeypatch.setattr(runtime.files, "write", fail_task_write)
        with pytest.raises(OSError, match="Task write failed"):
            await runtime.create({"agent_name": "monitor", "task_description": "Monitor"})
        assert (await runtime.list_agents())["total"] == 0


@pytest.mark.parametrize("content", [None, "", "  \n"])
@pytest.mark.asyncio
async def test_missing_or_blank_task_description_keeps_inputs_pending(tmp_path, monkeypatch, content):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "TASK.md"
        if content is None:
            path.unlink()
        else:
            runtime.files.write(path, content)
        model = AsyncMock()
        monkeypatch.setattr(wrapper, "reply_stream", model)
        with pytest.raises((FileNotFoundError, ValueError)):
            await ProactiveAgentStep(app_context=context)(agent_name="monitor")
        model.assert_not_called()
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1
        if content is None:
            assert not path.exists()
        else:
            assert path.read_text() == content


@pytest.mark.asyncio
async def test_status_refuses_task_symlinks_and_respects_tool_permissions(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        tool = ManageProactiveAgentStep(app_context=context)
        with pytest.raises(PermissionError, match="agent runtime files"):
            await tool(RuntimeContext(_allowed_paths=["daily"]), action="status", agent_name="monitor")
        task_file = runtime.agents.path("monitor") / "TASK.md"
        task_file.unlink()
        outside = tmp_path.parent / f"{tmp_path.name}-task.md"
        outside.write_text("Private content")
        task_file.symlink_to(outside)
        with pytest.raises(ValueError, match="without symlinks"):
            await tool(action="status", agent_name="monitor")


@pytest.mark.parametrize(
    "agents",
    [
        {},
        [{"agent_name": "monitor", "task_description": "   "}],
        [{"agent_name": "timer", "task_description": "Report", "cron": "invalid"}],
        [
            {"agent_name": "monitor", "task_description": "Monitor"},
            {"agent_name": "MONITOR", "task_description": "Other"},
        ],
    ],
)
def test_initial_agent_configuration_rejects_invalid_tasks(tmp_path, agents):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    with pytest.raises(ValueError):
        LocalAgentRuntime(app_context=context, agents=agents)
    assert not (tmp_path / "reme_agents").exists()


@pytest.mark.asyncio
async def test_inbox_reads_frontmatter_across_dates_and_excludes_uncommitted_reminders(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await create_agent(runtime)
        await runtime.create({"agent_name": "other", "task_description": "Other task"})
        now = runtime.now()
        yesterday = now - timedelta(days=1)
        directory = runtime.agents.path("monitor") / "reminder"
        old_path = directory / f"{yesterday.date()}" / "old.md"
        new_path = directory / f"{now.date()}" / "new.md"
        uncommitted = directory / f"{now.date()}" / "uncommitted.md"
        other_path = runtime.agents.path("other") / "reminder" / f"{now.date()}" / "other.md"
        for path, identifier, created_at, delivered in (
            (old_path, "old", yesterday, True),
            (new_path, "new", now, True),
            (uncommitted, "uncommitted", now + timedelta(seconds=1), False),
            (other_path, "other", now + timedelta(seconds=2), True),
        ):
            runtime.files.write(
                path,
                frontmatter.dumps(
                    frontmatter.Post(
                        identifier,
                        id=identifier,
                        created_at=created_at.isoformat(),
                        action="P1",
                        paths=[path.relative_to(tmp_path).as_posix(), "daily/result.md"],
                        delivered_at=created_at.isoformat() if delivered else None,
                        read_at=now.isoformat() if identifier == "old" else None,
                    ),
                ),
            )
        contents = {path: path.read_bytes() for path in (old_path, new_path, uncommitted, other_path)}

        await runtime.outbox.sync()
        unread = (await runtime.notifications.read())["items"]
        assert [item["id"] for item in unread] == ["new", "other"]
        history = inbox_history(runtime)
        assert [item["id"] for item in history] == ["old", "new", "other"]
        assert history[0]["read_at"] and history[1]["read_at"] is None
        assert history[1]["delivered_at"] and history[1]["from_agent_name"] == "monitor"
        assert history[2]["from_agent_name"] == "other"
        assert inbox_history(runtime)[0]["id"] == "old"
        document = frontmatter.loads(runtime.files.read(new_path))
        document["read_at"] = now.isoformat()
        runtime.files.write(new_path, frontmatter.dumps(document))
        contents[new_path] = new_path.read_bytes()
        await runtime.outbox.sync()
        assert [item["id"] for item in (await runtime.notifications.read())["items"]] == ["other"]
        history = inbox_history(runtime)
        assert next(item for item in history if item["id"] == "new")["read_at"]
        assert contents == {path: path.read_bytes() for path in contents}
        assert set(runtime.notifications.pending) == {"other"} and not wrapper.turns


@pytest.mark.asyncio
async def test_committed_outbox_recovers_delivery_without_repeating_execution(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        execute = AsyncMock(return_value=completed_result(runtime))
        await runtime.run("monitor", execute)
        monkeypatch.setattr(runtime.outbox, "update", AsyncMock(side_effect=OSError("interrupted delivery")))
        with pytest.raises(OSError, match="interrupted delivery"):
            await runtime.outbox.sync()
        assert execute.await_count == 1
        assert (await runtime.status("monitor"))["pending_events"] == 0

    async with running_runtime(tmp_path) as (restored, _, _):
        execute = AsyncMock(side_effect=AssertionError("committed work must not repeat"))
        assert (await restored.run("monitor", execute))["processed"] == 0
        execute.assert_not_awaited()
        await restored.outbox.sync()
        items = (await restored.notifications.read())["items"]
        assert len(items) == 1
        record = items[0]
        assert record["from_agent_name"] == "monitor"
        await restored.outbox.sync()
        assert len((await restored.notifications.read())["items"]) == 1
        await restored.notifications.consume([record["id"]])
        read_at = inbox_history(restored)[0]["read_at"]
        await restored.notifications.consume([record["id"]])
        assert not (await restored.notifications.read())["items"]
        assert inbox_history(restored)[0]["read_at"] == read_at

    async with running_runtime(tmp_path) as (restored, _, _):
        assert not restored.notifications.pending
        assert len(inbox_history(restored)) == 1


@pytest.mark.asyncio
async def test_failed_consumption_confirmation_retries_uncommitted_work(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        monkeypatch.setattr(runtime.inbox, "ack", AsyncMock(side_effect=OSError("interrupted ack")))
        with pytest.raises(OSError, match="interrupted ack"):
            await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        assert not (await runtime.notifications.read())["items"]
        assert (await runtime.status("monitor"))["pending_events"] == 1

    async with running_runtime(tmp_path) as (runtime, _, _):
        execute = AsyncMock(return_value=completed_result(runtime))
        await runtime.run("monitor", execute)
        execute.assert_awaited_once()
        await runtime.outbox.sync()
        assert len((await runtime.notifications.read())["items"]) == 1


@pytest.mark.asyncio
async def test_chat_rotation_preserves_inbox_and_wakeup_uses_current_session(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        listing = ListChatSessionsStep(app_context=context)
        assert (await listing()).answer["total"] == 0
        assert not wrapper.created
        chat = ChatAgentStep(app_context=context)
        response = await chat(RuntimeContext(stream_queue=asyncio.Queue()), query="Hello")
        assert response.success and response.answer == "Processed"
        first = (await runtime.sessions.current("Remy")).session_id
        await create_agent(runtime)
        execute = AsyncMock(return_value=completed_result(runtime))
        await runtime.run("monitor", execute)
        await runtime.outbox.sync()
        identifier = (await runtime.notifications.read())["items"][0]["id"]
        assert (await NewChatSessionStep(app_context=context)()).success
        current = (await runtime.sessions.current("Remy")).session_id
        assert current != first and (await listing()).answer["total"] == 2
        assert (await runtime.notifications.read())["items"][0]["id"] == identifier

        async def run_job(name, **kwargs):
            assert name == "process_chat_reminders"
            return await InboxChatStep(app_context=context)(**kwargs)

        async def decide(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            await SendChatMessageStep(app_context=context)(content="", **bound)
            assert options["resume"] == current and options["ephemeral"]
            return {"status": "completed", "result": "Internal final reply"}

        wrapper.reply = AsyncMock(side_effect=decide)
        await runtime.notifications.tick(run_job)
        await runtime.notifications.worker
        assert identifier not in runtime.notifications.pending
        assert runtime.notifications.retry_at is None
        await runtime.notifications.tick(run_job)
        assert runtime.notifications.worker is None
        wrapper.reply.assert_awaited_once()
        assert len(wrapper.turns) == 1


@pytest.mark.asyncio
async def test_agent_tools_preserve_receiver_identity_and_workspace_permissions(tmp_path):
    async with running_runtime(tmp_path) as (_, context, _):
        tool = ReadInboxStep(app_context=context)
        with pytest.raises(ValueError, match="calling Chat Agent"):
            await tool(_agent_name="other")
        with pytest.raises(PermissionError, match="agent runtime files"):
            await tool(RuntimeContext(_allowed_paths=["daily"]), _agent_name="Remy")
        with pytest.raises(ValueError, match="active reminder fork"):
            await tool(_agent_name="Remy")


@pytest.mark.asyncio
async def test_agent_directories_and_reference_only_inbox(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        await runtime.create(
            {"agent_name": "experiment", "task_description": "Observe"},
        )
        await runtime.create(
            {
                "agent_name": "timer",
                "task_description": "Report",
                "cron": "* * * * *",
            },
        )
        assert runtime.agents.path("experiment") == tmp_path / "reme_agents/experiment"
        assert runtime.agents.path("timer") == tmp_path / "reme_agents/timer"
        event_path = next((runtime.agents.path("monitor") / "inbox").glob("inbox_unread.jsonl"))
        assert json.loads(event_path.read_text())["metadata"] == {}
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        refs = [item["event"] for item in (await runtime.list_inputs("Remy"))["items"]]
        assert not inbox_path(runtime).exists()
        assert len(refs) == 1 and set(refs[0]) == {"id", "event_ts", "source", "description", "paths", "metadata"}
        assert refs[0]["paths"][0].startswith("reme_agents/monitor/reminder/")
        assert refs[0]["source"] == "monitor" and "message" not in refs[0]
        receipt = (await runtime.inbox.snapshot("monitor"))["receipt"]
        assert receipt.metadata["reminders"] == [{"id": refs[0]["id"], "file": refs[0]["paths"][0]}]
        await ChatAgentStep(app_context=context)(RuntimeContext(stream_queue=asyncio.Queue()), query="Hello")
        assert (tmp_path / "reme_agents/Remy/MEMORY.md").is_file()
        assert (tmp_path / "reme_agents/Remy/sessions.jsonl").is_file()
        assert not (tmp_path / "session").exists()
        assert not (tmp_path / "digest").exists()


@pytest.mark.asyncio
async def test_task_names_and_directory_identity_are_validated(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        for name in ("Remy", "remy", "../escape"):
            with pytest.raises(ValueError):
                await runtime.create({"agent_name": name, "task_description": "Observe"})
        await runtime.create({"agent_name": "tmp", "task_description": "Observe"})
        assert runtime.agents.path("tmp") == tmp_path / "reme_agents" / "tmp"
        await create_agent(runtime)
        with pytest.raises(ValueError, match="already exists"):
            await runtime.create(
                {"agent_name": "MONITOR", "task_description": "Other"},
            )
        with pytest.raises(ValueError):
            await runtime.update("monitor", {"temporary": True})
        with pytest.raises(ValueError):
            await runtime.update("monitor", {"created_at": runtime.now()})
        config = runtime.agents.path("monitor") / "config.json"
        value = json.loads(config.read_text())
        value["agent_name"] = "another"
        config.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="directory"):
            await runtime.definition("monitor")


@pytest.mark.asyncio
async def test_unhanded_off_results_protect_read_history_and_survive_restart(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        runtime.inbox.max_read_items = 1
        runtime.inbox.append_timeout_seconds = 0.02
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        second = {"id": "event-2", "event_ts": runtime.now(), "source": "lab", "description": "Next"}
        await runtime.append("monitor", second)
        batch = await runtime.inbox.claim("monitor")
        with pytest.raises(TimeoutError, match="handoffs"):
            await runtime.inbox.ack("monitor", batch["token"])
        await runtime.inbox.nack("monitor", batch["token"])
        await runtime.outbox.sync()
        assert len(list((runtime.agents.path("monitor") / "inbox").glob("inbox_unread.jsonl"))) == 1
    async with running_runtime(tmp_path) as (runtime, _, _):
        await runtime.outbox.sync()
        assert len((await runtime.notifications.read())["items"]) == 1
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.asyncio
async def test_read_view_retention_does_not_resurrect_old_messages(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        now = runtime.now()
        directory = runtime.agents.path("monitor") / "reminder" / f"{now.date()}"
        for index in range(1001):
            path = directory / f"msg-{index:04}.md"
            runtime.files.write(
                path,
                frontmatter.dumps(
                    frontmatter.Post(
                        "Done",
                        id=f"msg-{index:04}",
                        created_at=now.isoformat(),
                        action="P2",
                        paths=[path.relative_to(tmp_path).as_posix(), "daily/result.md"],
                        delivered_at=now.isoformat(),
                        read_at=now.isoformat(),
                    ),
                ),
            )
        await runtime.outbox.sync()
        assert len(runtime.notifications.read_refs) == 1000
        assert "msg-0000" not in runtime.notifications.read_refs
        assert not inbox_path(runtime, True).exists()
        await runtime.outbox.sync()
        assert len(runtime.notifications.read_refs) == 1000
        assert not runtime.notifications.pending


@pytest.mark.asyncio
async def test_external_frontmatter_updates_and_corrupt_reminders_fail_closed(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.notifications.pending.values()))
        path = runtime.files.path(ref.file)
        document = frontmatter.loads(runtime.files.read(path))
        document["user_read_at"] = runtime.now().isoformat()
        runtime.files.write(path, frontmatter.dumps(document))
        await runtime.outbox.sync()
        assert runtime.outbox.records[ref].user_read_at is not None
        document["action"] = "invalid"
        invalid = frontmatter.dumps(document)
        runtime.files.write(path, invalid)
        await runtime.outbox.sync()
        assert path.read_text() == invalid
        assert ref.file in runtime.outbox.issues
        assert ref.id not in runtime.notifications.pending
        assert (await runtime.list_views())["items"][1]["error"]
        document["action"] = "P0"
        runtime.files.write(path, frontmatter.dumps(document))
        await runtime.outbox.sync()
        assert not runtime.outbox.issues
        assert ref.id in runtime.notifications.pending


@pytest.mark.asyncio
async def test_failed_markdown_state_write_preserves_body_and_retries(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.notifications.pending.values()))
        path = runtime.files.path(ref.file)
        original = path.read_bytes()
        from reme.utils import workspace_files as files

        def failed_sync(_descriptor):
            raise OSError("disk unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(files, "sync_file", failed_sync)
            with pytest.raises(OSError, match="disk unavailable"):
                await runtime.notifications.consume([ref.id])
        assert path.read_bytes() == original
        assert ref.id in runtime.notifications.pending
        await runtime.notifications.consume([ref.id])
        document = frontmatter.loads(path.read_text())
        assert document.content == "Experiment finished" and document["read_at"]


@pytest.mark.parametrize("relative", ["daily/experiment.md", "reme_agents/other/reminder/2026-11-15/other.md"])
@pytest.mark.asyncio
async def test_notification_cannot_overwrite_a_file_outside_its_reminder_directory(tmp_path, relative):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        result, session_id = completed_result(runtime)
        result.notifications[0].paths[0] = relative
        path = runtime.files.path(relative)
        runtime.files.write(path, "# Original\n")
        with pytest.raises(ValueError, match="first notification path"):
            await runtime.run("monitor", AsyncMock(return_value=(result, session_id)))
        assert path.read_text() == "# Original\n"
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.asyncio
async def test_cron_restarts_from_future_and_keeps_queued_triggers(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await runtime.create(
            {"agent_name": "timer", "task_description": "Report", "cron": "* * * * *"},
        )
        agent = await runtime.definition("timer")
        start = runtime.now().replace(second=0, microsecond=0)
        monkeypatch.setattr(runtime, "now", lambda: start)
        await runtime._schedule_cron(agent)
        first = runtime.cron["timer"][1]
        assert first == start + timedelta(minutes=1)
        monkeypatch.setattr(runtime, "now", lambda: first)
        await runtime._schedule_cron(agent)
        await runtime._schedule_cron(agent)
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 1
    async with running_runtime(tmp_path) as (runtime, _, _):
        assert not runtime.cron
        future = first + timedelta(days=1)
        monkeypatch.setattr(runtime, "now", lambda: future)
        await runtime._schedule_cron(await runtime.definition("timer"))
        assert runtime.cron["timer"][1] > future
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_shutdown_cancels_owned_execution_and_leaves_input_pending(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        entered = asyncio.Event()

        async def execute(*_args):
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(runtime.run("monitor", execute))
        await entered.wait()
        await runtime.close()
        assert task.cancelled()
        assert not runtime.workers
    async with running_runtime(tmp_path) as (runtime, _, _):
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.parametrize("operation", ["run", "query"])
@pytest.mark.asyncio
async def test_shutdown_cancels_agent_tasks_waiting_for_locks(tmp_path, operation):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        lock = runtime.management_lock if operation == "query" else runtime.execution("monitor").lock
        execute = AsyncMock(return_value=(ProactiveRunResult(notifications=[]), "session-1"))

        async def invoke():
            if operation == "run":
                await runtime.run("monitor", execute)
            else:
                async with runtime.query_session("monitor", {}):
                    await execute()

        async with lock:
            task = asyncio.create_task(invoke())
            try:
                await asyncio.sleep(0)
                await runtime.close()
                assert task.cancelled()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        execute.assert_not_called()
    async with running_runtime(tmp_path) as (runtime, _, _):
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_agent_directory_symlinks_are_rejected(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "reme_agents").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="without symlinks"):
        async with running_runtime(tmp_path):
            pass


@pytest.mark.asyncio
async def test_query_forks_latest_session_without_selecting_the_temporary_branch(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        source = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now())
        result, _ = completed_result(runtime)
        await runtime.run("monitor", AsyncMock(return_value=(result, source)))
        current = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now(), new=True)
        response = await ProactiveQueryStep(app_context=context)(agent_name="monitor", query="What happened?")
        assert response.success and response.answer["answer"] == "Processed"
        assert wrapper.created[-1]["resume"] == current
        assert wrapper.created[-1]["fork_session"]
        assert wrapper.created[-1]["sandbox"] == "read-only"
        assert wrapper.created[-1]["job_tools"] == []
        assert "config" not in wrapper.created[-1]
        assert (await runtime.sessions.current("monitor")).session_id == current
        records = (await runtime.sessions.list("monitor"))["items"]
        assert records[0]["temporary"]
        assert wrapper.turns[-1][0] == "What happened?"


@pytest.mark.asyncio
async def test_query_registration_owns_requests_during_checkpoint_selection(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        start = runtime.now()
        await runtime.create(
            {"agent_name": "experiment", "task_description": "Observe"},
        )
        await runtime.append("experiment", {"event_ts": start, "source": "lab", "description": "Finished"})
        source = await runtime.sessions.ensure("experiment", wrapper, {}, start)
        await runtime.run("experiment", AsyncMock(return_value=(ProactiveRunResult(notifications=[]), source)))
        directory = runtime.agents.path("experiment")
        monkeypatch.setattr(runtime, "now", lambda: start + timedelta(days=8))
        current = runtime.sessions.current
        checkpoint_entered, checkpoint_release = asyncio.Event(), asyncio.Event()
        model_entered, model_release = asyncio.Event(), asyncio.Event()

        async def checkpoint(agent_name):
            result = await current(agent_name)
            if asyncio.current_task() is querying:
                checkpoint_entered.set()
                await checkpoint_release.wait()
            return result

        async def answer(_query, **kwargs):
            model_entered.set()
            await model_release.wait()
            yield StreamChunk(
                chunk_type=ChunkEnum.REPLY_END,
                status="completed",
                chunk={
                    "status": "completed",
                    "result": "Finished",
                    "session_id": kwargs["resume"],
                    "structured_output": None,
                    "usage": None,
                    "error": None,
                    "raw": {},
                },
            )

        monkeypatch.setattr(runtime.sessions, "current", checkpoint)
        monkeypatch.setattr(wrapper, "reply_stream", answer)
        querying = asyncio.create_task(
            ProactiveQueryStep(app_context=context)(agent_name="experiment", query="Status?"),
        )
        collecting = None
        try:
            await asyncio.wait_for(checkpoint_entered.wait(), 2)
            assert runtime.agent_tasks.get(querying) == "experiment"
            collecting = asyncio.create_task(runtime.tick(AsyncMock()))
            checkpoint_release.set()
            await asyncio.wait_for(model_entered.wait(), 2)
            await asyncio.wait_for(collecting, 2)
            assert directory.is_dir()
            model_release.set()
            assert (await asyncio.wait_for(querying, 2)).success
            assert not runtime.agent_tasks
        finally:
            checkpoint_release.set()
            model_release.set()
            for task in (querying, collecting):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(task for task in (querying, collecting) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_query_without_a_queryable_session_releases_ownership(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        with pytest.raises(ValueError, match="no queryable session"):
            await ProactiveQueryStep(app_context=context)(agent_name="monitor", query="Status?")
        assert not runtime.agent_tasks and not wrapper.created


@pytest.mark.parametrize("failure", ["fork", "record", "reply"])
@pytest.mark.asyncio
async def test_failed_query_releases_only_its_fork_and_preserves_primary(tmp_path, monkeypatch, failure):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        source = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now())
        release = AsyncMock()
        monkeypatch.setattr(wrapper, "release_session", release)
        if failure == "fork":
            monkeypatch.setattr(wrapper, "create_session", AsyncMock(side_effect=RuntimeError("fork failed")))
        elif failure == "record":
            monkeypatch.setattr(runtime.sessions, "record", AsyncMock(side_effect=RuntimeError("record failed")))
        else:

            async def fail(*_args, **_kwargs):
                raise RuntimeError("reply failed")
                yield  # pylint: disable=unreachable

            monkeypatch.setattr(wrapper, "reply_stream", fail)
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            await ProactiveQueryStep(app_context=context)(agent_name="monitor", query="Progress?")
        assert (await runtime.sessions.current("monitor")).session_id == source
        assert not runtime.agent_tasks
        if failure == "fork":
            release.assert_not_awaited()
        else:
            release.assert_awaited_once_with("session-2")
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_cancelled_query_finishes_fork_registration_and_cleanup(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        source = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now())
        create = wrapper.create_session
        entered, proceed = asyncio.Event(), asyncio.Event()

        async def fork(**options):
            entered.set()
            await proceed.wait()
            return await create(**options)

        wrapper.create_session = fork
        wrapper.release_session = AsyncMock()
        querying = asyncio.create_task(ProactiveQueryStep(app_context=context)(agent_name="monitor", query="Status?"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            querying.cancel()
            proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(querying, 2)
            assert (await runtime.sessions.current("monitor")).session_id == source
            assert (await runtime.sessions.list("monitor"))["items"][0]["temporary"]
            wrapper.release_session.assert_awaited_once_with("session-2")
            assert not wrapper.turns and not runtime.agent_tasks
        finally:
            proceed.set()
            querying.cancel()
            await asyncio.gather(querying, return_exceptions=True)


@pytest.mark.asyncio
async def test_session_registry_recovers_an_incomplete_tail_without_losing_complete_history(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        original = await runtime.sessions.ensure("Remy", wrapper, {}, runtime.now())
        path = runtime.sessions.path("Remy")
        complete = path.read_bytes()
        path.write_bytes(complete + b'{"session_id":"interrupted')
        assert (await runtime.sessions.current("Remy")).session_id == original
        assert (await runtime.sessions.list("Remy"))["total"] == 1
        assert path.read_bytes() == complete + b'{"session_id":"interrupted'
        await runtime.sessions.record("Remy", "recovered-query", runtime.now(), temporary=True)
        assert path.read_bytes().startswith(complete)
        assert (await runtime.sessions.list("Remy"))["total"] == 2
        assert (await runtime.sessions.current("Remy")).session_id == original
        with path.open("ab") as handle:
            handle.write(b"{invalid}\n")
        with pytest.raises(json.JSONDecodeError):
            await runtime.sessions.list("Remy")


@pytest.mark.parametrize("directory", ["", ".", "..", "/tmp/agents", "../agents", "~/agents", "a//b", "C:\\agents"])
def test_agent_root_must_be_a_normalized_workspace_subdirectory(tmp_path, directory):
    with pytest.raises(ValueError):
        ApplicationContext(workspace_dir=str(tmp_path), reme_agent_dir=directory)


@pytest.mark.asyncio
async def test_custom_root_and_unicode_identity_share_the_same_layout(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path), reme_agent_dir="runtime/agents")
    wrapper = FakeWrapper(app_context=context)
    inbox = AgentEventInbox(app_context=context)
    runtime = LocalAgentRuntime(app_context=context)
    context.components = {"agent_wrapper": {"default": wrapper}, "agent_inbox": {"default": inbox}}
    async with wrapper, inbox, runtime:
        created = await runtime.create(
            {"agent_name": "屏幕观察", "task_description": "Observe screens"},
        )
        await runtime.append("屏幕观察", {"event_ts": runtime.now(), "source": "screen", "description": "New"})
        path = tmp_path / "runtime/agents/屏幕观察"
        assert (path / "config.json").is_file()
        assert created["task_path"] == "runtime/agents/屏幕观察/TASK.md"
        assert (path / "TASK.md").read_text() == "Observe screens"
        assert list((path / "inbox").glob("inbox_unread.jsonl"))


@pytest.mark.asyncio
async def test_desktop_views_are_cold_and_include_main_and_task_files(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await create_agent(runtime)
        runtime.files.write(runtime.files.path("resource/input.json"), '{"value": 1}\n')
        await runtime.append(
            "monitor",
            {"event_ts": runtime.now(), "source": "test", "description": "Input", "paths": ["resource/input.json"]},
        )
        views = (await runtime.list_views())["items"]
        assert [view["agent_name"] for view in views] == ["Remy", "monitor"]
        assert views[0]["memory_path"] == "reme_agents/Remy/MEMORY.md"
        assert views[0]["task_path"] is None
        assert views[1]["task_path"] == "reme_agents/monitor/TASK.md"
        assert views[1]["memory_path"] is None
        assert all(view["session_id"] is None for view in views)
        assert "files" not in views[1] and "config" not in views[1]
        assert (await runtime.list_inputs("monitor"))["items"][0]["event"]["paths"] == ["resource/input.json"]
        assert views[1]["pending_events"] == 2
        assert not wrapper.created
        value = await runtime.read_agent("monitor")
        assert value["agent"]["task_path"] == views[1]["task_path"]
        assert value["messages"] == []


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_task_chat_forks_latest_primary_for_each_question_without_consuming_events(tmp_path, language):
    async with running_runtime(tmp_path, language=language) as (runtime, context, wrapper):
        await create_agent(runtime)
        source = await runtime.sessions.ensure(
            "monitor",
            wrapper,
            {"system_prompt": "Full task context"},
            runtime.now(),
        )
        for query in ["Hello", "Continue"]:
            response = await AgentChatStep(app_context=context)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                agent_name="monitor",
                query=query,
            )
            assert response.success
        assert len(wrapper.created) == 3
        assert all(options["resume"] == source and options["fork_session"] for options in wrapper.created[1:])
        assert all(options["sandbox"] == "read-only" for options in wrapper.created[1:])
        assert wrapper.turns[0][1]["resume"] != wrapper.turns[1][1]["resume"]
        assert (await runtime.status("monitor"))["pending_events"] == 1
        assert not list((runtime.agents.path("monitor") / "reminder").rglob("*.md"))
        assert (await runtime.sessions.current("monitor")).session_id == source
        assert not (await runtime.sessions.current("monitor")).temporary


@pytest.mark.asyncio
async def test_query_allows_background_work_and_is_cancelled_on_delete(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await create_agent(runtime)
        await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now())
        release = AsyncMock()
        wrapper.release_session = release
        started = asyncio.Event()

        async def chat():
            async with runtime.query_session("monitor", {}):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(chat())
        await started.wait()
        try:
            assert (await runtime.list_views())["items"][1]["status"] == "idle"
            execute = AsyncMock(return_value=(ProactiveRunResult(notifications=[]), "session-1"))
            assert (await runtime.run("monitor", execute))["processed"] == 1
            assert not task.done()
            await runtime.delete("monitor")
            assert task.cancelled()
            assert not runtime.agent_tasks
            release.assert_any_await("session-2")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_task_questions_and_session_reset_do_not_create_empty_task_sessions(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        with pytest.raises(ValueError, match="no queryable session"):
            await AgentChatStep(app_context=context)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                agent_name="monitor",
                query="Status?",
            )
        with pytest.raises(ValueError, match="accepts no arguments"):
            await AgentNewSessionStep(app_context=context)(agent_name="monitor")
        assert not wrapper.created
        response = await AgentNewSessionStep(app_context=context)()
        assert response.answer["agent"]["agent_name"] == "Remy"
        assert len(wrapper.created) == 1


@pytest.mark.asyncio
async def test_agent_subscription_reconnects_from_snapshot_and_observes_new_session(tmp_path):
    from reme.components.agent_runtime._internal.views import AgentViews

    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        wrapper.read_session = AsyncMock(
            side_effect=lambda sid: SessionSnapshot(session_id=sid, state=SessionStatus(session_id=sid, status="idle")),
        )
        events = AgentViews(runtime).events("Remy", interval=0, heartbeat=0)
        try:
            first = await anext(events)
            assert first["agent"]["session_id"] is None and first["type"] == "snapshot"
            session_id = await runtime.sessions.ensure("Remy", wrapper, {}, runtime.now())
            changed = await anext(events)
            assert changed["type"] == "snapshot" and changed["agent"]["session_id"] == session_id
            await runtime.sessions.record("Remy", "temporary", runtime.now(), temporary=True)
            assert (await runtime.read_agent("Remy"))["agent"]["session_id"] == session_id
            assert (await anext(events))["type"] == "heartbeat"
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_agent_endpoints_reject_unknown_fields_and_enforce_workspace_permissions(tmp_path):
    async with running_runtime(tmp_path) as (_, context, wrapper):
        with pytest.raises(ValueError):
            await AgentChatStep(app_context=context)(query="hi", agent_name="../chat")
        with pytest.raises(ValueError):
            await AgentSessionStep(app_context=context)(agent_name="Remy", session_id="foreign")
        with pytest.raises(PermissionError):
            await AgentSessionStep(app_context=context)(RuntimeContext(_allowed_paths=["daily"]), agent_name="Remy")
        assert not wrapper.created


@pytest.mark.asyncio
async def test_desktop_reminder_read_does_not_consume_agent_inbox(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        tool = AgentReminderStep(app_context=context)
        page = await runtime.list_inputs("Remy", state="pending")
        assert len(page["items"]) == 1
        item = page["items"][0]["event"]
        response = await tool(message_id=item["id"], mark_read=True)
        assert response.answer["user_read_at"]
        assert (await tool(message_id=item["id"])).answer["user_read_at"]
        assert (await runtime.list_inputs("Remy", state="pending"))["total"] == 1
        ref = next(iter(runtime.outbox.committed))
        assert runtime.outbox.records[ref].read_at is None
        assert ref.id in runtime.notifications.pending


@pytest.mark.asyncio
async def test_desktop_retains_empty_session_reference_when_native_thread_is_not_loaded(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await runtime.sessions.record("Remy", "empty-before-restart", runtime.now())
        wrapper.read_session = AsyncMock(side_effect=AssertionError("missing native thread must not be read"))
        value = await runtime.read_agent("Remy")
        assert value["agent"]["session_id"] == "empty-before-restart"
        assert value["messages"] == []
        assert value["agent"]["status"] == "not_loaded"
        assert not wrapper.created


@pytest.mark.asyncio
async def test_reminder_message_provenance_survives_history_projection(tmp_path):
    from reme.components.agent_runtime._internal.views import AgentViews
    from reme.schema import SessionMessage

    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        sid = await runtime.sessions.ensure("Remy", wrapper, {}, runtime.now())
        snapshot = SessionSnapshot(
            session_id=sid,
            state=SessionStatus(session_id=sid, status="idle"),
            messages=[
                SessionMessage(id="reply", role="assistant", content="Normal reply"),
                SessionMessage(id="reminder", role="assistant", content="A useful update", source="proactive"),
            ],
        )
        wrapper.read_session = AsyncMock(return_value=snapshot)
        value = await runtime.read_agent("Remy")
        assert [message["source"] for message in value["messages"]] == ["chat", "proactive"]
        assert not (runtime.agents.path("Remy") / "turns.jsonl").exists()
        events = AgentViews(runtime).events("Remy", interval=0)
        try:
            event = await anext(events)
            assert [message["notification_eligible"] for message in event["messages"]] == [False, True]
            snapshot.messages[1].completed = False
            assert not (await anext(events))["messages"][-1]["notification_eligible"]
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_first_automatic_reply_in_new_session_is_an_atomic_snapshot(tmp_path):
    from reme.components.agent_runtime._internal.views import AgentViews
    from reme.schema import SessionMessage

    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        events = AgentViews(runtime).events("Remy", interval=0)
        try:
            assert (await anext(events))["messages"] == []
            sid = await runtime.sessions.ensure("Remy", wrapper, {}, runtime.now())
            wrapper.read_session = AsyncMock(
                return_value=SessionSnapshot(
                    session_id=sid,
                    state=SessionStatus(session_id=sid, status="idle"),
                    messages=[SessionMessage(id="first", role="assistant", content="New update", source="proactive")],
                ),
            )
            update = await anext(events)
            assert update["type"] == "snapshot" and update["messages"][-1]["notification_eligible"]
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_task_configuration_rejects_retired_fields_without_rewriting_user_files(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await runtime.create({"agent_name": "experiment", "task_description": "Keep original evidence"})
        path = runtime.agents.path("experiment") / "config.json"
        original = json.loads(path.read_text())
        invalid = {**original, "temporary": True}
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="Extra inputs"):
            runtime.agents.load()
        assert json.loads(path.read_text()) == invalid
