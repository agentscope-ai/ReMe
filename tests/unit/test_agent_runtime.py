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
from reme.components.event_stream import AgentEventStream
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import SessionStatus, StreamChunk
from reme.schema.proactive_agent import CreateProactiveAgentConfig, ProactiveAgentConfig, ProactiveRunResult
from reme.steps.proactive.agent import (
    ChatAgentStep,
    InboxChatStep,
    ListChatSessionsStep,
    NewChatSessionStep,
    ProactiveAgentStep,
    ProactiveQueryStep,
)
from reme.steps.proactive.tools.manage_inbox import ManageInboxStep
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
async def running_runtime(tmp_path, *, watch=False, agents=None, main=None, language="en"):
    context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai", language=language)
    wrapper = FakeWrapper(app_context=context)
    stream = AgentEventStream(app_context=context)
    runtime = LocalAgentRuntime(app_context=context, agents=agents, main=main)
    context.components = {
        "agent_wrapper": {"default": wrapper},
        "event_stream": {"default": stream},
        "agent_runtime": {"default": runtime, "selected": runtime},
    }
    async with wrapper, stream, runtime:
        if not watch:
            runtime.watcher.cancel()
            await asyncio.gather(runtime.watcher, return_exceptions=True)
            runtime.watcher = None
        yield runtime, context, wrapper


async def create_agent(runtime):
    await runtime.create({"name": "monitor", "description": "Monitor experiments", "trigger_type": "event"})
    await runtime.append(
        "monitor",
        {"id": "event-1", "event_ts": runtime.now(), "source": "experiment", "description": "Finished"},
    )


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_runtime_initializes_main_memory_and_preserves_it_across_turns_and_restart(
    tmp_path,
    monkeypatch,
    language,
):
    async with running_runtime(tmp_path, main={"description": "Confirmed initial interests"}, language=language) as (
        runtime,
        context,
        wrapper,
    ):
        memory = runtime.agents.path("main") / "MEMORY.md"
        assert "Confirmed initial interests" in memory.read_text()
        assert not wrapper.turns
        assert "description" not in json.loads((memory.parent / "config.json").read_text())
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
    async with running_runtime(tmp_path, main={"description": "Changed startup text"}, language=language) as (
        runtime,
        _,
        _,
    ):
        assert (runtime.agents.path("main") / "MEMORY.md").read_text() == content


@pytest.mark.parametrize("main", [{"description": None}, {"description": 42}, {"unknown": "text"}, []])
def test_initial_main_memory_configuration_is_validated(tmp_path, main):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    with pytest.raises(ValueError):
        LocalAgentRuntime(app_context=context, main=main)
    assert not (tmp_path / "reme_agents").exists()


def completed_result(runtime, agent_id="monitor"):
    detail = "daily/experiment.md"
    runtime.files.write(runtime.files.path(detail), "# Experiment finished\n")
    path = runtime.agents.path(agent_id) / "reminder" / f"{runtime.now():%Y-%m-%d}" / f"{agent_id}_090000.md"
    runtime.files.write(path, "---\npriority: P0\n---\nExperiment finished\n")
    reminder = path.relative_to(runtime.files.workspace).as_posix()
    return (
        ProactiveRunResult.model_validate(
            {"notifications": [{"priority": "P0", "message": "Experiment finished", "paths": [reminder, detail]}]},
        ),
        "session-1",
    )


def test_configuration_registers_runtime_and_agent_steps_without_legacy_aliases():
    config = resolve_app_config(config="proactive", log_config=False)
    assert "agent_runtime" in config["components"] and "proactive" not in config["components"]
    assert R.get(ComponentEnum.AGENT_RUNTIME, "local") is LocalAgentRuntime
    assert config["components"]["event_stream"]["default"]["backend"] == "agent"
    assert R.get(ComponentEnum.EVENT_STREAM, "agent") is AgentEventStream
    assert R.get("proactive", "local") is None
    assert R.get(ComponentEnum.STEP, "chat_agent_step") is ChatAgentStep
    assert R.get(ComponentEnum.STEP, "proactive_agent_step") is ProactiveAgentStep
    assert R.get(ComponentEnum.STEP, "proactive_chat_step") is None
    for job in config["jobs"].values():
        for step in job.get("steps", []):
            assert R.get(ComponentEnum.STEP, step["backend"]) is not None
    assert not config["jobs"]["proactive_run"]["enable_serve"]
    assert not config["jobs"]["inbox_chat"]["enable_serve"]
    schema = CreateProactiveAgentConfig.model_json_schema()
    properties = config["jobs"]["manage_proactive_agent"]["parameters"]["properties"]["config"]["properties"]
    assert properties.keys() == schema["properties"].keys()
    assert "trigger_type" in schema["required"]
    assert "type" not in schema["properties"]
    assert "description" not in ProactiveAgentConfig.model_json_schema()["properties"]


@pytest.mark.parametrize("trigger_type", ["cron", "event"])
def test_agent_configuration_rejects_legacy_type(trigger_type):
    values = {"name": "monitor", "description": "Monitor", "type": trigger_type}
    if trigger_type == "cron":
        values["cron"] = "0 * * * *"
    with pytest.raises(ValueError, match="Extra inputs"):
        ProactiveAgentConfig(**values)


@pytest.mark.parametrize("trigger_type", ["cron", "event"])
@pytest.mark.asyncio
async def test_trigger_controls_tools_and_network(tmp_path, trigger_type):
    async with running_runtime(tmp_path) as (runtime, context, _):
        values = {"name": "monitor", "description": "Monitor", "trigger_type": trigger_type}
        if trigger_type == "cron":
            values["cron"] = "0 * * * *"
        await runtime.create(values)
        agent = await runtime.definition("monitor")
        options = ProactiveAgentStep(app_context=context)._options(agent)
        online = trigger_type == "cron"
        assert options["job_tools"] == (["web_extract"] if online else [])
        assert options["config"]["web_search"] == ("live" if online else "disabled")
        assert options["config"]["sandbox_workspace_write"]["network_access"] is online


def test_network_policy_has_no_per_agent_override():
    values = {"name": "monitor", "trigger_type": "event"}
    for field in ("enable_web", "web_search_enable"):
        with pytest.raises(ValueError, match="Extra inputs"):
            ProactiveAgentConfig(**values, **{field: True})
    config = resolve_app_config(config="proactive", log_config=False)
    jobs = config["jobs"]
    assert jobs["chat"]["steps"][0]["job_tools"] == ["manage_proactive_agent", "manage_inbox", "web_extract"]
    assert "enable_web" not in jobs["manage_proactive_agent"]["parameters"]["properties"]["config"]["properties"]


@pytest.mark.asyncio
async def test_chat_tool_changes_fork_existing_history_and_then_reuse_session(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        previous = await runtime.sessions.ensure(
            "main",
            wrapper,
            {"job_tools": ["manage_proactive_agent", "manage_inbox"]},
            runtime.now(),
        )
        tools = resolve_app_config(config="proactive", log_config=False)["jobs"]["chat"]["steps"][0]["job_tools"]
        for _ in range(2):
            await ChatAgentStep(app_context=context, job_tools=tools)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                query="Hello",
            )
        assert len(wrapper.created) == 2
        assert wrapper.created[-1]["resume"] == previous and wrapper.created[-1]["fork_session"]
        options = wrapper.turns[-1][1]
        assert options["job_tools"] == tools
        assert options["config"]["web_search"] == "live"
        assert (await runtime.sessions.list("main"))["total"] == 2


@pytest.mark.asyncio
async def test_chat_tools_are_selected_only_from_step_configuration(tmp_path):
    async with running_runtime(tmp_path) as (_, context, wrapper):
        for tools in ([], ["manage_inbox"]):
            kwargs = {"job_tools": tools} if tools else {}
            await ChatAgentStep(app_context=context, **kwargs)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                query="Hello",
            )
            assert wrapper.turns[-1][1]["job_tools"] == tools


@pytest.mark.asyncio
async def test_trigger_updates_fork_sessions_with_the_new_tool_policy(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        tool = ManageProactiveAgentStep(app_context=context)
        previous = None
        for trigger_type in ("event", "cron", "event"):
            await tool(
                action="update",
                agent_id="monitor",
                config={"trigger_type": trigger_type, "cron": "0 * * * *" if trigger_type == "cron" else None},
            )
            agent = await runtime.definition("monitor")
            assert agent.trigger_type == trigger_type
            current = await runtime.sessions.ensure(
                "monitor",
                wrapper,
                ProactiveAgentStep()._options(agent),
                runtime.now(),
            )
            if previous is not None:
                assert current != previous and wrapper.created[-1]["resume"] == previous
            assert wrapper.created[-1]["job_tools"] == (["web_extract"] if trigger_type == "cron" else [])
            previous = current
        saved = json.loads((runtime.agents.path("monitor") / "config.json").read_text())
        assert saved["trigger_type"] == "event" and saved["cron"] is None


@pytest.mark.asyncio
async def test_named_runtime_reference_and_durable_definition_layout(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        step = ManageProactiveAgentStep(app_context=context, agent_runtime="selected")
        response = await step(
            action="create",
            config={"name": "monitor", "description": "Monitor", "trigger_type": "event"},
        )
        assert response.success and response.answer["agent_id"] == "monitor"
        assert runtime.agents.path("monitor") == tmp_path / "reme_agents/monitor"
        assert runtime.agent_wrapper is wrapper
        assert runtime.event_stream is context.components["event_stream"]["default"]
        assert (runtime.agents.path("monitor") / "config.json").is_file()
        saved = json.loads((runtime.agents.path("monitor") / "config.json").read_text())
        assert saved["trigger_type"] == "event" and "type" not in saved
        assert "description" not in saved
        assert not list(tmp_path.rglob("state.json"))
        assert not list(tmp_path.rglob("agents.json"))
        assert not wrapper.turns
    assert runtime.closed and not runtime.is_started
    assert not runtime.event_stream.is_started and not wrapper.is_started


@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.asyncio
async def test_management_exposes_memory_and_settings_without_config_paths(tmp_path, temporary):
    async with running_runtime(tmp_path) as (_, context, _):
        tool = ManageProactiveAgentStep(app_context=context)
        response = await tool(
            action="create",
            config={
                "name": "monitor",
                "description": "Only report failures",
                "trigger_type": "event",
                "temporary": temporary,
            },
        )
        created = response.answer
        path = tmp_path / created["memory_path"]
        assert path == tmp_path / "reme_agents" / ("tmp/monitor" if temporary else "monitor") / "MEMORY.md"
        assert created["config"]["trigger_type"] == "event"
        assert "description" not in created["config"] and "config_path" not in created
        assert created["memory_content"] == path.read_text()
        assert "Only report failures" in created["memory_content"]
        assert not created["memory_truncated"]
        path.write_text("# Current task\nOnly report completion\n")
        status = (await tool(action="status", agent_id="monitor")).answer
        assert status["memory_content"] == path.read_text()
        listing = (await tool(action="list")).answer
        assert listing["total"] == 1
        assert listing["items"][0]["memory_path"] == created["memory_path"]
        assert "memory_content" not in listing["items"][0]
        assert "config_path" not in listing["items"][0]


@pytest.mark.asyncio
async def test_status_marks_long_memory_previews_without_changing_the_file(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "MEMORY.md"
        content = "已确认背景\n" * 1000
        runtime.files.write(path, content)
        status = await runtime.status("monitor")
        assert status["memory_truncated"]
        assert content.startswith(status["memory_content"])
        assert 0 < len(status["memory_content"]) < len(content)
        assert path.read_text() == content


@pytest.mark.asyncio
async def test_update_changes_execution_settings_without_replacing_task_memory(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "MEMORY.md"
        runtime.files.write(path, "# Current task\nOnly record results\n\n# Confirmed background\nExperiment 42\n")
        before = path.read_bytes(), path.stat().st_mtime_ns
        status = await runtime.update("monitor", {"min_trigger_interval_seconds": 30, "enabled": False})
        assert status["config"]["min_trigger_interval_seconds"] == 30
        assert not status["config"]["enabled"]
        assert "memory_content" not in status and "memory_truncated" not in status
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
        config = (path.parent / "config.json").read_bytes()
        with pytest.raises(ValueError, match="description"):
            await runtime.update("monitor", {"description": "Restore the old task"})
        assert (path.parent / "config.json").read_bytes() == config
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.asyncio
async def test_configuration_updates_do_not_require_or_recreate_task_memory(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        memory = runtime.agents.path("monitor") / "MEMORY.md"
        memory.unlink()
        updated = await runtime.update("monitor", {"enabled": False})
        assert not updated["config"]["enabled"]
        assert updated["memory_path"] == "reme_agents/monitor/MEMORY.md"
        assert "memory_content" not in updated and "memory_truncated" not in updated
        assert not memory.exists()
        assert (await runtime.list_agents())["total"] == 1


@pytest.mark.asyncio
async def test_round_executor_receives_only_current_agent_and_event_batch(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        batches = []

        async def execute(agent, events):
            assert agent.agent_id == "monitor"
            batches.append(events)
            return ProactiveRunResult(notifications=[]), "session-1"

        result = await runtime.run("monitor", execute)
        assert result["processed"] == 1
        assert len(batches) == 1 and batches[0][0]["id"] == "event-1"
        assert (await runtime.snapshot("monitor"))["pending_count"] == 0


@pytest.mark.asyncio
async def test_restoring_a_deleted_agent_retains_edited_memory(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "MEMORY.md"
        content = "# Current task\nOnly report completion\n\n# Long-term findings\nConfirmed result\n"
        runtime.files.write(path, content)
        await runtime.delete("monitor")
        restored = await runtime.create({"name": "monitor", "description": "Initial text", "trigger_type": "event"})
        assert restored["memory_content"] == content
        assert path.read_text() == content


@pytest.mark.asyncio
async def test_initial_agents_use_creation_once_and_preserve_updates_and_deletions_on_restart(tmp_path):
    initial = [
        {"name": "monitor", "description": "Initial monitoring task", "trigger_type": "event"},
        {"name": "timer", "description": "Initial digest task", "trigger_type": "cron", "cron": "0 8 * * *"},
    ]
    async with running_runtime(tmp_path, agents=initial) as (runtime, _, _):
        path = runtime.agents.path("monitor") / "MEMORY.md"
        assert "Initial monitoring task" in path.read_text()
        assert "Initial digest task" in (runtime.agents.path("timer") / "MEMORY.md").read_text()
        runtime.files.write(path, "# Current task\nUpdated by the user\n")
        await runtime.update("monitor", {"min_trigger_interval_seconds": 60, "enabled": False})
        await runtime.delete("timer")
    async with running_runtime(tmp_path, agents=initial) as (runtime, _, _):
        status = await runtime.status("monitor")
        assert status["memory_content"] == "# Current task\nUpdated by the user\n"
        assert status["config"]["min_trigger_interval_seconds"] == 60
        assert not status["config"]["enabled"]
        assert (await runtime.list_agents())["total"] == 1
        assert runtime.agents.get("timer", include_deleted=True).deleted_at is not None


@pytest.mark.asyncio
async def test_failed_initial_memory_write_never_publishes_an_agent(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        write = runtime.files.write

        def fail_memory(path, *args, **kwargs):
            if path.name == "MEMORY.md":
                raise OSError("Memory write failed")
            return write(path, *args, **kwargs)

        monkeypatch.setattr(runtime.files, "write", fail_memory)
        with pytest.raises(OSError, match="Memory write failed"):
            await runtime.create({"name": "monitor", "description": "Monitor", "trigger_type": "event"})
        assert (await runtime.list_agents())["total"] == 0


@pytest.mark.parametrize("content", [None, "", "  \n"])
@pytest.mark.asyncio
async def test_missing_or_blank_task_memory_keeps_inputs_pending(tmp_path, monkeypatch, content):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "MEMORY.md"
        if content is None:
            path.unlink()
        else:
            runtime.files.write(path, content)
        model = AsyncMock()
        monkeypatch.setattr(wrapper, "reply", model)
        with pytest.raises((FileNotFoundError, ValueError)):
            await ProactiveAgentStep(app_context=context)(agent_id="monitor")
        model.assert_not_called()
        assert (await runtime.snapshot("monitor"))["pending_count"] == 1
        if content is None:
            assert not path.exists()
        else:
            assert path.read_text() == content


@pytest.mark.asyncio
async def test_status_refuses_memory_symlinks_and_respects_tool_permissions(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        tool = ManageProactiveAgentStep(app_context=context)
        with pytest.raises(PermissionError, match="agent runtime files"):
            await tool(RuntimeContext(_allowed_paths=["daily"]), action="status", agent_id="monitor")
        memory = runtime.agents.path("monitor") / "MEMORY.md"
        memory.unlink()
        outside = tmp_path.parent / f"{tmp_path.name}-memory.md"
        outside.write_text("Private content")
        memory.symlink_to(outside)
        with pytest.raises(ValueError, match="without symlinks"):
            await tool(action="status", agent_id="monitor")


@pytest.mark.parametrize(
    "agents",
    [
        {},
        [{"name": "monitor", "description": "   ", "trigger_type": "event"}],
        [{"name": "timer", "description": "Report", "trigger_type": "cron", "cron": "invalid"}],
        [
            {"name": "monitor", "description": "Monitor", "trigger_type": "event"},
            {"name": "MONITOR", "description": "Other", "trigger_type": "event"},
        ],
    ],
)
def test_initial_agent_configuration_rejects_invalid_tasks(tmp_path, agents):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    with pytest.raises(ValueError):
        LocalAgentRuntime(app_context=context, agents=agents)
    assert not (tmp_path / "reme_agents").exists()


@pytest.mark.asyncio
async def test_proactive_agent_step_saves_artifacts_and_publishes_through_runtime(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        paths = ProactiveAgentStep(app_context=context)._paths(await runtime.definition("monitor"), now)

        async def model_reply(_query, **kwargs):
            assert "output_schema" not in kwargs
            runtime.files.write(paths["summary_path"], "# Experiment finished\n")
            runtime.files.write(paths["detail_path"], "# Experiment results\n")
            runtime.files.write(paths["reminder_path"], "---\npriority: P0\n---\nFinished\n")
            return {
                "status": "completed",
                "session_id": kwargs["resume"],
                "result": "Saved the daily summary and reminder.",
            }

        monkeypatch.setattr(wrapper, "reply", AsyncMock(side_effect=model_reply))
        response = await ProactiveAgentStep(app_context=context)(agent_id="monitor")
        assert response.success and response.answer["processed"] == 1
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        item = (await runtime.inbox.read("main"))["items"][0]
        assert item["message"] == "Finished" and item["priority"] == "P0"
        assert item["reminder_path"] == paths["reminder_path"].relative_to(tmp_path).as_posix()
        document = frontmatter.loads(paths["reminder_path"].read_text())
        assert document.content == "Finished" and document["delivered_at"]
        assert "message" not in document.metadata
        assert not list(paths["reminder_dir"].glob("*.jsonl"))
        assert item["paths"] == [
            paths[key].relative_to(tmp_path).as_posix() for key in ("reminder_path", "detail_path")
        ]
        assert paths["detail_path"].name == paths["reminder_path"].name == f"monitor_{now:%H%M%S}.md"
        assert wrapper.created[0]["sandbox"] == "workspace-write"


@pytest.mark.asyncio
async def test_round_paths_select_history_and_keep_paired_filenames(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        agent = await runtime.definition("monitor")
        step = ProactiveAgentStep(app_context=context)
        now = runtime.now()
        paths = step._paths(agent, now)
        assert paths["history_summary_path"] is None
        for delta in (-5, -2, 1):
            path = tmp_path / "daily" / f"{now + timedelta(days=delta):%Y-%m-%d}" / "monitor.md"
            runtime.files.write(path, "# Summary\n")
        runtime.files.write(tmp_path / "daily/not-a-date/monitor.md", "# Invalid date\n")
        paths = step._paths(agent, now)
        assert paths["history_summary_path"].parent.name == f"{now - timedelta(days=2):%Y-%m-%d}"
        runtime.files.write(paths["summary_path"], "# Today\n")
        runtime.files.write(paths["detail_path"], "# Existing details\n")
        paths = step._paths(agent, now)
        assert paths["history_summary_path"] == paths["summary_path"]
        assert paths["detail_path"].name == paths["reminder_path"].name == f"monitor_{now:%H%M%S}_1.md"
        runtime.files.write(paths["reminder_path"], "---\npriority: P1\n---\nExisting reminder\n")
        paths = step._paths(agent, now)
        assert paths["detail_path"].name == paths["reminder_path"].name == f"monitor_{now:%H%M%S}_2.md"


@pytest.mark.parametrize(
    "failure",
    [
        "missing_summary",
        "unchanged_summary",
        "empty_summary",
        "missing_priority",
        "invalid_priority",
        "empty_reminder",
        "missing_detail",
        "failed_reply",
    ],
)
@pytest.mark.asyncio
async def test_incomplete_round_keeps_events_pending_without_publishing(tmp_path, monkeypatch, failure):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if failure == "unchanged_summary":
            runtime.files.write(paths["summary_path"], "# Previous summary\n")

        async def model_reply(_query, **kwargs):
            if failure not in {"missing_summary", "unchanged_summary"}:
                runtime.files.write(paths["summary_path"], "" if failure == "empty_summary" else "# Summary\n")
            if failure != "missing_detail":
                runtime.files.write(paths["detail_path"], "# Details\n")
            priority = "P9" if failure == "invalid_priority" else "P0"
            header = "" if failure == "missing_priority" else f"---\npriority: {priority}\n---\n"
            body = "" if failure == "empty_reminder" else "Finished\n"
            runtime.files.write(paths["reminder_path"], header + body)
            return {
                "status": "failed" if failure == "failed_reply" else "completed",
                "session_id": kwargs["resume"],
                "error": "Interrupted" if failure == "failed_reply" else None,
                "result": "Done",
            }

        monkeypatch.setattr(wrapper, "reply", AsyncMock(side_effect=model_reply))
        with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
            await step(agent_id="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.inbox.pending


@pytest.mark.asyncio
async def test_round_without_reminder_uses_saved_files_instead_of_response_json(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        paths = ProactiveAgentStep(app_context=context)._paths(await runtime.definition("monitor"), now)

        async def model_reply(_query, **kwargs):
            runtime.files.write(paths["summary_path"], "# No material change\n")
            return {
                "status": "completed",
                "session_id": kwargs["resume"],
                "result": "No reminder needed.",
                "structured_output": {"notifications": [{"priority": "P0", "message": "Ignore response JSON"}]},
            }

        monkeypatch.setattr(wrapper, "reply", AsyncMock(side_effect=model_reply))
        response = await ProactiveAgentStep(app_context=context)(agent_id="monitor")
        assert response.success and response.answer["notifications"] == []
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert not runtime.inbox.pending


@pytest.mark.asyncio
async def test_inbox_reads_frontmatter_across_dates_and_excludes_uncommitted_reminders(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await create_agent(runtime)
        await runtime.create({"name": "other", "description": "Other task", "trigger_type": "event"})
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
                        priority="P1",
                        paths=[path.relative_to(tmp_path).as_posix(), "daily/result.md"],
                        delivered_at=created_at.isoformat() if delivered else None,
                        read_at=now.isoformat() if identifier == "old" else None,
                        woken_at=now.isoformat() if identifier == "new" else None,
                    ),
                ),
            )
        contents = {path: path.read_bytes() for path in (old_path, new_path, uncommitted, other_path)}

        await runtime.outbox.sync()
        unread = (await runtime.inbox.read("main"))["items"]
        assert [item["id"] for item in unread] == ["new", "other"]
        history = (await runtime.inbox.read("main", unread_only=False))["items"]
        assert [item["id"] for item in history] == ["old", "new", "other"]
        assert history[0]["read_at"] and history[1]["read_at"] is None
        assert history[1]["delivered_at"] and history[1]["from_agent_id"] == "monitor"
        assert frontmatter.loads(new_path.read_text())["woken_at"]
        assert history[2]["from_agent_id"] == "other"
        assert (await runtime.inbox.read("main", unread_only=False, limit=1))["items"][0]["id"] == "old"
        document = frontmatter.loads(runtime.files.read(new_path))
        document["read_at"] = now.isoformat()
        runtime.files.write(new_path, frontmatter.dumps(document))
        contents[new_path] = new_path.read_bytes()
        await runtime.outbox.sync()
        assert [item["id"] for item in (await runtime.inbox.read("main"))["items"]] == ["other"]
        history = (await runtime.inbox.read("main", unread_only=False))["items"]
        assert next(item for item in history if item["id"] == "new")["read_at"]
        assert contents == {path: path.read_bytes() for path in contents}
        assert set(runtime.inbox.pending) == {"other"} and not wrapper.turns


@pytest.mark.asyncio
async def test_committed_outbox_recovers_delivery_without_repeating_execution(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        execute = AsyncMock(return_value=completed_result(runtime))
        await runtime.run("monitor", execute)
        monkeypatch.setattr(runtime.inbox, "deliver", AsyncMock(side_effect=OSError("interrupted delivery")))
        with pytest.raises(OSError, match="interrupted delivery"):
            await runtime.outbox.sync()
        assert execute.await_count == 1
        assert (await runtime.status("monitor"))["pending_events"] == 0

    async with running_runtime(tmp_path) as (restored, _, _):
        execute = AsyncMock(side_effect=AssertionError("committed work must not repeat"))
        assert (await restored.run("monitor", execute))["processed"] == 0
        execute.assert_not_awaited()
        await restored.outbox.sync()
        items = (await restored.inbox.read("main"))["items"]
        assert len(items) == 1
        record = items[0]
        assert record["from_agent_id"] == "monitor" and record["to_agent_id"] == "main"
        await restored.outbox.sync()
        assert len((await restored.inbox.read("main"))["items"]) == 1
        await restored.inbox.mark_read("main", [record["id"]])
        read_at = (await restored.inbox.read("main", unread_only=False))["items"][0]["read_at"]
        await restored.inbox.mark_read("main", [record["id"]])
        assert not (await restored.inbox.read("main"))["items"]
        assert (await restored.inbox.read("main", unread_only=False))["items"][0]["read_at"] == read_at

    async with running_runtime(tmp_path) as (restored, _, _):
        assert not restored.inbox.pending
        assert len((await restored.inbox.read("main", unread_only=False))["items"]) == 1


@pytest.mark.asyncio
async def test_failed_consumption_confirmation_retries_uncommitted_work(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        monkeypatch.setattr(runtime.event_stream, "ack", AsyncMock(side_effect=OSError("interrupted ack")))
        with pytest.raises(OSError, match="interrupted ack"):
            await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        assert not (await runtime.inbox.read("main"))["items"]
        assert (await runtime.status("monitor"))["pending_events"] == 1

    async with running_runtime(tmp_path) as (runtime, _, _):
        execute = AsyncMock(return_value=completed_result(runtime))
        await runtime.run("monitor", execute)
        execute.assert_awaited_once()
        await runtime.outbox.sync()
        assert len((await runtime.inbox.read("main"))["items"]) == 1


@pytest.mark.asyncio
async def test_chat_rotation_preserves_inbox_and_wakeup_uses_current_session(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        listing = ListChatSessionsStep(app_context=context)
        assert (await listing()).answer["total"] == 0
        assert not wrapper.created
        chat = ChatAgentStep(app_context=context)
        response = await chat(RuntimeContext(stream_queue=asyncio.Queue()), query="Hello")
        assert response.success and response.answer == "Processed"
        first = (await runtime.sessions.current("main")).session_id
        await create_agent(runtime)
        execute = AsyncMock(return_value=completed_result(runtime))
        await runtime.run("monitor", execute)
        await runtime.outbox.sync()
        identifier = (await runtime.inbox.read("main"))["items"][0]["id"]
        assert (await NewChatSessionStep(app_context=context)()).success
        current = (await runtime.sessions.current("main")).session_id
        assert current != first and (await listing()).answer["total"] == 2
        assert (await runtime.inbox.read("main"))["items"][0]["id"] == identifier

        async def run_job(name, **kwargs):
            assert name == "inbox_chat"
            return await InboxChatStep(app_context=context)(**kwargs)

        await runtime.inbox.tick(run_job, "inbox_chat")
        await runtime.inbox.worker
        assert wrapper.turns[-1][1]["resume"] == current
        assert identifier in runtime.inbox.pending
        assert runtime.outbox.records[runtime.inbox.pending[identifier]].woken_at is not None
        await runtime.inbox.tick(run_job, "inbox_chat")
        assert runtime.inbox.worker is None
        assert len(wrapper.turns) == 2


@pytest.mark.asyncio
async def test_agent_tools_preserve_receiver_identity_and_workspace_permissions(tmp_path):
    async with running_runtime(tmp_path) as (_, context, _):
        tool = ManageInboxStep(app_context=context)
        with pytest.raises(ValueError, match="calling Agent"):
            await tool(action="read", _agent_id="main", agent_id="other")
        with pytest.raises(PermissionError, match="agent runtime files"):
            await tool(RuntimeContext(_allowed_paths=["daily"]), action="read", _agent_id="main")


@pytest.mark.asyncio
async def test_agent_directories_memory_and_reference_only_inbox(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        await runtime.create(
            {"name": "experiment", "description": "Observe", "trigger_type": "event", "temporary": True},
        )
        await runtime.create(
            {"name": "timer", "description": "Report", "trigger_type": "cron", "cron": "* * * * *", "temporary": True},
        )
        assert runtime.agents.path("experiment") == tmp_path / "reme_agents/tmp/experiment"
        assert runtime.agents.path("timer") == tmp_path / "reme_agents/tmp/timer"
        event_path = next((runtime.agents.path("monitor") / "events").glob("events_unread.jsonl"))
        assert json.loads(event_path.read_text())["metadata"]["_received_at"]
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        refs = list(runtime.files.iter_jsonl(runtime.inbox.path()))
        assert len(refs) == 1 and set(refs[0]) == {"file", "id"}
        assert refs[0]["file"].startswith("reme_agents/monitor/reminder/")
        receipt = (await runtime.snapshot("monitor"))["consumption"]
        assert receipt.metadata["reminders"] == refs
        await ChatAgentStep(app_context=context)(RuntimeContext(stream_queue=asyncio.Queue()), query="Hello")
        assert (tmp_path / "reme_agents/main/MEMORY.md").is_file()
        assert (tmp_path / "reme_agents/main/sessions.jsonl").is_file()
        assert not (tmp_path / "session").exists()
        assert not (tmp_path / "digest").exists()


@pytest.mark.asyncio
async def test_external_read_reference_updates_canonical_receipt_and_repairs_views(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.inbox.pending.values()))
        runtime.files.append_jsonl(runtime.inbox.path(True), ref.model_dump())
        await runtime.outbox.sync()
        assert not runtime.inbox.pending
        document = frontmatter.loads(runtime.files.read(runtime.files.path(ref.file)))
        assert ref.file.endswith(".md")
        assert document.content == "Experiment finished"
        assert set(document.metadata) == {
            "id",
            "created_at",
            "priority",
            "paths",
            "delivered_at",
            "read_at",
            "woken_at",
        }
        assert document["read_at"] and document["delivered_at"]
        assert not list((runtime.agents.path("monitor") / "reminder").glob("*.jsonl"))
        runtime.inbox.path().unlink()
        runtime.inbox.path(True).unlink()
        await runtime.outbox.sync()
        assert not runtime.inbox.pending
        assert ref.id in runtime.inbox.read_refs


@pytest.mark.asyncio
async def test_watcher_observes_external_acknowledgements(tmp_path):
    async with running_runtime(tmp_path, watch=True) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        async with asyncio.timeout(5):
            while not runtime.inbox.pending:
                await asyncio.sleep(0.02)
        ref = next(iter(runtime.inbox.pending.values()))
        runtime.files.append_jsonl(runtime.inbox.path(True), ref.model_dump())
        async with asyncio.timeout(5):
            while runtime.inbox.pending:
                await asyncio.sleep(0.02)
        assert runtime.outbox.records[ref].read_at is not None


@pytest.mark.asyncio
async def test_temporary_collection_uses_new_input_reception_and_preserves_references(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        start = runtime.now()
        await runtime.create(
            {"name": "experiment", "description": "Observe", "trigger_type": "event", "temporary": True},
        )
        future = start + timedelta(days=6)
        monkeypatch.setattr(runtime, "now", lambda: future)
        event = {"id": "old-observation", "event_ts": start - timedelta(days=30), "source": "lab", "description": "Old"}
        await runtime.append("experiment", event)
        monkeypatch.setattr(runtime, "now", lambda: start + timedelta(days=8))
        await runtime.append("experiment", event)  # Duplicate delivery must not extend lifetime.
        assert (await runtime.snapshot("experiment"))["event"]["metadata"]["_received_at"] == future.isoformat()
        await runtime.run("experiment", AsyncMock(return_value=(ProactiveRunResult(notifications=[]), "session-1")))
        agent = await runtime.definition("experiment")
        await runtime._collect(agent)
        assert runtime.agents.path("experiment").is_dir()
        monkeypatch.setattr(runtime, "now", lambda: start + timedelta(days=14))
        await runtime._collect(agent)
        assert not (tmp_path / "reme_agents/tmp/experiment").exists()
        assert await runtime.event_stream.list() == []


@pytest.mark.asyncio
async def test_names_and_lifetime_are_validated_across_both_directories(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        for name in ("main", "tmp", "../escape"):
            with pytest.raises(ValueError):
                await runtime.create({"name": name, "description": "Observe", "trigger_type": "event"})
        await create_agent(runtime)
        with pytest.raises(ValueError, match="already exists"):
            await runtime.create(
                {"name": "MONITOR", "description": "Other", "trigger_type": "event", "temporary": True},
            )
        with pytest.raises(ValueError, match="immutable"):
            await runtime.update("monitor", {"temporary": True})
        with pytest.raises(ValueError):
            await runtime.update("monitor", {"created_at": runtime.now()})
        config = runtime.agents.path("monitor") / "config.json"
        config.write_text(config.read_text().replace('"temporary": false', '"temporary": true'))
        with pytest.raises(ValueError, match="directory"):
            await runtime.definition("monitor")


@pytest.mark.asyncio
async def test_unhanded_off_results_protect_read_history_and_survive_restart(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        runtime.event_stream.max_read_events = 1
        runtime.event_stream.append_timeout_seconds = 0.02
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        second = {"id": "event-2", "event_ts": runtime.now(), "source": "lab", "description": "Next"}
        await runtime.append("monitor", second)
        batch = await runtime.event_stream.claim("monitor")
        with pytest.raises(TimeoutError, match="handoffs"):
            await runtime.event_stream.ack("monitor", batch["token"])
        await runtime.event_stream.nack("monitor", batch["token"])
        await runtime.outbox.sync()
        assert len(list((runtime.agents.path("monitor") / "events").glob("events_unread.jsonl"))) == 1
    async with running_runtime(tmp_path) as (runtime, _, _):
        await runtime.outbox.sync()
        assert len((await runtime.inbox.read("main"))["items"]) == 1
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.asyncio
async def test_idle_collection_checks_newly_committed_outbox_before_deleting(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await runtime.create(
            {"name": "experiment", "description": "Observe", "trigger_type": "event", "temporary": True},
        )
        await runtime.append("experiment", {"event_ts": runtime.now(), "source": "lab", "description": "Observe"})
        agent = await runtime.definition("experiment")
        future = runtime.now() + timedelta(days=8)
        monkeypatch.setattr(runtime, "now", lambda: future)
        await runtime._collect(agent)
        assert runtime.agents.path("experiment").exists()  # Pending work is never expired.
        await runtime.run("experiment", AsyncMock(return_value=completed_result(runtime, "experiment")))
        assert not runtime.outbox.records  # The file watcher has not reconciled this round yet.
        await runtime._collect(agent)
        assert runtime.agents.path("experiment").exists()
        assert runtime.inbox.pending
        await runtime.inbox.mark_read("main", list(runtime.inbox.pending))
        await runtime._collect(agent)
        assert runtime.agents.path("experiment").exists()  # Retained read references remain valid too.


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
                        priority="P2",
                        paths=[path.relative_to(tmp_path).as_posix(), "daily/result.md"],
                        delivered_at=now.isoformat(),
                        read_at=now.isoformat(),
                    ),
                ),
            )
        await runtime.outbox.sync()
        assert len(runtime.inbox.read_refs) == 1000
        assert "msg-0000" not in runtime.inbox.read_refs
        runtime.inbox.path(True).unlink()
        await runtime.outbox.sync()
        assert len(runtime.inbox.read_refs) == 1000
        assert not runtime.inbox.pending


@pytest.mark.asyncio
async def test_external_frontmatter_updates_and_corrupt_reminders_fail_closed(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.inbox.pending.values()))
        path = runtime.files.path(ref.file)
        document = frontmatter.loads(runtime.files.read(path))
        document["woken_at"] = runtime.now().isoformat()
        runtime.files.write(path, frontmatter.dumps(document))
        await runtime.outbox.sync()
        assert runtime.outbox.records[ref].woken_at is not None
        document["priority"] = "invalid"
        invalid = frontmatter.dumps(document)
        runtime.files.write(path, invalid)
        with pytest.raises(ValueError):
            await runtime.outbox.sync()
        assert path.read_text() == invalid
        assert ref.id in runtime.inbox.pending


@pytest.mark.asyncio
async def test_failed_markdown_state_write_preserves_body_and_retries(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.inbox.pending.values()))
        path = runtime.files.path(ref.file)
        original = path.read_bytes()
        from reme.components.agent_runtime._internal import files

        def failed_sync(_descriptor):
            raise OSError("disk unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(files, "sync_file", failed_sync)
            with pytest.raises(OSError, match="disk unavailable"):
                await runtime.inbox.mark_read("main", [ref.id])
        assert path.read_bytes() == original
        assert ref.id in runtime.inbox.pending
        await runtime.inbox.mark_read("main", [ref.id])
        document = frontmatter.loads(path.read_text())
        assert document.content == "Experiment finished" and document["read_at"]


@pytest.mark.asyncio
async def test_legacy_reminder_logs_require_explicit_migration(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        path = runtime.agents.path("monitor") / "reminder" / f"{runtime.now().date()}.jsonl"
        runtime.files.write(path, '{"id":"legacy"}\n')
        original = path.read_bytes()
        with pytest.raises(ValueError, match="migration"):
            await runtime.outbox.sync()
        assert path.read_bytes() == original


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
        await runtime.create({"name": "timer", "description": "Report", "trigger_type": "cron", "cron": "* * * * *"})
        agent = await runtime.definition("timer")
        start = runtime.now().replace(second=0, microsecond=0)
        monkeypatch.setattr(runtime, "now", lambda: start)
        await runtime._schedule_cron(agent)
        first = runtime.cron["timer"][1]
        assert first == start + timedelta(minutes=1)
        monkeypatch.setattr(runtime, "now", lambda: first)
        await runtime._schedule_cron(agent)
        await runtime._schedule_cron(agent)
        assert (await runtime.snapshot("timer"))["pending_count"] == 1
    async with running_runtime(tmp_path) as (runtime, _, _):
        assert not runtime.cron
        future = first + timedelta(days=1)
        monkeypatch.setattr(runtime, "now", lambda: future)
        await runtime._schedule_cron(await runtime.definition("timer"))
        assert runtime.cron["timer"][1] > future
        assert (await runtime.snapshot("timer"))["pending_count"] == 1


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
        assert (await runtime.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_agent_directory_symlinks_are_rejected(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "reme_agents").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="without symlinks"):
        async with running_runtime(tmp_path):
            pass


@pytest.mark.asyncio
async def test_query_forks_committed_session_without_selecting_the_temporary_branch(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        source = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now())
        result, _ = completed_result(runtime)
        await runtime.run("monitor", AsyncMock(return_value=(result, source)))
        current = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now(), new=True)
        response = await ProactiveQueryStep(app_context=context)(agent_id="monitor", query="What happened?")
        assert response.success and response.answer["answer"] == "Processed"
        assert wrapper.created[-1]["resume"] == source
        assert wrapper.created[-1]["fork_session"]
        assert wrapper.created[-1]["sandbox"] == "read-only"
        assert wrapper.created[-1]["job_tools"] == []
        assert (await runtime.sessions.current("monitor")).session_id == current
        records = (await runtime.sessions.list("monitor"))["items"]
        assert records[0]["temporary"]


@pytest.mark.asyncio
async def test_query_registration_protects_temporary_agent_during_checkpoint_selection(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        start = runtime.now()
        await runtime.create(
            {"name": "experiment", "description": "Observe", "trigger_type": "event", "temporary": True},
        )
        await runtime.append("experiment", {"event_ts": start, "source": "lab", "description": "Finished"})
        source = await runtime.sessions.ensure("experiment", wrapper, {}, start)
        await runtime.run("experiment", AsyncMock(return_value=(ProactiveRunResult(notifications=[]), source)))
        directory = runtime.agents.path("experiment")
        monkeypatch.setattr(runtime, "now", lambda: start + timedelta(days=8))
        snapshot = runtime.snapshot
        checkpoint_entered, checkpoint_release = asyncio.Event(), asyncio.Event()
        model_entered, model_release = asyncio.Event(), asyncio.Event()

        async def checkpoint(agent_id):
            result = await snapshot(agent_id)
            if asyncio.current_task() is querying:
                checkpoint_entered.set()
                await checkpoint_release.wait()
            return result

        async def answer(_query, **kwargs):
            model_entered.set()
            await model_release.wait()
            return {"status": "completed", "result": "Finished", "session_id": kwargs["resume"]}

        monkeypatch.setattr(runtime, "snapshot", checkpoint)
        monkeypatch.setattr(wrapper, "reply", answer)
        querying = asyncio.create_task(ProactiveQueryStep(app_context=context)(agent_id="experiment", query="Status?"))
        collecting = None
        try:
            await asyncio.wait_for(checkpoint_entered.wait(), 2)
            assert runtime.queries.get(querying) == "experiment"
            collecting = asyncio.create_task(runtime.tick(AsyncMock(), "proactive_run"))
            checkpoint_release.set()
            await asyncio.wait_for(model_entered.wait(), 2)
            await asyncio.wait_for(collecting, 2)
            assert directory.is_dir()
            model_release.set()
            assert (await asyncio.wait_for(querying, 2)).success
            assert not runtime.queries
        finally:
            checkpoint_release.set()
            model_release.set()
            for task in (querying, collecting):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(task for task in (querying, collecting) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_query_without_a_completed_checkpoint_releases_ownership(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        with pytest.raises(ValueError, match="no completed session"):
            await ProactiveQueryStep(app_context=context)(agent_id="monitor", query="Status?")
        assert not runtime.queries and not wrapper.created


@pytest.mark.asyncio
async def test_session_registry_recovers_an_incomplete_tail_without_losing_complete_history(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        original = await runtime.sessions.ensure("main", wrapper, {}, runtime.now())
        path = runtime.sessions.path("main")
        complete = path.read_bytes()
        path.write_bytes(complete + b'{"session_id":"interrupted')
        assert (await runtime.sessions.current("main")).session_id == original
        assert (await runtime.sessions.list("main"))["total"] == 1
        assert path.read_bytes() == complete + b'{"session_id":"interrupted'
        await runtime.sessions.record("main", "recovered-query", runtime.now(), temporary=True)
        assert path.read_bytes().startswith(complete)
        assert (await runtime.sessions.list("main"))["total"] == 2
        assert (await runtime.sessions.current("main")).session_id == original
        with path.open("ab") as handle:
            handle.write(b"{invalid}\n")
        with pytest.raises(json.JSONDecodeError):
            await runtime.sessions.list("main")


@pytest.mark.parametrize("directory", ["", ".", "..", "/tmp/agents", "../agents", "~/agents", "a//b", "C:\\agents"])
def test_agent_root_must_be_a_normalized_workspace_subdirectory(tmp_path, directory):
    with pytest.raises(ValueError):
        ApplicationContext(workspace_dir=str(tmp_path), reme_agent_dir=directory)


@pytest.mark.asyncio
async def test_custom_root_and_unicode_identity_share_the_same_layout(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path), reme_agent_dir="runtime/agents")
    wrapper = FakeWrapper(app_context=context)
    stream = AgentEventStream(app_context=context)
    runtime = LocalAgentRuntime(app_context=context)
    context.components = {"agent_wrapper": {"default": wrapper}, "event_stream": {"default": stream}}
    async with wrapper, stream, runtime:
        created = await runtime.create({"name": "屏幕观察", "description": "Observe screens", "trigger_type": "event"})
        await runtime.append("屏幕观察", {"event_ts": runtime.now(), "source": "screen", "description": "New"})
        path = tmp_path / "runtime/agents/屏幕观察"
        assert (path / "config.json").is_file()
        assert created["memory_path"] == "runtime/agents/屏幕观察/MEMORY.md"
        assert created["memory_content"] == (path / "MEMORY.md").read_text()
        assert list((path / "events").glob("events_unread.jsonl"))
