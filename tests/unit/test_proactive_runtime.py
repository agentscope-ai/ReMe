"""Runtime ownership, configuration and task lifecycle contracts."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
import json

import pytest

from reme.core.context import RuntimeContext
from reme.components.proactive import ChatAgentRuntime, ProactiveAgentRuntime
from reme.config import resolve_app_config
from reme.core.registry import R
from reme.enumeration import ComponentEnum
from reme.schema.proactive_agent import ManageProactiveRuntimeRequest
from reme.steps.proactive.agent_api import AgentListStep, AgentSessionStep, AskAgentStep, TaskDocumentStep
from reme.steps.proactive.tools import ManageProactiveRuntimeStep
from .proactive_helpers import running_system, queue_event, runtime_config


@pytest.mark.parametrize("name", ["proactive", "proactive_cc", "demo"])
def test_builtin_configuration_has_five_runtimes_and_private_execution_jobs(name):
    config = resolve_app_config(config=name, log_config=False)
    components, jobs = config["components"], config["jobs"]
    assert set(components["agent_runtime"]) == {
        "Remy",
        "proactive_memory",
        "proactive_note",
        "proactive_fin",
        "proactive_news",
    }
    assert not {"agent_inbox", "proactive_manager"} & components.keys()
    manager = components["agent_manager"]["default"]
    assert set(manager["runtimes"]) == set(components["agent_runtime"]) - {"Remy"}
    for job in jobs.values():
        for step in job.get("steps", []) + job.get("init_steps", []):
            assert R.get(ComponentEnum.STEP, step["backend"]) is not None
    for job in [
        "run_chat_agent",
        "process_chat_notifications",
        "chat_notifications_loop",
        "agent_manager_loop",
        "run_proactive_agent",
        "submit_chat_reply",
    ]:
        assert jobs[job]["enable_serve"] is False
    assert (
        not {
            "todo",
            "idea",
            "append_task_events",
            "manage_proactive_agent",
            "run_memory_agent",
            "run_notebook_agent",
            "run_finance_agent",
            "run_news_agent",
        }
        & jobs.keys()
    )


@pytest.mark.asyncio
async def test_distinct_owners_and_create_only_memory(tmp_path):
    async with running_system(tmp_path, initial_memory="Original") as (manager, context, wrapper):
        chat = context.components["agent_runtime"]["Remy"]
        assert isinstance(chat, ChatAgentRuntime) and not manager.runtimes
        response = await ManageProactiveRuntimeStep(app_context=context)(
            agent_type="event",
            action="create",
            agent_name="monitor",
            task_description="Track market changes of 3% or more.",
        )
        assert response.success, response.answer
        task = await manager.resolve("monitor")
        assert isinstance(task, ProactiveAgentRuntime)
        assert task.agent_wrapper is chat.agent_wrapper is wrapper
        assert task.inbox is not chat.inbox and task.outbox.target is chat
        assert "3%" in task.task_path.read_text()
        assert task.is_started and task.inbox.is_started and task.outbox.is_started
        assert not wrapper.turns
        chat.memory_path.write_text("User edited")
    assert not task.is_started and not chat.is_started
    async with running_system(tmp_path, initial_memory="Changed") as (manager, context, _):
        assert context.components["agent_runtime"]["Remy"].memory_path.read_text() == "User edited"


@pytest.mark.asyncio
async def test_delete_cancels_active_and_waiting_work_and_restore_preserves_role_and_files(tmp_path):
    async with running_system(
        tmp_path,
        runtimes=[
            {
                "agent_type": "event",
                "agent_name": "monitor",
                "task_description": "Keep this",
            },
        ],
    ) as (manager, _, wrapper):
        runtime = await manager.resolve("monitor")
        await queue_event(manager)
        entered = asyncio.Event()

        async def work():
            async with runtime.execution():
                entered.set()
                await asyncio.Event().wait()

        active = asyncio.create_task(work())
        await entered.wait()
        waiting = asyncio.create_task(work())
        await asyncio.sleep(0)
        await manager.delete("monitor")
        assert active.cancelled() and waiting.cancelled() and not runtime.is_started
        assert runtime.task_path.read_text() == "Keep this"
        result = await manager.create({"agent_type": "event", "agent_name": "monitor"})
        restored = await manager.resolve("monitor")
        assert result["restored"] and restored.binding == runtime.binding
        assert (await restored.inbox.snapshot())["pending_count"] == 1
        assert restored.task_path.read_text() == "Keep this" and wrapper.is_started


@pytest.mark.asyncio
async def test_identity_and_trigger_contracts_and_disk_edits(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (manager, _, _):
        for values in ({"run_job": "run_news_agent"}, {"schedule": "0 8 * * *"}, {"max_batch_events": 4}):
            with pytest.raises(ValueError):
                await manager.update("monitor", values)
        for name in ("Remy", "MONITOR", "../outside", "CON"):
            with pytest.raises(ValueError):
                await manager.create({"agent_type": "event", "agent_name": name})
        path = (await manager.resolve("monitor")).config_path
        saved = json.loads(path.read_text())
        saved["enabled"] = False
        path.write_text(json.dumps(saved))
        assert (await manager.status("monitor"))["config"]["enabled"] is False
        with pytest.raises(ValueError):
            ManageProactiveRuntimeRequest(action="update", agent_name="monitor", run_job="run_news_agent")


@pytest.mark.asyncio
async def test_chat_internal_job_preserves_one_service_response_and_formal_binding(tmp_path):
    async with running_system(tmp_path) as (_, context, wrapper):
        queue = asyncio.Queue()
        step = AskAgentStep(app_context=context, agent_runtime="Remy")
        response = await step(RuntimeContext(stream_queue=queue), query="Hello")
        assert response.success and response.answer == "Processed"
        assert wrapper.turns[-1][1]["job_tools"] == ["manage_proactive_runtime", "manage_connector", "manage_notebook"]
        assert [item["name"].rsplit("-", 1)[0] for item in wrapper.turns[-1][1]["skills"]] == [
            "manage-proactive-runtime",
            "manage-connector",
            "manage-notebook",
        ]
        assert not any(chunk.chunk_type == "done" for chunk in list(queue._queue))
        listing = (await AgentListStep(app_context=context, agent_runtime="Remy")()).answer
        assert listing["total"] == 1 and listing["items"][0]["agent_type"] == "chat"
        assert (await AgentSessionStep(app_context=context, agent_runtime="Remy")()).answer["agent"]["session_id"]


@pytest.mark.asyncio
async def test_task_document_defaults_are_created_without_model_calls(tmp_path):
    async with running_system(tmp_path) as (manager, context, wrapper):
        await manager.create(
            {"agent_type": "time", "agent_name": "brief", "schedule": "0 8 * * *"},
        )
        response = await TaskDocumentStep(app_context=context)(action="read", agent_name="brief")
        assert response.success and response.answer["content"] and response.answer["mtime"]
        assert not wrapper.turns


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("owner", ["notes", "实验观察", "a" * 80])
@pytest.mark.asyncio
async def test_yaml_skill_templates_are_scoped_per_invocation_and_do_not_rotate_chat(tmp_path, language, owner):
    """The same template can be loaded concurrently without changing logical session bindings."""
    async with running_system(tmp_path, language=language, runtimes=[runtime_config("proactive_note", owner)]) as (
        manager,
        context,
        wrapper,
    ):
        chat = context.components["agent_runtime"]["Remy"]
        runtime = await manager.resolve(owner)
        first = chat.get_skills()
        second = runtime.get_skills()
        assert len(first) == 3 and len(second) == 2
        repeated = runtime.get_skills()
        for left, right in zip(second, repeated, strict=True):
            assert left == right
        assert first[-1]["content"] == second[-1]["content"]
        async with wrapper.skill_scope(skills=first) as foreground, wrapper.skill_scope(skills=second) as background:
            with pytest.raises(FileExistsError):
                async with wrapper.skill_scope(skills=repeated):
                    pytest.fail("Concurrent scopes must not reuse the same agent's skill names")
            assert all(
                (source / "SKILL.md").is_file()
                for scope in [foreground, background]
                for source in scope["_skills"].sources.values()
            )
        repeated[0]["content"] = "Changed by one invocation"
        assert runtime.get_skills() == second
        for _ in range(2):
            response = await AskAgentStep(app_context=context, agent_runtime="Remy")(
                RuntimeContext(stream_queue=asyncio.Queue()),
                query="Hello",
            )
            assert response.success
        assert len(wrapper.created) == 1


@pytest.mark.asyncio
async def test_application_orders_static_runtimes_and_manager_and_closes_dynamic_children(tmp_path, monkeypatch):
    """Exercise the real Application dependency graph without an SDK process or external service."""
    from reme.core.application import Application
    from reme.core.registry import create_application_registry
    from reme.plugin.manager import PluginRuntime
    from .proactive_helpers import FakeWrapper

    config = resolve_app_config(config="proactive", log_config=False)
    config.update(workspace_dir=str(tmp_path), enable_logo=False, log_to_file=False)
    config["service"] = {"backend": "cli"}
    config["components"] = {
        name: value for name, value in config["components"].items() if name in {"agent_runtime", "agent_manager"}
    }
    config["components"]["agent_wrapper"] = {"default": {"backend": "test_native"}}
    config["jobs"] = {
        name: value
        for name, value in config["jobs"].items()
        if name
        in {
            "run_chat_agent",
            "run_proactive_agent",
            "agent_manager_loop",
            "chat_notifications_loop",
            "process_chat_notifications",
            "manage_connector",
            "manage_proactive_runtime",
            "manage_notebook",
            "submit_chat_reply",
        }
    }
    registry = create_application_registry()
    registry.register(FakeWrapper, "test_native")
    monkeypatch.setattr(
        "reme.core.application.resolve_plugin_runtime",
        lambda _: PluginRuntime(config=config, registry=registry),
    )
    app = Application()
    async with app:
        runtimes = app.context.components["agent_runtime"]
        manager = app.context.components["agent_manager"]["default"]
        assert set(manager.runtimes) == set(runtimes) - {"Remy"}
        assert all(manager.runtimes[name] is runtimes[name] for name in manager.static_names)
        assert all(runtime.inbox.is_started and runtime.outbox.is_started for runtime in runtimes.values())
        await manager.create({"agent_type": "event", "agent_name": "extra"})
        dynamic = await manager.resolve("extra")
    assert not dynamic.is_started and all(not runtime.is_started for runtime in runtimes.values())


@pytest.mark.parametrize(
    "patch",
    [{"job_tools": ["manage_notebook"]}, {"skills": []}, {"agent_type": "time", "schedule": "0 8 * * *"}],
)
@pytest.mark.asyncio
async def test_discovery_rejects_changed_wiring_before_scheduling_or_accepting_inputs(tmp_path, patch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (manager, _, _):
        runtime = await manager.resolve("monitor")
        value = json.loads(runtime.config_path.read_text())
        runtime.config_path.write_text(json.dumps({**value, **patch}))
        with pytest.raises(ValueError, match="wiring"):
            await manager.resolve("monitor")
        assert not await runtime.inbox.pending_ids()


@pytest.mark.asyncio
async def test_retired_static_runtime_rejects_already_resolved_query_calls(tmp_path):
    async with running_system(tmp_path) as (manager, context, wrapper):
        runtime = ProactiveAgentRuntime(
            agent_type="event",
            name="monitor",
            agent_wrapper=wrapper,
            notification_target=context.components["agent_runtime"]["Remy"],
            app_context=context,
        )
        async with runtime:
            manager.runtimes["monitor"] = runtime
            await runtime.ensure_session({})
            await manager.delete("monitor")
            assert runtime.is_started
            with pytest.raises(RuntimeError, match="retir"):
                async with runtime.query_session({}):
                    pytest.fail("A previously resolved static Runtime ran after deletion")
            assert len(wrapper.created) == 1


@pytest.mark.asyncio
async def test_runtime_initializes_without_jobs_and_restores_its_own_binding(tmp_path):
    """Task files and capabilities are component-owned before any inference runs."""
    async with running_system(tmp_path) as (manager, context, wrapper):
        context.app_config.jobs.clear()
        config = runtime_config("proactive_note")
        config.update(task_description="Original task", recent_history_hours=12, max_local_search_calls=3)
        await manager.create(config)
        runtime = await manager.resolve("monitor")
        assert runtime.task_path.read_text() == "Original task"
        runtime.task_path.write_text("User edited {requirements}")
        await runtime.initialize_task()
        options = runtime.session_options()
        assert runtime.task_path.read_text() == "User edited {requirements}"
        assert options["job_tools"] == ["manage_notebook"]
        assert options["injected_job_kwargs"]["_automatic_notebook"] is True
        assert runtime.initial_config.recent_history_hours == 12
        assert not wrapper.created and not wrapper.turns
        expected = options
    async with running_system(tmp_path) as (manager, _, wrapper):
        runtime = await manager.resolve("monitor")
        assert runtime.task_path.read_text() == "User edited {requirements}"
        restored = runtime.session_options()
        assert {key: value for key, value in restored.items() if key != "system_prompt"} == {
            key: value for key, value in expected.items() if key != "system_prompt"
        }
        assert runtime.initial_config.max_local_search_calls == 3
        assert not wrapper.created


@pytest.mark.asyncio
async def test_public_creation_selects_capabilities_under_trusted_tool_constraints(tmp_path):
    async with running_system(tmp_path) as (manager, context, _):
        tool = ManageProactiveRuntimeStep(app_context=context)
        response = await tool(
            action="create",
            agent_name="notes",
            agent_type="event",
            task_description="Track progress",
            skills=["proactive-record", "manage-notebook"],
            job_tools=["manage_notebook"],
        )
        assert response.success, response.answer
        runtime = await manager.resolve("notes")
        assert runtime.binding.job_tools == ["manage_notebook"]
        assert runtime.session_options()["injected_job_kwargs"]["_automatic_notebook"] is True
        with pytest.raises(ValueError, match="disallowed"):
            await tool(action="create", agent_name="unsafe", agent_type="event", job_tools=["manage_connector"])
        assert not (manager.registry.directory / "unsafe").exists()
        with pytest.raises(ValueError):
            ManageProactiveRuntimeRequest(
                action="create",
                agent_name="unsafe",
                agent_type="event",
                injected_job_kwargs={"_automatic_notebook": False},
            )


@pytest.mark.parametrize(
    "values",
    [
        {"skills": ["../outside"]},
        {"skills": ["proactive-record", "proactive-record"]},
        {"job_tools": "manage_notebook"},
        {"skills": [False]},
        {"skills": ["MixedCase"]},
        {"skills": ["a" * 65]},
        {"skills": ["_invalid"]},
        {"run_job": "run_proactive_agent"},
    ],
)
def test_runtime_bindings_reject_paths_duplicates_and_invalid_types(values):
    from reme.schema.proactive_agent import ProactiveRuntimeConfig

    with pytest.raises(ValueError):
        ProactiveRuntimeConfig(agent_name="monitor", agent_type="event", **values)


@pytest.mark.parametrize("skill", ["proactive-record-project", "manage_notebook"])
@pytest.mark.asyncio
async def test_runtime_accepts_project_skills_without_a_builtin_template(tmp_path, skill):
    async with running_system(tmp_path) as (manager, _, _):
        await manager.create({"agent_name": "monitor", "agent_type": "event", "skills": [skill]})
        runtime = await manager.resolve("monitor")
        assert runtime.get_skills() == [{"name": skill}]


@pytest.mark.asyncio
async def test_creation_preserves_requirements_when_definition_publication_is_interrupted(tmp_path, monkeypatch):
    """A durable definition must never outlive the user's initial requirements."""
    async with running_system(tmp_path) as (manager, _, _):
        save = manager.registry.save

        def interrupted(definition):
            save(definition)
            raise OSError("Lost result after publishing the definition")

        monkeypatch.setattr(manager.registry, "save", interrupted)
        with pytest.raises(OSError):
            await manager.create(
                {
                    "agent_name": "monitor",
                    "agent_type": "event",
                    "task_description": "Keep exact user requirements",
                },
            )
        assert (manager.registry.directory / "monitor/TASK.md").read_text() == "Keep exact user requirements"
        assert "monitor" not in manager.runtimes
    async with running_system(tmp_path) as (manager, _, _):
        runtime = await manager.resolve("monitor")
        assert runtime.task_path.read_text() == "Keep exact user requirements"


@pytest.mark.asyncio
async def test_task_write_failure_leaves_identity_available_for_retry(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, _, _):

        async def failed_write(*_args):
            raise OSError("Task write failed")

        with monkeypatch.context() as patch:
            patch.setattr(ProactiveAgentRuntime, "initialize_task", failed_write)
            with pytest.raises(OSError):
                await manager.create({"agent_name": "monitor", "agent_type": "event"})
        assert not (manager.registry.directory / "monitor/config.json").exists()
        assert "monitor" not in manager.runtimes
        await manager.create({"agent_name": "monitor", "agent_type": "event", "task_description": "Retry"})
        assert (await manager.resolve("monitor")).task_path.read_text() == "Retry"


@pytest.mark.asyncio
async def test_skill_validation_failure_does_not_publish_or_reserve_task(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, _, _):

        def invalid_skill(_self):
            raise ValueError("Invalid skill template")

        with monkeypatch.context() as patch:
            patch.setattr(ProactiveAgentRuntime, "get_skills", invalid_skill)
            with pytest.raises(ValueError, match="skill"):
                await manager.create({"agent_name": "monitor", "agent_type": "event"})
        assert not (manager.registry.directory / "monitor").exists()
        await manager.create({"agent_name": "monitor", "agent_type": "event"})
        assert (await manager.resolve("monitor")).is_started


@pytest.mark.asyncio
async def test_closed_manager_rejects_mutations_before_writing_files(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (manager, _, _):
        runtime = await manager.resolve("monitor")
        before = runtime.config_path.read_bytes()
        await manager.close()
        with pytest.raises(RuntimeError, match="closed"):
            await manager.create({"agent_name": "another", "agent_type": "event"})
        with pytest.raises(RuntimeError, match="closed"):
            await manager.update("monitor", {"enabled": False})
        assert runtime.config_path.read_bytes() == before
        assert not (manager.registry.directory / "another").exists()
