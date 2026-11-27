"""Connector controls and independent, retryable recipient delivery."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import json

import pytest

from reme.components.job import BaseJob
from reme.schema import Event
from reme.schema.proactive_agent import ManageConnectorRequest
from reme.steps.proactive.tools import ManageConnectorStep
from reme.steps.proactive.connector import BaseConnectorStep
from .proactive_helpers import running_system


@pytest.mark.asyncio
async def test_send_reports_success_permanent_rejection_and_retryable_capacity_separately(tmp_path):
    async with running_system(
        tmp_path,
        runtimes=[
            {"agent_type": "event", "agent_name": "good"},
            {"agent_type": "event", "agent_name": "full"},
            {"agent_type": "time", "agent_name": "clock", "schedule": "0 8 * * *"},
        ],
    ) as (manager, context, _):
        full = await manager.resolve("full")
        full.inbox.max_pending_items = 1
        event = Event(id="first", event_ts=manager.now(), source="test", description="One")
        await manager.append_events("full", [event])
        next_event = event.model_copy(update={"id": "next"})
        tool = ManageConnectorStep(app_context=context)
        response = await tool(
            action="send",
            runtime_names=["good", "full", "clock", "missing"],
            events=[next_event.model_dump(mode="json")],
        )
        assert not response.success
        result = response.answer
        assert (
            result["deliveries"] == {"good": ["next"]}
            and set(result["rejected"]) == {"clock", "missing"}
            and set(result["errors"]) == {"full"}
        )
        batch = await full.inbox.claim()
        await full.inbox.ack(batch["token"])
        retry = await tool(action="send", runtime_names=["full"], events=[next_event.model_dump(mode="json")])
        assert retry.success and await full.inbox.pending_ids() == {"next"}
        conflict = await tool(
            action="send",
            runtime_names=["good"],
            events=[next_event.model_copy(update={"description": "Changed"}).model_dump(mode="json")],
        )
        assert set(conflict.answer["rejected"]) == {"good"}


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"action": "pause"},
        {"action": "send", "runtime_names": ["a", "A"], "events": []},
        {
            "action": "send",
            "runtime_names": ["a"],
            "events": [{"source": "time", "description": "Reserved", "event_ts": "2026-11-20T00:00:00Z"}],
        },
        {"action": "enable", "connector_job": "a", "events": []},
        {"action": "send", "runtime_names": ["a"], "events": [], "connector_job": "a"},
    ],
)
def test_connector_action_boundaries(values):
    with pytest.raises(ValueError):
        ManageConnectorRequest.model_validate(values)


class SmallConnector(BaseConnectorStep):
    """Write an isolated source payload without external devices."""

    source = "test_source"

    def collect(self, directory):
        (directory / "data.txt").write_text("Source")
        return [{"path": "data.txt"}]


@pytest.mark.asyncio
async def test_pause_gate_survives_restart_and_never_changes_task_enabled(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        _,
    ):
        job = BaseJob(name="screen_capture_cron", app_context=context, steps=[])
        context.jobs[job.name] = job
        async with job:
            tool = ManageConnectorStep(app_context=context, connector_jobs=[job.name])
            await tool(action="pause", connector_job=job.name)
            step = SmallConnector(app_context=context, connector_job=job.name, runtime_names=["monitor"])
            assert (await step()).answer["skipped"] == "paused"
            assert (await manager.definition("monitor")).enabled and not (tmp_path / "resource").exists()
            with pytest.raises(ValueError):
                await tool(action="pause", connector_job="run_memory_agent")
    async with running_system(tmp_path) as (manager, context, _):
        job = BaseJob(name="screen_capture_cron", app_context=context, steps=[])
        context.jobs[job.name] = job
        async with job:
            step = SmallConnector(app_context=context, connector_job=job.name, runtime_names=["monitor"])
            assert (await step()).answer["skipped"] == "paused"
            await ManageConnectorStep(app_context=context, connector_jobs=[job.name])(
                action="enable",
                connector_job=job.name,
            )
            assert (await step()).success
            events = (await (await manager.resolve("monitor")).inbox.read())["items"]
            assert len(events) == 1 and events[0]["item"]["paths"][0].endswith("metadata.json")


@pytest.mark.asyncio
async def test_pausing_after_publication_defers_frozen_delivery_until_resume(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        _,
    ):
        from reme.steps.proactive.connector.base_connector import update_connector_control

        step = SmallConnector(app_context=context, connector_job="sample", runtime_names=["monitor"])
        original = step._collect

        async def collect(*args):
            response = await original(*args)
            await update_connector_control(step, "sample", enabled=False)
            return response

        monkeypatch.setattr(step, "_collect", collect)
        assert (await step()).success
        runtime = await manager.resolve("monitor")
        assert not await runtime.inbox.pending_ids()
        state = next(tmp_path.glob("resource/test_source/*/*/.delivery.json"))
        frozen = json.loads(state.read_text())["events"]
        await update_connector_control(step, "sample", enabled=True)
        monkeypatch.setattr(step, "_collect", original)
        monkeypatch.setattr(step, "collect", lambda directory: [])
        await step()
        assert await runtime.inbox.pending_ids() == {frozen[0]["id"]}


@pytest.mark.parametrize("kind", ["resource", "harness", "baseline"])
@pytest.mark.asyncio
async def test_paused_connector_waiting_for_source_lock_does_not_begin_collection(tmp_path, monkeypatch, kind):
    import asyncio
    from reme.steps.proactive.connector.base_connector import update_connector_control
    from reme.steps.proactive.connector.harness import CodexDialogueStep, CodexDialogueInitStep
    from reme.utils.files.path_lock import get_path_lock

    async with running_system(tmp_path) as (_, context, _):
        reached = asyncio.Event()
        if kind == "resource":
            step = SmallConnector(app_context=context, connector_job="sample", runtime_names=["monitor"])
            original = step.resource_target

            def target():
                value = original()
                reached.set()
                return value

            monkeypatch.setattr(step, "resource_target", target)
        else:
            cls = CodexDialogueStep if kind == "harness" else CodexDialogueInitStep
            step = cls(
                app_context=context,
                connector_job="sample",
                codex_home=tmp_path / "codex",
                runtime_names=["monitor"],
            )
            original = step._authorize

            def authorize():
                value = original()
                reached.set()
                return value

            monkeypatch.setattr(step, "_authorize", authorize)
        source_root = tmp_path / "resource" / step.source
        async with await get_path_lock(source_root):
            task = asyncio.create_task(step())
            await asyncio.wait_for(reached.wait(), 1)
            await update_connector_control(step, "sample", enabled=False)
        response = await asyncio.wait_for(task, 1)
        assert response.answer["skipped"] == "paused"
        assert not source_root.exists()
