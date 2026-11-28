"""Preset discovery, durable routing and native HTTP/MCP management."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from contextlib import asynccontextmanager
import json
from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastmcp import Client
import httpx
import pytest

from reme.components.job import BaseJob
from reme.components.proactive.connector import ConnectorManager
from reme.components.service import HttpService
from reme.core.context import RuntimeContext
from reme.schema.proactive_connector import ManageConnectorRequest
from reme.steps.proactive.connector import BaseConnectorStep
from reme.steps.proactive.connector.base import connector_control_path, read_connector_control
from reme.steps.proactive.tools import ManageConnectorStep
from .proactive_helpers import running_system


class ProbeConnector(BaseConnectorStep):
    """Replace the device or network boundary with one source file."""

    source = "connector_probe"

    def collect(self, directory):
        (directory / "data.txt").write_text("Observed source data")
        return [{"path": "data.txt"}]


@asynccontextmanager
async def preset(tmp_path, *, seed=True):
    tasks = [
        {"agent_type": "event", "agent_name": "first"},
        {"agent_type": "event", "agent_name": "second"},
        {"agent_type": "time", "agent_name": "clock", "schedule": "0 8 * * *"},
    ]
    async with running_system(tmp_path, runtimes=tasks if seed else None) as (manager, context, _):
        job = BaseJob(
            name="screen_capture_cron",
            description="A preset source",
            app_context=context,
            steps=[
                {
                    "backend": "screen_capture_step",
                    "connector_job": "screen_capture_cron",
                    "runtime_names": ["first"],
                },
            ],
        )
        context.jobs[job.name] = job
        connectors = ConnectorManager(app_context=context, agent_manager=manager)
        context.components["connector_manager"] = {"default": connectors}
        async with job, connectors:
            tool = ManageConnectorStep(app_context=context)
            yield manager, context, job, tool


def collector(context, job):
    return ProbeConnector(app_context=context, connector_job=job.name, runtime_names=["first"])


@pytest.mark.asyncio
async def test_preset_name_and_partial_updates_share_the_collection_source_identity(tmp_path):
    async with preset(tmp_path) as (manager, context, job, tool):
        source_id = "preset:" + job.name
        result = (await tool(action="update", source_id=source_id, name="Project observations", enabled=False)).answer
        assert result["name"] == "Project observations" and result["runtime_names"] == ["first"]
        await tool(action="update", source_id=source_id, enabled=True)
        await collector(context, job)()
        source = (await (await manager.resolve("first")).inbox.overview())["items"][0]["source_ref"]
        assert source == {"source_id": source_id, "name": "Project observations", "trigger_type": "event"}
        await tool(action="update", source_id=source_id, name="Renamed")
        assert (await (await manager.resolve("first")).inbox.overview())["items"][0]["source_ref"] == source


@pytest.mark.asyncio
async def test_list_is_readonly_and_includes_defaults_and_availability(tmp_path):
    async with preset(tmp_path) as (_, context, job, tool):
        before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
        result = (await tool(action="list")).answer
        assert result["total"] == 8
        item = next(item for item in result["items"] if item["source_id"] == "preset:" + job.name)
        assert item["name"] == job.description
        assert item["available"] and item["enabled"] and item["runtime_names"] == ["first"]
        assert before == {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
        unavailable = ManageConnectorStep(app_context=context)
        item = next(
            item
            for item in (await unavailable(action="list")).answer["items"]
            if item["source_id"] == "preset:finance_news_cron"
        )
        assert not item["available"] and item["runtime_names"] == ["proactive_fin"]
        assert not (tmp_path / "resource").exists()


@pytest.mark.asyncio
async def test_targets_and_gate_survive_restart_and_apply_to_collection(tmp_path):
    async with preset(tmp_path) as (manager, context, job, tool):
        await tool(action="update", enabled=False, source_id="preset:" + job.name)
        updated = (await tool(action="update", source_id="preset:" + job.name, runtime_names=["second"])).answer
        assert not updated["enabled"] and updated["runtime_names"] == ["second"]
        assert (await collector(context, job)()).answer["skipped"] == "paused"
        resumed = (await tool(action="update", enabled=True, source_id="preset:" + job.name)).answer
        assert resumed["runtime_names"] == ["second"]
        assert (await collector(context, job)()).success
        assert not await (await manager.resolve("first")).inbox.pending_ids()
        assert len(await (await manager.resolve("second")).inbox.pending_ids()) == 1
    async with preset(tmp_path, seed=False) as (manager, context, job, tool):
        item = next(
            item for item in (await tool(action="list")).answer["items"] if item["source_id"] == "preset:" + job.name
        )
        assert item["enabled"] and item["runtime_names"] == ["second"]
        assert (await collector(context, job)()).success
        assert len(await (await manager.resolve("second")).inbox.pending_ids()) == 2
        assert not await (await manager.resolve("first")).inbox.pending_ids()


@pytest.mark.asyncio
async def test_multiple_targets_replacement_and_download_without_targets(tmp_path):
    async with preset(tmp_path) as (manager, context, job, tool):
        for targets in (["first", "second"], ["second"], []):
            await tool(action="update", source_id="preset:" + job.name, runtime_names=targets)
            assert (await collector(context, job)()).success
        assert len(await (await manager.resolve("first")).inbox.pending_ids()) == 1
        assert len(await (await manager.resolve("second")).inbox.pending_ids()) == 2
        manifests = list((tmp_path / "resource/connector_probe").glob("*/*/metadata.json"))
        assert len(manifests) == 3
        assert sum(path.with_name(".delivery.json").exists() for path in manifests) == 2
        assert job.step_configs[0]["runtime_names"] == ["first"]


@pytest.mark.asyncio
async def test_detaching_targets_still_finishes_frozen_delivery(tmp_path, monkeypatch):
    async with preset(tmp_path) as (manager, context, job, tool):
        original = manager.deliver_events
        failed = AsyncMock(return_value={"deliveries": {}, "rejected": {}, "errors": {"first": "offline"}})
        monkeypatch.setattr(manager, "deliver_events", failed)
        with pytest.raises(RuntimeError, match="offline"):
            await collector(context, job)()
        frozen = failed.await_args.kwargs["events"]
        await tool(action="update", source_id="preset:" + job.name, runtime_names=[])
        delivery = AsyncMock(side_effect=original)
        monkeypatch.setattr(manager, "deliver_events", delivery)
        assert (await collector(context, job)()).success
        delivery.assert_awaited_once_with(runtime_names=["first"], events=frozen)
        assert await (await manager.resolve("first")).inbox.pending_ids() == {frozen[0].id}


@pytest.mark.asyncio
async def test_target_changes_during_collection_start_with_next_batch(tmp_path, monkeypatch):
    async with preset(tmp_path) as (manager, context, job, tool):
        step = collector(context, job)
        started, release = ThreadEvent(), ThreadEvent()
        collect = step.collect

        def blocked(directory):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test collection was not released")
            return collect(directory)

        monkeypatch.setattr(step, "collect", blocked)
        task = asyncio.create_task(step())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await tool(action="update", source_id="preset:" + job.name, runtime_names=["second"])
        finally:
            release.set()
        assert (await task).success
        assert len(await (await manager.resolve("first")).inbox.pending_ids()) == 1
        assert not await (await manager.resolve("second")).inbox.pending_ids()
        assert (await collector(context, job)()).success
        assert len(await (await manager.resolve("second")).inbox.pending_ids()) == 1


@pytest.mark.asyncio
async def test_concurrent_gate_and_target_updates_do_not_lose_fields(tmp_path):
    async with preset(tmp_path) as (_, context, job, tool):
        other = ManageConnectorStep(app_context=context)
        await asyncio.gather(
            tool(action="update", source_id="preset:" + job.name, runtime_names=["second"]),
            other(action="update", enabled=False, source_id="preset:" + job.name),
        )
        control = await read_connector_control(tool, job.name)
        assert control["runtime_names"] == ["second"] and not control["enabled"]


@pytest.mark.parametrize("target", ["missing", "clock", "deleted"])
@pytest.mark.asyncio
async def test_invalid_target_updates_leave_control_unchanged(tmp_path, target):
    async with preset(tmp_path) as (manager, _, job, tool):
        if target == "deleted":
            await manager.create({"agent_type": "event", "agent_name": target})
            await manager.delete(target)
        await tool(action="update", enabled=False, source_id="preset:" + job.name)
        _, path = connector_control_path(tool, job.name)
        before = path.read_bytes()
        with pytest.raises(ValueError):
            await tool(action="update", source_id="preset:" + job.name, runtime_names=["second", target])
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    "patch",
    [
        {"runtime_names": None},
        {"runtime_names": ["first", "FIRST"]},
        {"runtime_names": ["../outside"]},
        {"enabled": "true"},
        {"unexpected": True},
        {"updated_at": "2026-11-27T10:00:00"},
    ],
)
@pytest.mark.asyncio
async def test_corrupt_control_fails_closed_before_collecting(tmp_path, patch):
    async with preset(tmp_path) as (_, context, job, tool):
        await tool(action="update", enabled=False, source_id="preset:" + job.name)
        _, path = connector_control_path(tool, job.name)
        path.write_text(json.dumps({**json.loads(path.read_text()), **patch}))
        items = (await tool(action="list")).answer["items"]
        assert next(item for item in items if item["source_id"] == "preset:" + job.name)["last_error"]
        with pytest.raises(ValueError):
            await collector(context, job)()
        assert not (tmp_path / "resource").exists()


@pytest.mark.parametrize(
    "values",
    [
        {"action": "list", "connector_job": "sample"},
        {"action": "update", "connector_job": "sample"},
        {"action": "update", "connector_job": "sample", "runtime_names": None},
        {"action": "update", "connector_job": "sample", "runtime_names": ["a", "A"]},
        {"action": "update", "connector_job": "sample", "runtime_names": [], "events": []},
        {"action": "pause", "connector_job": "sample", "runtime_names": []},
        {
            "action": "send",
            "runtime_names": [],
            "events": [{"event_ts": "2026-11-27T10:00:00Z", "source": "log", "description": "New data"}],
        },
    ],
)
def test_management_requests_reject_irrelevant_or_invalid_arguments(values):
    with pytest.raises(ValueError):
        ManageConnectorRequest.model_validate(values)


@pytest.mark.asyncio
async def test_management_permissions_cover_control_and_target_definitions(tmp_path):
    async with preset(tmp_path) as (_, _, job, tool):
        with pytest.raises(PermissionError):
            await tool(RuntimeContext(_allowed_paths=["resource"]), action="list")
        with pytest.raises(PermissionError):
            await tool(
                RuntimeContext(_allowed_paths=["metadata"]),
                action="update",
                source_id="preset:" + job.name,
                runtime_names=["second"],
            )


@pytest.mark.asyncio
async def test_http_and_mcp_expose_list_and_empty_recipient_updates(tmp_path):
    async with preset(tmp_path) as (_, context, job, _):
        expected = len(context.components["connector_manager"]["default"].presets())
        service = HttpService(app_context=context, web_enabled=False, jobs=["manage_connector"])
        app = SimpleNamespace(config=context.app_config, context=context)
        service.build_service(app)
        service.add_jobs(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.service),
            base_url="http://test",
        ) as client:
            response = await client.post("/manage_connector", json={"action": "list"})
            assert response.status_code == 200 and response.json()["success"]
            assert response.json()["answer"]["total"] == 8
            response = await client.post(
                "/manage_connector",
                json={
                    "action": "update",
                    "source_id": "preset:" + job.name,
                    "runtime_names": ["second"],
                },
            )
            assert response.json()["success"] and response.json()["answer"]["runtime_names"] == ["second"]
        async with Client(service.mcp_server) as client:
            result = await client.call_tool("manage_connector", {"action": "list"})
            assert result.structured_content["total"] == expected
            result = await client.call_tool(
                "manage_connector",
                {
                    "action": "update",
                    "source_id": "preset:" + job.name,
                    "runtime_names": [],
                },
            )
            assert result.structured_content["runtime_names"] == [] and result.structured_content["enabled"]
