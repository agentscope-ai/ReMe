"""Exercise real subscription lifecycle and Inbox persistence with a mocked HTTP boundary."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastmcp import Client
import httpx
import pytest

from reme.components.proactive.connector import ConnectorManager
from reme.components.proactive.connector.sse import read_sse
from reme.components.service import HttpService
from reme.core.context import RuntimeContext
from reme.schema.cloud_event import CloudEvent
from reme.schema.proactive_connector import ManageConnectorRequest
from reme.steps.proactive.tools import ManageConnectorStep
from .proactive_helpers import running_system


def cloud_event(**changes):
    return {"specversion": "1.0", "id": "event-1", "source": "/experiment/a", "type": "experiment.progress", **changes}


def frame(value, cursor="cursor-1"):
    return f"id: {cursor}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n".encode()


def create_request(**changes):
    return {
        "action": "create",
        "name": "实验进度",
        "runtime_names": ["first"],
        "binding": {"type": "sse", "url": "https://events.example/stream"},
        **changes,
    }


class Parts(httpx.AsyncByteStream):
    """Keep a small event stream open to prove messages arrive before EOF or a full buffer."""

    def __init__(self, parts, *, keep_open=False):
        self.parts = parts
        self.keep_open = keep_open
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            yield part
        if self.keep_open:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def subscription_system(tmp_path, handler, *, seed=True):
    tasks = [
        {"agent_type": "event", "agent_name": "first"},
        {"agent_type": "event", "agent_name": "second"},
        {"agent_type": "time", "agent_name": "clock", "schedule": "0 8 * * *"},
    ]
    async with running_system(tmp_path, runtimes=tasks if seed else None) as (agents, context, _):
        connectors = ConnectorManager(app_context=context, reconnect_seconds=0.01)
        context.components["connector_manager"] = {"default": connectors}
        await connectors.start()
        original = connectors.client
        connectors.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        await original.aclose()
        try:
            yield agents, connectors, context, ManageConnectorStep(app_context=context)
        finally:
            await connectors.close()


async def wait_for(predicate):
    async with asyncio.timeout(5):
        while not await predicate():
            await asyncio.sleep(0.01)


async def cursor_is(connectors, source_id, cursor):
    return (await connectors.status(source_id))["last_event_id"] == cursor


@pytest.mark.asyncio
async def test_sse_decoder_handles_fragmented_utf8_multiline_crlf_and_controls():
    body = (
        '\ufeff: heartbeat\r\nid: 游标\r\nretry: 1500\r\ndata: {"text":\r\ndata: "中文\u2028text"}\r\n\r\nid:\r\n\r\n'
    )
    response = httpx.Response(200, stream=Parts([bytes([byte]) for byte in body.encode()]))
    values = [value async for value in read_sse(response, 4096)]
    assert len(values) == 2
    assert json.loads(values[0].data) == {"text": "中文\u2028text"}
    assert values[0].cursor == "游标" and values[0].retry_ms == 1500
    assert values[1].data is None and values[1].cursor == ""


@pytest.mark.asyncio
async def test_sse_decoder_discards_unterminated_blocks_and_bounds_lines():
    response = httpx.Response(200, stream=Parts([b"id: lost\ndata: not-complete\n"]))
    assert [value async for value in read_sse(response, 64)] == []
    for body in (b"data:" + b"x" * 65, b"data: " + b"x" * 40 + b"\ndata: " + b"y" * 40 + b"\n\n"):
        with pytest.raises(ValueError, match="exceeds"):
            _ = [value async for value in read_sse(httpx.Response(200, content=body), 64)]


@pytest.mark.asyncio
async def test_sse_comments_do_not_accumulate_as_pending_event_data():
    response = httpx.Response(200, stream=Parts([b": heartbeat\r\n"] * 20 + [b"data: ok\r\n\r\n"]))
    values = [value async for value in read_sse(response, 32)]
    assert len(values) == 1 and values[0].data == "ok"


@pytest.mark.asyncio
async def test_partial_source_updates_keep_identity_connection_and_historical_names(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        event = cloud_event(id=f"event-{len(requests)}")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=Parts([frame(event)], keep_open=True),
        )

    async with subscription_system(tmp_path, handler) as (agents, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]
        await wait_for(lambda: cursor_is(connectors, source_id, "cursor-1"))
        first = await agents.resolve("first")
        original = (await first.inbox.overview())["items"][0]["source_ref"]
        renamed = (await tool(action="update", source_id=source_id, name="Renamed")).answer
        assert renamed["name"] == "Renamed" and renamed["enabled"] and renamed["runtime_names"] == ["first"]
        assert len(requests) == 1
        assert (await first.inbox.overview())["items"][0]["source_ref"] == original
        paused = (
            await tool(
                action="update",
                source_id=source_id,
                name="Second name",
                runtime_names=["second"],
                enabled=False,
            )
        ).answer
        assert not paused["enabled"] and paused["runtime_names"] == ["second"]
        await tool(action="update", source_id=source_id, enabled=True)
        second = await agents.resolve("second")

        async def admitted():
            return bool(await second.inbox.pending_ids())

        await wait_for(admitted)
        source = (await second.inbox.overview())["items"][0]["source_ref"]
        assert source == {"source_id": source_id, "name": "Second name", "trigger_type": "event"}


def test_creation_uses_protocol_defaults_and_rejects_display_only_fields():
    request = create_request(binding={"url": "https://events.example/stream"})
    assert ManageConnectorRequest.model_validate(request).binding.type == "sse"
    with pytest.raises(ValueError):
        ManageConnectorRequest.model_validate({**request, "emoji": "news"})


@pytest.mark.parametrize(
    "changes",
    [
        {"specversion": "0.3"},
        {"id": ""},
        {"source": "bad source"},
        {"type": 42},
        {"time": "2026-11-27T10:00:00"},
        {"time": 42},
        {"time": "42"},
        {"source": "http://[broken"},
        {"DATA": "bad extension"},
        {"data": {}, "data_base64": "eA=="},
        {"data_base64": "%%%"},
        {"datacontenttype": "text/plain", "data": {}},
        {"datacontenttype": "text/plain", "data": None},
        {"extension": {}},
    ],
)
def test_cloud_events_reject_invalid_envelopes(changes):
    with pytest.raises(ValueError):
        CloudEvent.model_validate(cloud_event(**changes))


@pytest.mark.asyncio
async def test_live_create_delivery_deduplication_and_pause_close_stream(tmp_path):
    stream = Parts(
        [
            frame(cloud_event(data={"text": "中文\u2028正文"})),
            frame(cloud_event(data={"text": "中文\u2028正文"}), "cursor-2"),
            frame(cloud_event(source="/experiment/b"), "cursor-3"),
        ],
        keep_open=True,
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream; charset=utf-8"}, stream=stream)

    async with subscription_system(tmp_path, handler) as (agents, connectors, _, tool):
        source = (await tool(**create_request())).answer
        source_id = source["source_id"]
        await wait_for(lambda: cursor_is(connectors, source_id, "cursor-3"))
        runtime = await agents.resolve("first")
        assert len(await runtime.inbox.pending_ids()) == 2
        resources = list((tmp_path / "resource/cloudevents").glob("*.json"))
        assert len(resources) == 2
        saved = [json.loads(path.read_text())["cloud_event"] for path in resources]
        assert cloud_event(data={"text": "中文\u2028正文"}) in saved
        assert len(requests) == 1 and requests[0].method == "GET"
        assert requests[0].headers["Accept"] == "text/event-stream"
        listed = (await tool(action="list")).answer
        assert next(item for item in listed["items"] if item["source_id"] == source_id)["connected"]
        paused = (await tool(action="update", enabled=False, source_id=source_id)).answer
        assert not paused["enabled"] and not paused["connected"] and stream.closed
        assert not connectors.tasks
        assert len(await runtime.inbox.pending_ids()) == 2
    assert connectors.client is None


@pytest.mark.asyncio
async def test_restart_resumes_committed_cursor_and_retries_identical_event(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Parts(
                [
                    frame(cloud_event(), "durable-cursor"),
                ],
                keep_open=True,
            ),
        )

    async with subscription_system(tmp_path, handler) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]
        await wait_for(lambda: cursor_is(connectors, source_id, "durable-cursor"))
    async with subscription_system(tmp_path, handler, seed=False) as (agents, connectors, _, tool):

        async def reconnected():
            return len(requests) >= 2 and (await connectors.status(source_id))["connected"]

        await wait_for(reconnected)
        assert requests[-1].headers["Last-Event-ID"] == "durable-cursor"
        assert len(await (await agents.resolve("first")).inbox.pending_ids()) == 1
        await tool(action="update", enabled=False, source_id=source_id)
    async with subscription_system(tmp_path, handler, seed=False) as (_, connectors, _, tool):
        assert not connectors.tasks
        assert not next(item for item in (await tool(action="list")).answer["items"] if item["source_id"] == source_id)[
            "enabled"
        ]


@pytest.mark.asyncio
async def test_partial_delivery_is_durable_and_freezes_original_targets(tmp_path, monkeypatch):
    requests, calls = [], []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Parts(
                [
                    frame(cloud_event(), "pending-cursor"),
                ],
                keep_open=True,
            ),
        )

    async with subscription_system(tmp_path, handler) as (agents, connectors, _, tool):
        deliver = agents.deliver_events

        async def partially(names, events, **kwargs):
            calls.append(list(names))
            result = await deliver([name for name in names if name != "second"], events, **kwargs)
            result["errors"]["second"] = "temporary failure"
            return result

        monkeypatch.setattr(agents, "deliver_events", partially)
        source_id = (await tool(**create_request(runtime_names=["first", "second"]))).answer["source_id"]

        async def blocked():
            return (await connectors.status(source_id))["last_error"] is not None

        await wait_for(blocked)
        state = await connectors._read_state(source_id)
        assert state.cursor == "" and state.pending.remaining == ["second"]
        assert len(requests) == 1
        await tool(action="update", enabled=False, source_id=source_id)
        await tool(action="update", source_id=source_id, runtime_names=[])
        assert (await connectors._read_state(source_id)).pending.remaining == ["second"]
    async with subscription_system(tmp_path, handler, seed=False) as (agents, connectors, _, tool):
        deliver = agents.deliver_events
        calls.clear()

        async def recovered(names, events, **kwargs):
            calls.append(list(names))
            return await deliver(names, events, **kwargs)

        monkeypatch.setattr(agents, "deliver_events", recovered)
        await tool(action="update", enabled=True, source_id=source_id)
        await wait_for(lambda: cursor_is(connectors, source_id, "pending-cursor"))
        assert calls[0] == ["second"]
        assert len(await (await agents.resolve("first")).inbox.pending_ids()) == 1
        assert len(await (await agents.resolve("second")).inbox.pending_ids()) == 1
        await tool(action="update", enabled=False, source_id=source_id)


@pytest.mark.asyncio
async def test_failed_rejected_and_conflicting_events_do_not_advance_cursor(tmp_path, monkeypatch):
    streams = []

    def handler(_request):
        stream = Parts([frame(cloud_event(data={"value": len(streams)}))], keep_open=True)
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with subscription_system(tmp_path, handler) as (agents, connectors, _, tool):
        original = agents.deliver_events
        monkeypatch.setattr(
            agents,
            "deliver_events",
            AsyncMock(
                return_value={
                    "deliveries": {},
                    "errors": {},
                    "rejected": {"first": "removed"},
                },
            ),
        )
        source_id = (await tool(**create_request())).answer["source_id"]

        async def blocked():
            return (await connectors.status(source_id))["last_error"] is not None

        await wait_for(blocked)
        assert (await connectors.status(source_id))["last_event_id"] == ""
        await tool(action="update", enabled=False, source_id=source_id)
        monkeypatch.setattr(agents, "deliver_events", original)
        await tool(action="update", enabled=True, source_id=source_id)

        async def conflict():
            return "Conflicting CloudEvent" in ((await connectors.status(source_id))["last_error"] or "")

        await wait_for(conflict)
        assert (await connectors.status(source_id))["last_event_id"] == "cursor-1"
        assert len(await (await agents.resolve("first")).inbox.pending_ids()) == 1


@pytest.mark.asyncio
async def test_permissions_and_target_validation_precede_network_and_files(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(204)

    async with subscription_system(tmp_path, handler) as (_, connectors, _, tool):
        for values in (create_request(runtime_names=["clock"]), create_request(runtime_names=["missing"])):
            with pytest.raises(ValueError):
                await tool(**values)
        for allowed in (["metadata"], ["resource", "reme_agents"], [], ["../outside"]):
            with pytest.raises(PermissionError):
                await tool(RuntimeContext(_allowed_paths=allowed), **create_request())
        assert not connectors.directory.exists() and not requests
        source_id = (await tool(**create_request())).answer["source_id"]
        await tool(action="update", enabled=False, source_id=source_id)
        before = (connectors.source_directory(source_id) / "config.json").read_bytes()
        with pytest.raises(ValueError):
            await tool(action="update", source_id=source_id, runtime_names=["clock"])
        assert before == (connectors.source_directory(source_id) / "config.json").read_bytes()


@pytest.mark.parametrize(
    "values",
    [
        create_request(runtime_names=[]),
        create_request(runtime_names=["first", "FIRST"]),
        create_request(binding={"type": "sse", "url": "file:///etc/passwd"}),
        create_request(binding={"type": "sse", "url": "https://events.example", "format": "json"}),
        create_request(trigger_type=None),
        create_request(connector_job="preset"),
        {"action": "pause", "source_id": "../escape"},
        {"action": "enable", "source_id": "source_" + "a" * 32, "connector_job": "preset"},
        {"action": "send", "source_id": "source_" + "a" * 32, "runtime_names": ["first"]},
    ],
)
def test_subscription_requests_are_action_specific(values):
    with pytest.raises(ValueError):
        ManageConnectorRequest.model_validate(values)


@pytest.mark.asyncio
async def test_http_and_mcp_create_and_control_native_subscriptions(tmp_path):
    async with subscription_system(tmp_path, lambda _request: httpx.Response(204)) as (_, _, context, _):
        service = HttpService(app_context=context, web_enabled=False, jobs=["manage_connector"])
        app = SimpleNamespace(config=context.app_config, context=context)
        service.build_service(app)
        service.add_jobs(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.service),
            base_url="http://test",
        ) as client:
            response = await client.post("/manage_connector", json=create_request())
            assert response.status_code == 200 and response.json()["success"]
            source_id = response.json()["answer"]["source_id"]
        async with Client(service.mcp_server) as client:
            result = await client.call_tool(
                "manage_connector",
                {"action": "update", "enabled": False, "source_id": source_id},
            )
            assert result.structured_content["source_id"] == source_id
            assert not result.structured_content["enabled"]
            result = await client.call_tool(
                "manage_connector",
                {
                    "action": "update",
                    "source_id": source_id,
                    "runtime_names": ["second"],
                },
            )
            assert result.structured_content["runtime_names"] == ["second"]


@pytest.mark.parametrize(
    "response",
    [
        {"status_code": 503},
        {"status_code": 200, "headers": {"content-type": "application/json"}, "content": b"{}"},
        {"status_code": 200, "headers": {"content-type": "text/event-stream"}, "content": b"id: lost\ndata: {}\n\n"},
    ],
)
@pytest.mark.asyncio
async def test_invalid_http_or_event_response_keeps_cursor_and_surfaces_error(tmp_path, response):
    async with subscription_system(tmp_path, lambda _request: httpx.Response(**response)) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]

        async def failed():
            return bool((await connectors.status(source_id))["last_error"])

        await wait_for(failed)
        assert (await connectors.status(source_id))["last_event_id"] == ""
        assert not connectors.resource_root.exists()
        await tool(action="update", enabled=False, source_id=source_id)


@pytest.mark.asyncio
async def test_204_stops_reconnect_and_controls_retain_concurrent_changes(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(204)

    async with subscription_system(tmp_path, handler) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]

        async def stopped():
            return connectors.tasks[source_id].done()

        await wait_for(stopped)
        await asyncio.sleep(0.03)
        assert len(requests) == 1
        await asyncio.gather(
            ManageConnectorStep(app_context=tool.app_context)(action="update", enabled=False, source_id=source_id),
            ManageConnectorStep(app_context=tool.app_context)(
                action="update",
                source_id=source_id,
                runtime_names=["second"],
            ),
        )
        status = await connectors.status(source_id)
        assert not status["enabled"] and status["runtime_names"] == ["second"]
        await tool(action="update", enabled=True, source_id=source_id)
        await wait_for(stopped)
        assert len(requests) == 2


@pytest.mark.asyncio
async def test_target_update_affects_only_new_events(tmp_path):
    streams = []
    next_event = asyncio.Event()

    class ControlledStream(Parts):
        """Release the next event without closing the active connection."""

        async def __aiter__(self):
            yield frame(cloud_event(id="1"), "1")
            await next_event.wait()
            yield frame(cloud_event(id="2"), "2")
            await asyncio.Event().wait()

    def handler(_request):
        stream = ControlledStream([])
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with subscription_system(tmp_path, handler) as (agents, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]
        await wait_for(lambda: cursor_is(connectors, source_id, "1"))
        await tool(action="update", source_id=source_id, runtime_names=["second"])
        await tool(action="update", enabled=True, source_id=source_id)
        assert len(streams) == 1 and not streams[0].closed
        next_event.set()
        await wait_for(lambda: cursor_is(connectors, source_id, "2"))
        assert len(await (await agents.resolve("first")).inbox.pending_ids()) == 1
        assert len(await (await agents.resolve("second")).inbox.pending_ids()) == 1


@pytest.mark.asyncio
async def test_connection_recovery_clears_error_without_waiting_for_an_event(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("temporary outage", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Parts([b": heartbeat\n\n"], keep_open=True),
        )

    async with subscription_system(tmp_path, handler) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]

        async def recovered():
            return (await connectors.status(source_id))["connected"]

        await wait_for(recovered)
        assert (await connectors.status(source_id))["last_error"] is None


@pytest.mark.asyncio
async def test_update_does_not_overwrite_manual_source_edits(tmp_path):
    async with subscription_system(tmp_path, lambda _request: httpx.Response(204)) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]
        await tool(action="update", enabled=False, source_id=source_id)
        path = connectors.source_directory(source_id) / "config.json"
        value = json.loads(path.read_text())
        value["name"] = "Manually edited"
        path.write_text(json.dumps(value))
        before = path.read_bytes()
        with pytest.raises(ValueError, match="changed"):
            await tool(action="update", source_id=source_id, runtime_names=["second"])
        assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_failed_definition_write_keeps_the_existing_connection_and_targets(tmp_path, monkeypatch):
    stream = Parts([b": heartbeat\n\n"], keep_open=True)

    def handler(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with subscription_system(tmp_path, handler) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]

        async def connected():
            return (await connectors.status(source_id))["connected"]

        await wait_for(connected)

        def failed(*_args, **_kwargs):
            raise OSError("disk unavailable")

        monkeypatch.setattr(connectors.files, "write_json", failed)
        with pytest.raises(OSError, match="disk unavailable"):
            await tool(action="update", source_id=source_id, runtime_names=["second"])
        assert not stream.closed and (await connectors.status(source_id))["connected"]
        assert connectors.sources[source_id].runtime_names == ["first"]


@pytest.mark.asyncio
async def test_restart_ignores_unpublished_definitions_without_deleting_files(tmp_path):
    async with subscription_system(tmp_path, lambda _request: httpx.Response(204)) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]
        await tool(action="update", enabled=False, source_id=source_id)
        orphan = connectors.source_directory("source_" + "a" * 32)
        orphan.mkdir()
        temporary = orphan / ".agent-interrupted.tmp"
        temporary.write_text("unfinished definition")
    async with subscription_system(tmp_path, lambda _request: httpx.Response(204), seed=False) as (_, _, _, tool):
        assert (await tool(action="list")).answer["total"] == len(connectors.presets()) + 1
        assert temporary.read_text() == "unfinished definition"


@pytest.mark.parametrize(
    "options",
    [
        {"reconnect_seconds": float("nan")},
        {"reconnect_seconds": float("inf")},
        {"reconnect_seconds": True},
        {"reconnect_seconds": 0},
        {"max_event_bytes": True},
        {"max_event_bytes": 1.5},
    ],
)
def test_connector_limits_reject_nonfinite_or_noninteger_values(options):
    with pytest.raises(ValueError):
        ConnectorManager(**options)


def test_explicit_null_data_is_distinct_from_an_absent_payload():
    assert (
        CloudEvent.model_validate(cloud_event(data=None)).canonical()
        != CloudEvent.model_validate(cloud_event()).canonical()
    )


@pytest.mark.parametrize(
    "original,replay",
    [
        (cloud_event(), cloud_event(time=None, subject=None, extension=None)),
        (
            cloud_event(time="2026-11-27T10:00:00Z"),
            cloud_event(time="2026-11-27T12:00:00+02:00"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_equivalent_cloud_events_keep_original_resource_and_event_time(tmp_path, original, replay):
    def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Parts([frame(original, "1"), frame(replay, "2")], keep_open=True),
        )

    async with subscription_system(tmp_path, handler) as (agents, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]
        await wait_for(lambda: cursor_is(connectors, source_id, "2"))
        assert len(await (await agents.resolve("first")).inbox.pending_ids()) == 1
        resources = list(connectors.resource_root.glob("*.json"))
        assert len(resources) == 1 and json.loads(resources[0].read_text())["cloud_event"] == original


@pytest.mark.asyncio
async def test_symlink_resource_root_fails_before_creating_network_resources(tmp_path):
    async with running_system(tmp_path) as (agents, context, _):
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "resource").symlink_to(outside, target_is_directory=True)
        connectors = ConnectorManager(app_context=context, agent_manager=agents)
        with pytest.raises(ValueError, match="without symlinks"):
            await connectors.start()
        assert connectors.closed and connectors.client is None and not connectors.tasks
        assert not list(outside.iterdir())


@pytest.mark.asyncio
async def test_redirected_stream_and_empty_network_errors_are_reported(tmp_path):
    def handler(request):
        if request.url.path == "/stream":
            return httpx.Response(307, headers={"location": "/redirected"})
        raise httpx.ReadTimeout("", request=request)

    async with subscription_system(tmp_path, handler) as (_, connectors, _, tool):
        source_id = (await tool(**create_request())).answer["source_id"]

        async def failed():
            return (await connectors.status(source_id))["last_error"] == "ReadTimeout"

        await wait_for(failed)
        assert (await connectors.status(source_id))["last_event_id"] == ""
