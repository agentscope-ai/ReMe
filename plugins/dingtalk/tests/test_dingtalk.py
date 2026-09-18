"""Focused tests for the DingTalk background agent bridge."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import frontmatter
import httpx
import pytest
import yaml

from reme_dingtalk import DingTalkMarkdownSendStep
from reme_dingtalk import send as dingtalk_send
from reme_dingtalk.wait import DingTalkWaitStep, _session_key
from reme.components import ApplicationContext
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.runtime_context import RuntimeContext

PLUGIN_MANIFEST = yaml.safe_load(
    (Path(__file__).parents[1] / "src" / "reme_dingtalk" / "plugin.yaml").read_text(encoding="utf-8"),
)


def test_plugin_manifest_declares_only_dingtalk_backends():
    assert set(PLUGIN_MANIFEST) == {"backends"}
    assert PLUGIN_MANIFEST["backends"] == {
        "dingtalk_markdown_send_step": "reme_dingtalk.send:DingTalkMarkdownSendStep",
        "dingtalk_wait_step": "reme_dingtalk.wait:DingTalkWaitStep",
    }


@pytest.mark.asyncio
async def test_markdown_send_delivers_body_to_groups_in_order(tmp_path, monkeypatch):
    report = tmp_path / "daily" / "report.md"
    report.parent.mkdir()
    report.write_text(
        frontmatter.dumps(frontmatter.Post("# Report\n\nBody", name="Frontmatter title")),
        encoding="utf-8",
    )
    payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"processQueryKey": f"query-{len(payloads)}"})

    transport = httpx.MockTransport(handler)
    transport_kwargs = {}

    def ipv4_transport(**kwargs):
        transport_kwargs.update(kwargs)
        return transport

    dingtalk_stream = importlib.import_module("dingtalk_stream")
    monkeypatch.setattr(
        dingtalk_stream.DingTalkStreamClient,
        "get_access_token",
        lambda _client: "access-token",
    )
    monkeypatch.setattr(dingtalk_send.httpx, "AsyncHTTPTransport", ipv4_transport)
    step = DingTalkMarkdownSendStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        app_key="app-key",
        app_secret="app-secret",
        robot_code="robot-code",
        conversation_ids=" group-one,group-two ",
        title="Configured title",
    )
    step.logger = MagicMock()

    response = await step(RuntimeContext(markdown_path="daily/report.md"))

    assert transport_kwargs == {"local_address": "0.0.0.0"}
    assert [payload["openConversationId"] for payload in payloads] == ["group-one", "group-two"]
    assert [json.loads(payload["msgParam"]) for payload in payloads] == [
        {"title": "Configured title", "text": "# Report\n\nBody"},
    ] * 2
    assert response.metadata["dingtalk_configured_count"] == 2
    assert response.metadata["dingtalk_sent_count"] == 2
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert all(value not in logs for value in ("app-key", "app-secret", "robot-code", "group-one", "group-two"))


@pytest.mark.asyncio
async def test_markdown_send_without_conversations_is_a_noop(tmp_path):
    response = await DingTalkMarkdownSendStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
    )(RuntimeContext(markdown_path="missing.md"))

    assert response.success is True
    assert response.metadata == {
        "dingtalk_configured_count": 0,
        "dingtalk_sent_count": 0,
    }


@pytest.mark.asyncio
async def test_markdown_send_surfaces_dingtalk_rejection_detail(tmp_path, monkeypatch):
    report = tmp_path / "daily" / "report.md"
    report.parent.mkdir()
    report.write_text(frontmatter.dumps(frontmatter.Post("# Report\n\nBody")), encoding="utf-8")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "code": "InvalidmsgParam",
                "requestid": "01A0B51A-96E1-735D-A1A9-F043A281EC5B",
                "message": "Specified parameter msgParam is not valid.",
            },
        )

    dingtalk_stream = importlib.import_module("dingtalk_stream")
    monkeypatch.setattr(
        dingtalk_stream.DingTalkStreamClient,
        "get_access_token",
        lambda _client: "access-token",
    )
    monkeypatch.setattr(
        dingtalk_send.httpx,
        "AsyncHTTPTransport",
        lambda **kwargs: httpx.MockTransport(handler),
    )
    step = DingTalkMarkdownSendStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        app_key="app-key",
        app_secret="app-secret",
        robot_code="robot-code",
        conversation_ids="group-one",
    )
    step.logger = MagicMock()

    detail = (
        "HTTP 400 code=InvalidmsgParam"
        " message=Specified parameter msgParam is not valid."
        " requestid=01A0B51A-96E1-735D-A1A9-F043A281EC5B"
    )
    with pytest.raises(RuntimeError, match="code=InvalidmsgParam"):
        await step(RuntimeContext(markdown_path="daily/report.md"))

    assert step.context.response.metadata["dingtalk_sent_count"] == 0
    assert step.context.response.metadata["dingtalk_delivery_errors"] == [f"recipient 1: {detail}"]
    warnings = "\n".join(call.args[0] for call in step.logger.warning.call_args_list)
    assert f"recipient=1/1 {detail}" in warnings
    assert all(value not in warnings for value in ("app-key", "app-secret", "robot-code", "group-one"))


class _AgentWrapper(BaseAgentWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reply_calls = []
        self.compact_calls = []
        self.result_text = "回答"
        self.is_error = False

    async def compact_session(self, session_id):
        self.compact_calls.append(session_id)

    async def reply(self, inputs, **kwargs):
        self.reply_calls.append((inputs, kwargs))
        session_id = kwargs.get("resume") or "session-1"
        return {
            "session_id": session_id,
            "last_message": {"is_error": self.is_error},
            "result": self.result_text,
        }


class _Handler:
    def __init__(self):
        self.replies = []
        self.markdown_replies = []
        self.markdown_result = {"errcode": 0}

    def reply_text(self, text, _message):
        self.replies.append(text)

    def reply_markdown(self, title, text, _message):
        self.markdown_replies.append((title, text))
        return self.markdown_result


class _WebSocket:
    def __init__(self, messages=(), wait_when_empty=True):
        self.messages = list(messages)
        self.wait_when_empty = wait_when_empty
        self.closed = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if not self.wait_when_empty:
            raise StopAsyncIteration
        await self.closed.wait()
        raise StopAsyncIteration

    async def close(self):
        self.closed.set()


class _StreamClient:
    TAG_DISCONNECT = "disconnect"

    def __init__(self, route_result=""):
        self.route_result = route_result
        self.websocket = None

    def pre_start(self):
        return None

    def open_connection(self):
        return {"endpoint": "wss://example.test/connect", "ticket": "ticket"}

    async def keepalive(self, _websocket):
        await asyncio.Event().wait()

    async def route_message(self, _message):
        return self.route_result


def _message(text="hello", sender="user-1", conversation="cid-1", conversation_type="1"):
    return SimpleNamespace(
        text=SimpleNamespace(content=text),
        sender_staff_id=sender,
        conversation_id=conversation,
        conversation_type=conversation_type,
    )


def test_session_key_uses_conversation_type_id_and_sender():
    assert _session_key(_message()) == "1:cid-1:user-1"
    assert _session_key(_message(sender="user-2")) == "1:cid-1:user-2"
    assert _session_key(_message(conversation="cid-2", conversation_type="2")) == "2:cid-2:user-1"


@pytest.mark.asyncio
async def test_final_reply_resumes_session_and_clear_only_removes_combined_key(
    tmp_path,
):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _AgentWrapper(app_context=app_context)
    step = DingTalkWaitStep(app_context=app_context, agent_wrapper=wrapper)
    step.logger = MagicMock()
    handler = _Handler()
    sessions = {}
    message = _message()
    key = _session_key(message)

    await step._handle_message(message, key, sessions, handler)
    await step._handle_message(message, key, sessions, handler)

    assert sessions == {key: "session-1"}
    tool_kwargs = {"builtin_tools": False, "job_tools": []}
    assert wrapper.reply_calls == [
        ("hello", tool_kwargs),
        ("hello", {"resume": "session-1", **tool_kwargs}),
    ]
    assert handler.markdown_replies == [("ReMe Agent", "回答"), ("ReMe Agent", "回答")]

    await step._handle_message(_message(text="/compact"), key, sessions, handler)
    assert wrapper.compact_calls == ["session-1"]
    assert sessions[key] == "session-1"
    assert handler.replies[-1] == "✅ Conversation compaction requested."

    other_key = _session_key(_message(sender="user-2"))
    sessions[other_key] = "session-2"
    await step._handle_message(_message(text="/clear"), key, sessions, handler)
    assert sessions == {other_key: "session-2"}
    assert handler.replies[-1] == "✅ Conversation cleared. The next message will start a new session."

    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "received DingTalk text" in logs
    assert "completed DingTalk reply" in logs
    assert "handled session command" in logs
    assert "conversation_type='1' conversation_id='cid-1' sender_staff_id='user-1'" in logs
    assert all(value not in logs for value in ("hello", "session-1"))


@pytest.mark.asyncio
async def test_final_reply_rejects_empty_agent_reply_and_dingtalk_send_failure(
    tmp_path,
):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _AgentWrapper(app_context=app_context)
    step = DingTalkWaitStep(app_context=app_context, agent_wrapper=wrapper)
    step.logger = MagicMock()
    handler = _Handler()
    message = _message()
    key = _session_key(message)

    wrapper.result_text = " "
    with pytest.raises(ValueError, match="空回复"):
        await step._handle_message(message, key, {}, handler)

    wrapper.result_text = "回答"
    handler.markdown_result = None
    with pytest.raises(RuntimeError, match="发送钉钉 Markdown 回复失败"):
        await step._handle_message(message, key, {}, handler)


@pytest.mark.asyncio
async def test_final_reply_injects_only_configured_tools(tmp_path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _AgentWrapper(app_context=app_context)
    step = DingTalkWaitStep(
        app_context=app_context,
        agent_wrapper=wrapper,
        builtin_tools=["bash"],
        job_tools=["read", "write", "edit"],
    )

    message = _message()
    await step._handle_message(message, _session_key(message), {}, _Handler())

    assert wrapper.reply_calls == [
        (
            "hello",
            {
                "builtin_tools": ["bash"],
                "job_tools": ["read", "write", "edit"],
            },
        ),
    ]


@pytest.mark.asyncio
async def test_stream_client_closes_when_background_stop_is_set(monkeypatch):
    websocket = _WebSocket()
    monkeypatch.setattr("websockets.connect", lambda _uri: websocket)
    stop_event = asyncio.Event()
    task = asyncio.create_task(DingTalkWaitStep._run_client(_StreamClient(), stop_event))
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)
    assert websocket.closed.is_set()


@pytest.mark.asyncio
async def test_stream_client_restarts_after_server_disconnect(monkeypatch):
    websocket = _WebSocket(
        [
            json.dumps(
                {
                    "type": "SYSTEM",
                    "headers": {"topic": "disconnect"},
                    "data": json.dumps({"reason": "connection is expired"}),
                },
            ),
        ],
    )
    monkeypatch.setattr("websockets.connect", lambda _uri: websocket)
    reason = await DingTalkWaitStep._run_client(_StreamClient("disconnect"), asyncio.Event())
    assert reason == "connection is expired"


@pytest.mark.asyncio
async def test_stream_client_raises_when_websocket_closes_unexpectedly(monkeypatch):
    websocket = _WebSocket(wait_when_empty=False)
    monkeypatch.setattr("websockets.connect", lambda _uri: websocket)
    with pytest.raises(ConnectionError, match="closed unexpectedly"):
        await DingTalkWaitStep._run_client(_StreamClient(), asyncio.Event())


@pytest.mark.asyncio
async def test_stream_client_reconnects_after_server_request(monkeypatch, tmp_path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    step = DingTalkWaitStep(app_context=app_context)
    step.logger = MagicMock()
    stop_event = asyncio.Event()
    calls = 0

    async def run_client(_client, _stop_event):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "connection is expired"
        stop_event.set()
        return None

    async def timeout(awaitable, *, timeout):
        del timeout
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(step, "_run_client", run_client)
    monkeypatch.setattr(asyncio, "wait_for", timeout)

    await step._run_with_reconnect(_StreamClient(), stop_event)

    assert calls == 2
    step.logger.info.assert_called_once_with(
        "[DingTalkWaitStep] DingTalk server requested reconnect reason='connection is expired'; reconnecting in 1.0s",
    )
