"""Focused tests for the DingTalk background agent bridge."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reme.components import ApplicationContext, R
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.config.config_parser import _load_config
from reme.enumeration import ComponentEnum
from reme.steps.cookbook.dingtalk.wait import DingTalkWaitStep, _session_key


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
    def __init__(self, messages=()):
        self.messages = list(messages)
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
    assert wrapper.reply_calls == [("hello", {}), ("hello", {"resume": "session-1"})]
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


def test_daily_cookbook_registers_one_step_background_wait_job(monkeypatch):
    for name in ("DINGTALK_APP_KEY", "DINGTALK_APP_SECRET", "DINGTALK_ROBOT_CODE"):
        monkeypatch.delenv(name, raising=False)
    config = _load_config("daily_cookbook")
    job = config["jobs"]["dingtalk_wait"]
    assert job["backend"] == "background"
    assert job["steps"] == [
        {
            "backend": "dingtalk_wait_step",
            "agent_wrapper": "dingtalk_wait",
            "app_key": "",
            "app_secret": "",
            "robot_code": "",
            "worker_count": 4,
        },
    ]
    dingtalk_wait = config["components"]["agent_wrapper"]["dingtalk_wait"]
    assert dingtalk_wait["skills"] == ["tushare-data"]
    assert dingtalk_wait["job_tools"] == ["search"]
    assert dingtalk_wait["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
        "append": (
            "Daily-paper Markdown is stored under the ReMe workspace. Detailed notes, including historical notes, "
            "are at daily/YYYY-MM-DD/paper-<arxiv-id>.md; daily briefs are at "
            "daily/YYYY-MM-DD/daily-paper-brief.md. Use search to recall relevant notes across dates."
        ),
    }
    assert R.get(ComponentEnum.STEP, "dingtalk_wait_step") is DingTalkWaitStep


def test_daily_cookbook_passes_dingtalk_environment_to_step(monkeypatch):
    monkeypatch.setenv("DINGTALK_APP_KEY", "app-key")
    monkeypatch.setenv("DINGTALK_APP_SECRET", "app-secret")
    monkeypatch.setenv("DINGTALK_ROBOT_CODE", "robot-code")
    step = _load_config("daily_cookbook")["jobs"]["dingtalk_wait"]["steps"][0]
    assert (step["app_key"], step["app_secret"], step["robot_code"]) == (
        "app-key",
        "app-secret",
        "robot-code",
    )


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
    websocket = _WebSocket(["{}"])
    monkeypatch.setattr("websockets.connect", lambda _uri: websocket)
    with pytest.raises(ConnectionError, match="WebSocket closed"):
        await DingTalkWaitStep._run_client(_StreamClient("disconnect"), asyncio.Event())
