"""Focused tests for the DingTalk background agent bridge."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reme.components import ApplicationContext, R
from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.config.config_parser import _load_config
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import StreamChunk
from reme.steps.cookbook.dingtalk.wait import DingTalkWaitStep, _CardRenderer, _session_key


class _AgentWrapper(BaseAgentWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def reply(self, inputs, **kwargs):
        raise NotImplementedError

    async def reply_stream(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        session_id = kwargs.get("resume") or "session-1"
        yield StreamChunk(
            chunk_type=ChunkEnum.THINK,
            chunk="检查上下文",
            block_id="think-1",
            session_id=session_id,
        )
        yield StreamChunk(chunk_type=ChunkEnum.THINK, chunk="", block_id="think-1", session_id=session_id)
        yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="回答", session_id=session_id)


class _Card:
    instances = []

    def __init__(self, _client, _message):
        self.card_template_id = "template"
        self.title = None
        self.markdown = ""
        self.finished = False
        self.failed = False
        self.updates = []
        self.stream_flags = []
        self.content = ""
        self.at_sender = False
        self.__class__.instances.append(self)

    def set_title_and_logo(self, title, _logo):
        self.title = title

    def get_card_data(self, flow_status=None):
        data = {"msgContent": self.markdown}
        if flow_status is not None:
            data["flowStatus"] = flow_status
        return data

    async def async_create_and_send_card(self, _template, _data, at_sender=False):
        self.at_sender = at_sender
        return "card-1"

    async def async_streaming(self, _card_id, _key, content, append, finished, failed):
        self.updates.append(content)
        self.stream_flags.append((append, finished, failed))
        self.content = self.content + content if append else content
        if finished:
            self.finished = not failed
            self.failed = failed


class _Handler:
    def __init__(self):
        self.dingtalk_client = object()
        self.replies = []

    def reply_text(self, text, _message):
        self.replies.append(text)


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


def test_renderer_keeps_blocks_in_stream_order_and_finishes_with_elapsed_time(monkeypatch):
    now = 10.0
    monkeypatch.setattr("reme.steps.cookbook.dingtalk.wait.time.monotonic", lambda: now)
    renderer = _CardRenderer()
    deltas = [
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="中间内容", block_id="content-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="", block_id="content-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.THINK, chunk="分析问题", block_id="think-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.THINK, chunk="", block_id="think-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="最终答案", block_id="answer-2")),
    ]

    markdown = "".join(deltas)
    assert "### 💬 Content\n\n中间内容" in markdown
    assert "### 🧠 Think\n\n分析问题" in markdown
    assert "### 💬 Answer" not in markdown
    assert markdown.count("### 💬 Content") == 2
    assert markdown.index("中间内容") < markdown.index("### 🧠 Think") < markdown.index("最终答案")

    now = 15.5
    markdown += renderer.finish()
    assert "### 💬 Answer" not in markdown
    assert "### 💬 Content\n\n最终答案" in markdown
    assert markdown.endswith("最终答案\n\n5.5s")
    assert all(value not in markdown for value in ("ReMe Agent", "question", "正在生成", "执行轨迹"))
    assert all(value not in markdown for value in ("✅", "### 当前"))


def test_renderer_limits_non_content_blocks_to_100_characters():
    renderer = _CardRenderer()
    deltas = [
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.THINK, chunk="a" * 101, block_id="think-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.THINK, chunk="", block_id="think-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.TOOL_RESULT, chunk="b" * 101, block_id="tool-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.TOOL_RESULT, chunk="", block_id="tool-1")),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="c" * 101, block_id="answer-1")),
        renderer.finish(),
    ]
    markdown = "".join(deltas)
    for character in "ab":
        assert f"{character * 100}..." in markdown
        assert character * 101 not in markdown
    assert "c" * 101 in markdown
    assert f"{'c' * 100}..." not in markdown


def test_renderer_keeps_tool_call_and_result_visible():
    renderer = _CardRenderer()
    tool_call_deltas = [
        renderer.feed(
            StreamChunk(
                chunk_type=ChunkEnum.TOOL_CALL,
                chunk='{"name":"Read","id":"tool-1"}',
                block_id="metadata-1",
                tool_call_name="Read",
            ),
        ),
        renderer.feed(
            StreamChunk(
                chunk_type=ChunkEnum.TOOL_CALL,
                chunk='{"path":',
                block_id="tool-1",
                tool_call_name="Read",
            ),
        ),
        renderer.feed(
            StreamChunk(chunk_type=ChunkEnum.TOOL_CALL, chunk='"memory.md"}', block_id="tool-1"),
        ),
        renderer.feed(StreamChunk(chunk_type=ChunkEnum.TOOL_CALL, chunk="", block_id="tool-1")),
    ]
    assert tool_call_deltas[:3] == ["", "", ""]

    deltas = [
        *tool_call_deltas,
        renderer.feed(
            StreamChunk(
                chunk_type=ChunkEnum.TOOL_RESULT,
                chunk={"content": '{"message":"memory content","count":1}'},
                block_id="tool-1",
            ),
        ),
        renderer.finish(),
    ]
    markdown = "".join(deltas)
    assert markdown.count("### 🔧 Tool Call") == 1
    assert "### 🔧 Tool Call · `Read`" in markdown
    assert '> {\n> 　"path": "memory.md"\n> }' in markdown
    assert "```" not in markdown
    assert '"name": "Read"' not in markdown
    assert "### 📦 Tool Result" in markdown
    assert '> 　"content": {\n> 　　"message": "memory content",\n> 　　"count": 1\n> 　}' in markdown


@pytest.mark.asyncio
async def test_messages_resume_session_and_clear_only_the_combined_key(tmp_path):
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _AgentWrapper(app_context=app_context)
    step = DingTalkWaitStep(app_context=app_context, agent_wrapper=wrapper, card_update_interval=0.05)
    step.logger = MagicMock()
    handler = _Handler()
    sdk = SimpleNamespace(
        AIMarkdownCardInstance=_Card,
        AICardStatus=SimpleNamespace(PROCESSING="1"),
    )
    sessions = {}
    message = _message()
    key = _session_key(message)

    await step._handle_message(message, key, sessions, handler, sdk)
    await step._handle_message(message, key, sessions, handler, sdk)

    assert sessions == {key: "session-1"}
    assert wrapper.calls == [("hello", {}), ("hello", {"resume": "session-1"})]
    assert all(card.finished for card in _Card.instances[-2:])
    assert all(card.stream_flags for card in _Card.instances[-2:])
    assert all(all(append for append, _finished, _failed in card.stream_flags) for card in _Card.instances[-2:])
    assert all("### 💬 Content\n\n回答" in card.content for card in _Card.instances[-2:])
    assert all(sum(update.count("检查上下文") for update in card.updates) == 1 for card in _Card.instances[-2:])
    assert all(sum(update.count("回答") for update in card.updates) == 1 for card in _Card.instances[-2:])
    assert all(card.at_sender for card in _Card.instances[-2:])
    assert all(card.title is None for card in _Card.instances[-2:])

    other_key = _session_key(_message(sender="user-2"))
    sessions[other_key] = "session-2"
    await step._handle_message(_message(text="/clear"), key, sessions, handler, sdk)
    assert sessions == {other_key: "session-2"}
    assert handler.replies[-1] == "✅ Conversation cleared. The next message will start a new session."
    logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
    assert "received DingTalk text" in logs
    assert "completed DingTalk reply" in logs
    assert "cleared DingTalk session" in logs
    assert "conversation_type='1' conversation_id='cid-1' sender_staff_id='user-1'" in logs
    assert all(value not in logs for value in ("hello", "session-1"))


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
            "card_update_interval": 1.0,
            "worker_count": 4,
        },
    ]
    dingtalk_wait = config["components"]["agent_wrapper"]["dingtalk_wait"]
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
    assert (step["app_key"], step["app_secret"], step["robot_code"]) == ("app-key", "app-secret", "robot-code")


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
