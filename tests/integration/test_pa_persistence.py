"""Opt-in live SDK restart coverage; all state stays under pytest's temporary workspace."""

import asyncio
from contextlib import aclosing
from copy import deepcopy
import json
import os

import pytest

from reme.application import Application
from reme.config import resolve_app_config
from reme.schema import collect_reply

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CODEX_INTEGRATION") != "1",
    reason="set REME_CODEX_INTEGRATION=1 to run live Codex tests",
)


@pytest.mark.asyncio
async def test_live_empty_session_chat_and_proactive_history_survive_restarting_the_native_process(tmp_path):
    """Keep the same ID across three Applications, including an initially empty native thread."""
    config = resolve_app_config(config="proactive", log_config=False)
    config.update(
        workspace_dir=str(tmp_path),
        service={"backend": "cli"},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )
    config["components"] = {key: config["components"][key] for key in ("codex_app_server", "agent_wrapper", "pa")}
    config["jobs"] = {
        key: config["jobs"][key] for key in ("pa_session", "pa_new_session", "pa_chat", "pa_events", "session_read")
    }
    config["jobs"]["pa_chat"]["steps"][0]["prompt_dict"] = {
        "system_prompt": "This is an isolated persistence test. Reply with SESSION_RESTORED only. Do not use tools.",
        "system_prompt_zh": "这是隔离的持久化测试。只回复 SESSION_RESTORED，不使用工具。",
    }
    async with Application(**deepcopy(config)) as app:
        selected = (await app.run_job("pa_new_session")).answer
        runtime = app.context.components["pa"]["default"]
        assert json.loads(runtime.session_ids_path.read_text())[-1]["session_id"] == selected
        old_cursor = runtime.cursor
        agent = app.context.components["agent_wrapper"]["default"]
        assert not (await agent.read_session(selected)).messages

    async with Application(**deepcopy(config)) as app:
        assert (await app.run_job("pa_session")).answer == {"session_id": selected, "status": "not_loaded"}
        runtime = app.context.components["pa"]["default"]
        async with aclosing(runtime.events(old_cursor)) as events:
            assert (await anext(events)).type == "reset"
            assert (await anext(events)).snapshot.session_id == selected
        async with asyncio.timeout(180):
            reply = await collect_reply(app.run_stream_job("pa_chat", query="Confirm the restored session."))
        assert reply["status"] == "completed" and "SESSION_RESTORED" in reply["result"]
        assert (await app.run_job("pa_session")).answer["session_id"] == selected
        agent = app.context.components["agent_wrapper"]["default"]
        await agent.append_session(selected, "Persistent proactive reminder.", message_id="restart-test")
        history = await agent.read_session(selected)
        assert any(message.source == "proactive" for message in history.messages)

    async with Application(**deepcopy(config)) as app:
        assert (await app.run_job("pa_session")).answer["session_id"] == selected
        agent = app.context.components["agent_wrapper"]["default"]
        restored = await agent.read_session(selected)
        assert [message.id for message in restored.messages] == [message.id for message in history.messages]
        new_session = (await app.run_job("pa_new_session")).answer
        assert new_session != selected
        assert (
            json.loads(app.context.components["pa"]["default"].session_ids_path.read_text())[-1]["session_id"]
            == new_session
        )
        assert [message.id for message in (await agent.read_session(selected)).messages] == [
            message.id for message in history.messages
        ]
