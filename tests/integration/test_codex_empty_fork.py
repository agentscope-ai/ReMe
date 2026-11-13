"""Opt-in native empty-thread coverage without model calls or real credentials."""

import os
from pathlib import Path

import pytest

from reme.application import Application

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CODEX_INTEGRATION") != "1",
    reason="set REME_CODEX_INTEGRATION=1 to run the native Codex process",
)


@pytest.mark.asyncio
async def test_live_empty_fork_and_first_proactive_message_preserve_source(tmp_path):
    """Exercise native start/fork/inject without ever starting a model turn."""
    async with Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components={
            "codex_app_server": {
                "default": {"backend": "sdk", "auth_mode": "api_key", "api_key": "unused-transport-test-key"},
            },
            "agent_wrapper": {"default": {"backend": "codex"}},
        },
    ) as app:
        agent = app.context.components["agent_wrapper"]["default"]
        server = app.context.components["codex_app_server"]["default"]
        source_id = await agent.create_session(system_prompt="Keep the original instructions.", job_tools=[])
        source = await agent.read_session(source_id)
        assert source.last_turn_id is None and source.messages == []
        assert source.path is None or not Path(source.path).exists()

        async with server.thread(source_id, fork=True) as fork:
            assert fork.id != source_id
            assert not (await fork.read(include_turns=False)).thread.turns
        assert (await agent.read_session(source_id)).messages == []

        message = await agent.append_session(source_id, "A first proactive message.", message_id="first")
        source = await agent.read_session(source_id)
        assert source.last_turn_id is None and source.messages == [message]
        async with server.thread(source_id, fork=True) as fork:
            inherited = await agent.read_session(fork.id)
            assert [item.content for item in inherited.messages] == [message.content]
            assert all(item.source == "proactive" for item in inherited.messages)
        assert (await agent.read_session(source_id)).messages == [message]
