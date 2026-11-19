"""Opt-in real Claude SDK session coverage using a configured Anthropic-compatible endpoint.

REME_CC_INTEGRATION=1 REME_CC_ENV_FILE=/path/to/.env REME_CC_BASE_URL=https://... pytest -s <this file>
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from reme.components.agent_wrapper import CcAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.utils.env_utils import parse_env_file

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CC_INTEGRATION") != "1",
    reason="set REME_CC_INTEGRATION=1 to run real Claude SDK/model tests",
)


@pytest.mark.asyncio
async def test_live_cc_empty_sessions_both_append_roles_restart_and_resume(tmp_path):
    """Prove native persistence and context visibility through the production wrapper."""
    config = parse_env_file(Path(os.environ["REME_CC_ENV_FILE"]))
    options = {
        "api_key": config["LLM_API_KEY"],
        "base_url": os.getenv("REME_CC_BASE_URL") or config["LLM_BASE_URL"],
        "model": os.getenv("REME_CC_MODEL") or config.get("LLM_MODEL_NAME") or "qwen3.7-plus",
        "tools": [],
        "permission_mode": "bypassPermissions",
        "thinking": {"type": "disabled"},
        "env": {
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "256",
            "API_TIMEOUT_MS": "30000",
        },
    }
    first = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)), **options)
    async with first:
        session_id = await first.create_session()
        assert await first.session_exists(session_id)
        assert (await first.read_session(session_id)).messages == []
        user_label, assistant_label = f"notebook-{uuid4().hex[:12]}", f"room-{uuid4().hex[:12]}"
        user_marker = f"Fictional public note: the notebook label is {user_label}."
        assistant_marker = f"Fictional public note: the meeting-room label is {assistant_label}."
        user = await first.append_session(session_id, user_marker, message_id="user", role="user")
        assistant = await first.append_session(session_id, assistant_marker, message_id="assistant")
        snapshot = await first.read_session(session_id)
        assert snapshot.messages == [user, assistant]
        assert snapshot.state.status == "idle"
        path = await first.get_session_path(session_id)
        original = path.read_bytes()
    reloaded = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)), **options)
    async with reloaded:
        assert await reloaded.append_session(session_id, user_marker, message_id="user", role="user") == user
        assert await reloaded.append_session(session_id, assistant_marker, message_id="assistant") == assistant
        assert path.read_bytes() == original
        result = await reloaded.reply(
            "What notebook label and meeting-room label were given in the earlier fictional public notes? "
            "Return both exact labels separated by a space, with no explanation.",
            resume=session_id,
        )
        assert result["status"] == "completed", result["error"]
        assert user_label in result["result"] and assistant_label in result["result"]
        assert result["usage"]["input_tokens"] > 0
        snapshot = await reloaded.read_session(session_id)
        assert snapshot.messages[:2] == [user, assistant]
        assert any(
            user_label in message.content and assistant_label in message.content for message in snapshot.messages[2:]
        )
        assert snapshot.last_turn_id and snapshot.state.status == "idle"
        with pytest.raises(RuntimeError, match="changed"):
            await reloaded.append_session(session_id, "stale", message_id="guard", expected_turn_id="previous-turn")
        forked = await reloaded.reply(
            "What notebook label and meeting-room label were given in the earlier fictional public notes? "
            "Return both exact labels separated by a space, with no explanation.",
            resume=session_id,
            fork_session=True,
        )
        assert forked["status"] == "completed", forked["error"]
        assert forked["session_id"] != session_id
        assert user_label in forked["result"] and assistant_label in forked["result"]
        assert (await reloaded.read_session(forked["session_id"])).messages[:2] == [user, assistant]
        assert await reloaded.append_session(forked["session_id"], user_marker, message_id="user", role="user") == user
        await reloaded.compact_session(session_id)
        compacted = await reloaded.read_session(session_id)
        assert compacted.messages[:2] == [user, assistant]
        assert any(message.content == result["result"] for message in compacted.messages[2:])
        await reloaded.release_session(session_id)
        assert await reloaded.session_exists(session_id)
    print(
        {"model": options["model"], "status": result["status"], "usage": result["usage"], "both_roles_recalled": True},
    )
