"""Isolated workspace and native-session boundaries for the current PA contracts."""

# pylint: disable=protected-access

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def pa_workspace(tmp_path):
    """Build PA fixtures lazily so packaging tests need no runtime dependencies."""
    from reme.components import ApplicationContext
    from reme.components.agent_wrapper import CodexAgentWrapper
    from reme.schema import SessionSnapshot, SessionStatus

    from .pa_helpers import bind_pa

    context = ApplicationContext(workspace_dir=str(tmp_path), metadata_dir="reme_metadata", language="zh")
    agent = CodexAgentWrapper(app_context=context, app_server="")
    runtime = bind_pa(context, agent)
    now = [datetime.fromisoformat("2026-11-08T12:00:00+08:00")]
    runtime._now = lambda: now[0]
    runtime._initialize_files()
    runtime.session_id = "session-1"
    runtime._write_json(
        runtime.session_ids_path,
        [
            {
                "session_id": runtime.session_id,
                "last_active_at": now[0].isoformat(),
                "last_read_message_id": None,
            },
        ],
    )
    snapshot = SessionSnapshot(
        session_id=runtime.session_id,
        state=SessionStatus(session_id=runtime.session_id, status="idle"),
        last_turn_id="turn-1",
    )
    agent.read_session = AsyncMock(return_value=snapshot)
    agent.get_session_status = AsyncMock(return_value=snapshot.state)
    agent.append_session = AsyncMock()
    return SimpleNamespace(context=context, agent=agent, runtime=runtime, snapshot=snapshot, now=now)
