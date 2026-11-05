"""Integration tests: session state persistence and forking in AsAgentWrapper.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import workspace_env  # noqa: E402

from reme.enumeration import ComponentEnum  # noqa: E402


async def _run_session_persistence() -> None:
    """Two consecutive replies with the same session_id should share context."""
    with workspace_env() as env:
        app = await env.make_app()
        try:
            wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

            sid = str(uuid4())

            # First call: establish session
            reply_1 = await wrapper.reply(
                "My favorite color is blue. Remember that.",
                session_id=sid,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            assert reply_1["status"] == "completed", reply_1["error"]
            text_1 = reply_1["result"].strip()
            print(f"\n[session_persist] reply 1: {text_1!r}")
            assert text_1, "Empty first reply"

            # Verify session file was created
            files_after_1 = env.session_state_files()
            print(f"[session_persist] session files after reply 1: {files_after_1}")
            assert len(files_after_1) == 1, f"Expected 1 session file, got {len(files_after_1)}"
            assert sid in files_after_1[0].name

            # Resume the returned session to retain its context.
            reply_2 = await wrapper.reply(
                "What is my favorite color?",
                resume=sid,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            assert reply_2["status"] == "completed", reply_2["error"]
            text_2 = reply_2["result"].strip()
            print(f"[session_persist] reply 2: {text_2!r}")
            assert "blue" in text_2.lower(), f"Agent should recall 'blue' from session context, got: {text_2!r}"

            print("✓ test_session_persistence passed")
        finally:
            await env.close_all()


async def _run_fork_session() -> None:
    """fork_session=True should create a new session file with a new session_id."""
    with workspace_env() as env:
        app = await env.make_app()
        try:
            wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

            sid = str(uuid4())

            # Establish original session
            await wrapper.reply(
                "The secret number is 42.",
                session_id=sid,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            files_before_fork = env.session_state_files()
            assert len(files_before_fork) == 1

            # Fork the session
            forked = await wrapper.reply(
                "What is the secret number?",
                resume=sid,
                fork_session=True,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            assert forked["status"] == "completed", forked["error"]
            forked_sid = forked["session_id"]
            text_fork = forked["result"].strip()
            print(f"\n[fork_session] forked reply: {text_fork!r}")
            assert "42" in text_fork, f"Forked session should recall '42', got: {text_fork!r}"

            # Verify: original file still exists + new forked file created
            files_after_fork = env.session_state_files()
            print(f"[fork_session] session files after fork: {[f.name for f in files_after_fork]}")
            assert (
                len(files_after_fork) == 2
            ), f"Expected 2 session files (original + fork), got {len(files_after_fork)}"

            # Forked session_id should differ from the original
            assert forked_sid != sid, f"Forked session_id should differ from original, got {forked_sid!r}"

            original_file = files_before_fork[0]
            assert original_file.exists(), "Original session file should still exist after fork"

            print("✓ test_fork_session passed")
        finally:
            await env.close_all()


async def _run_auto_session_id() -> None:
    """Omitting session_id creates and returns a persisted session."""
    with workspace_env() as env:
        app = await env.make_app()
        try:
            wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

            reply = await wrapper.reply(
                "Say hello.",
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            assert reply["status"] == "completed", reply["error"]
            text = reply["result"].strip()
            print(f"\n[no_session] reply: {text!r}")
            assert text, "Empty reply"

            files = env.session_state_files()
            assert len(files) == 1 and reply["session_id"] in files[0].name, files

            print("✓ test_auto_session_id passed")
        finally:
            await env.close_all()


async def _run_all() -> None:
    await _run_auto_session_id()
    await _run_session_persistence()
    await _run_fork_session()


if __name__ == "__main__":
    print("=== Agent session state integration tests ===")
    asyncio.run(_run_all())
    print("\nAll integration tests passed!")
