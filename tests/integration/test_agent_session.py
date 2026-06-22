"""Integration tests: session state persistence and forking in AsAgentWrapper.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import sys
from pathlib import Path

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

            sid = "test-persist-session"

            # First call: establish session
            _, msg_1 = await wrapper.reply(
                "My favorite color is blue. Remember that.",
                session_id=sid,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            text_1 = (msg_1.get_text_content() or "").strip()
            print(f"\n[session_persist] reply 1: {text_1!r}")
            assert text_1, "Empty first reply"

            # Verify session file was created
            files_after_1 = env.session_state_files()
            print(f"[session_persist] session files after reply 1: {files_after_1}")
            assert len(files_after_1) == 1, f"Expected 1 session file, got {len(files_after_1)}"
            assert sid in files_after_1[0].name

            # Second call: same session_id — agent should have previous context
            _, msg_2 = await wrapper.reply(
                "What is my favorite color?",
                session_id=sid,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            text_2 = (msg_2.get_text_content() or "").strip()
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

            sid = "test-fork-origin"

            # Establish original session
            await wrapper.reply(
                "The secret number is 42.",
                session_id=sid,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            files_before_fork = env.session_state_files()
            assert len(files_before_fork) == 1

            # Fork the session
            forked_sid, msg_fork = await wrapper.reply(
                "What is the secret number?",
                session_id=sid,
                fork_session=True,
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            text_fork = (msg_fork.get_text_content() or "").strip()
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


async def _run_no_session_id() -> None:
    """When session_id is empty, no session file should be created."""
    with workspace_env() as env:
        app = await env.make_app()
        try:
            wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

            _, msg = await wrapper.reply(
                "Say hello.",
                system_prompt="You are a helpful assistant. Keep answers short.",
            )
            text = (msg.get_text_content() or "").strip()
            print(f"\n[no_session] reply: {text!r}")
            assert text, "Empty reply"

            files = env.session_state_files()
            assert len(files) == 0, f"No session files should be created without session_id, found {files}"

            print("✓ test_no_session_id passed")
        finally:
            await env.close_all()


async def _run_all() -> None:
    await _run_no_session_id()
    await _run_session_persistence()
    await _run_fork_session()


if __name__ == "__main__":
    print("=== Agent session state integration tests ===")
    asyncio.run(_run_all())
    print("\nAll integration tests passed!")
