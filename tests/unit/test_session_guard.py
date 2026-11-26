"""Guarded decisions serialize native writers without invoking real models."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import asyncio

import pytest

from . import test_agent_session_api, test_cc_agent_sessions
from .test_as_active_capabilities import make_wrapper

runtime = test_agent_session_api.runtime
codex_wrapper = test_agent_session_api.codex_wrapper
cc_wrapper = test_cc_agent_sessions.wrapper


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["as", "cc", "codex"])
@pytest.mark.parametrize("delivery", ["idle", "immediate"])
async def test_decision_guard_blocks_native_appends_and_rejects_stale_revision(
    tmp_path,
    cc_wrapper,
    codex_wrapper,
    backend,
    delivery,
):
    if backend == "as":
        wrapper = make_wrapper(tmp_path / "as")
        session = await wrapper.create_session()
    elif backend == "cc":
        wrapper = cc_wrapper
        session = await wrapper.create_session()
    else:
        wrapper, _ = codex_wrapper
        session = "session-1"
    snapshot = await wrapper.read_session(session)
    append = None
    try:
        async with asyncio.timeout(3):
            async with wrapper.session_guard(session, expected_revision=snapshot.revision):
                append = asyncio.create_task(
                    wrapper.append_session(session, "New input", message_id="new", role="user", delivery=delivery),
                )
                await asyncio.sleep(0.02)
                assert not append.done()
                assert (await wrapper.read_session(session)).revision == snapshot.revision
            await append
            with pytest.raises(RuntimeError, match="Session changed"):
                async with wrapper.session_guard(session, expected_revision=snapshot.revision):
                    raise AssertionError("A stale decision must not commit")
    finally:
        if append is not None:
            append.cancel()
            await asyncio.gather(append, return_exceptions=True)
        if backend == "as":
            await wrapper.close()
