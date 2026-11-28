"""Offline process-exit checks for proactive input receipts and reminder recovery."""

import asyncio
from datetime import timedelta
from pathlib import Path
import sys

import pytest

from unit.proactive_helpers import running_system
from unit.test_proactive_agent import install_round, prepare_round, REMIND


@pytest.mark.parametrize("receipt_saved", [False, True])
@pytest.mark.parametrize("record_enabled", [False, True])
@pytest.mark.asyncio
async def test_process_exit_retries_uncommitted_inputs_and_recovers_committed_results(
    tmp_path,
    monkeypatch,
    receipt_saved,
    record_enabled,
):
    """Exit at the durable receipt boundary, bypassing invocation and application cleanup."""
    script = """
import asyncio
import os
from pathlib import Path
import sys
from pytest import MonkeyPatch

sys.path.insert(0, str(Path.cwd() / "tests"))
from unit.proactive_helpers import running_system, queue_event
from unit.test_proactive_agent import install_round, prepare_round, REMIND

async def main():
    root = Path(sys.argv[1])
    receipt_saved = sys.argv[2] == "True"
    record_enabled = sys.argv[3] == "True"
    definition = {"agent_name": "monitor", "agent_type": "event"}
    if not record_enabled:
        definition["skills"] = []
    async with running_system(root, runtimes=[definition]) as (manager, context, wrapper):
        patch = MonkeyPatch()
        runtime, record = await prepare_round(manager, context, patch)
        await queue_event(manager)
        text = REMIND.format(priority="P1", body="\\nEvidence\\n") if record_enabled else None
        install_round(wrapper, runtime, record, patch, text=text)
        replace = runtime.inbox._replace

        def crash(path, raw):
            if path.name == "inbox_read.jsonl":
                if receipt_saved:
                    replace(path, raw)
                os._exit(23)
            replace(path, raw)

        patch.setattr(runtime.inbox, "_replace", crash)
        await context.jobs["run_proactive_agent"](agent_name="monitor")

asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(tmp_path),
        str(receipt_saved),
        str(record_enabled),
        cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(30):
            _, stderr = await process.communicate()
        assert process.returncode == 23, stderr.decode()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()

    async with running_system(tmp_path) as (manager, context, wrapper):
        runtime = await manager.resolve("monitor")
        chat = context.components["agent_runtime"]["Remy"]
        assert await runtime.inbox.pending_ids() == (set() if receipt_saved else {"event-1"})
        assert bool(await runtime.inbox.receipts()) is receipt_saved
        await runtime.recover_handoffs()
        assert len(await chat.inbox.pending_ids()) == int(receipt_saved and record_enabled)
        assert not wrapper.turns
        if not receipt_saved:
            now = runtime.now() + timedelta(seconds=1)
            monkeypatch.setattr(runtime, "now", lambda: now)
            _, record = await prepare_round(manager, context, monkeypatch)
            install_round(
                wrapper,
                runtime,
                record,
                monkeypatch,
                text=REMIND.format(priority="P1", body="\nEvidence\n") if record_enabled else None,
            )
            response = await context.jobs["run_proactive_agent"](agent_name="monitor")
            assert response.success, response.answer
            assert not await runtime.inbox.pending_ids()
            assert len(await chat.inbox.pending_ids()) == int(record_enabled)
        assert len(await runtime.inbox.receipts()) == 1
        deliveries = await runtime.outbox.entries()
        assert len(deliveries) == int(record_enabled)
        assert all(delivery.status == "delivered" for delivery in deliveries)
