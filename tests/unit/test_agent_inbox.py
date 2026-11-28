"""Single-owner queues: durable receipts, bounded claims and restart safety."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Event as ThreadEvent

import pytest

from reme.components.proactive.inbox import EventInbox, NotificationInbox, InvalidInboxBatchError
from reme.core.context import ApplicationContext
from reme.schema import Event, NotificationRef


def event(identifier="event-1", **values):
    return Event(
        id=identifier,
        event_ts=datetime(2026, 11, 20, tzinfo=timezone.utc),
        source="test",
        description="Observed",
        **values,
    )


@asynccontextmanager
async def make_inbox(tmp_path, cls=EventInbox, **kwargs):
    inbox = cls(directory="inbox", app_context=ApplicationContext(workspace_dir=str(tmp_path)), **kwargs)
    async with inbox:
        yield inbox


@pytest.mark.asyncio
async def test_claim_is_exclusive_bounded_ordered_and_nack_preserves_inputs(tmp_path):
    async with make_inbox(tmp_path, claim_max_items=2) as inbox:
        await inbox.append([event("c"), event("a"), event("b")])
        batch = await inbox.claim()
        assert [item["id"] for item in batch["items"]] == ["a", "b"]
        assert await inbox.claim() is None
        assert (await inbox.snapshot())["in_flight_count"] == 2
        await inbox.nack(batch["token"])
        selected = await inbox.claim(item_ids=["c"])
        await inbox.ack(selected["token"], metadata={"result": "saved"})
        assert await inbox.pending_ids() == {"a", "b"}
        assert (await inbox.receipts())[-1].metadata == {"result": "saved"}
        with pytest.raises(ValueError):
            await inbox.ack(selected["token"])


@pytest.mark.asyncio
async def test_same_id_is_idempotent_and_conflicting_batch_is_atomic(tmp_path):
    async with make_inbox(tmp_path) as inbox:
        await inbox.append(event())
        await inbox.append(event())
        with pytest.raises(InvalidInboxBatchError):
            await inbox.append([event("fresh"), event().model_copy(update={"description": "changed"})])
        assert await inbox.pending_ids() == {"event-1"}
        batch = await inbox.claim()
        await inbox.ack(batch["token"])
    async with make_inbox(tmp_path) as inbox:
        await inbox.append(event())
        assert not await inbox.pending_ids()
        with pytest.raises(InvalidInboxBatchError):
            await inbox.append(event().model_copy(update={"description": "changed"}))


@pytest.mark.asyncio
async def test_pending_capacity_waits_until_durable_ack(tmp_path):
    async with make_inbox(tmp_path, max_pending_items=1, append_timeout_seconds=0.2) as inbox:
        await inbox.append(event("a"))
        with pytest.raises(TimeoutError):
            await inbox.append(event("b"))
        waiter = asyncio.create_task(inbox.append(event("b"), wait=True))
        await asyncio.sleep(0.01)
        assert not waiter.done()
        batch = await inbox.claim()
        await inbox.ack(batch["token"])
        assert await waiter == ["b"]
        assert await inbox.pending_ids() == {"b"}


@pytest.mark.asyncio
async def test_event_history_window_and_retained_handoffs(tmp_path):
    async with make_inbox(tmp_path, max_read_items=1) as inbox:
        await inbox.append(event("a"))
        batch = await inbox.claim()
        await inbox.ack(batch["token"], retain=True)
        await inbox.append(event("b"))
        batch = await inbox.claim()
        with pytest.raises(TimeoutError, match="handoffs"):
            await inbox.ack(batch["token"])
        await inbox.retain(["a"], release=True)
        await inbox.ack(batch["token"])
        assert (await inbox.read(state="processed"))["total"] == 1
        await inbox.append(event("a"))
        assert await inbox.pending_ids() == {"a"}


@pytest.mark.asyncio
@pytest.mark.parametrize("retain", [False, True])
async def test_claim_fits_history_window_and_preserves_receipt_after_restart(tmp_path, retain):
    async with make_inbox(tmp_path, max_read_items=1) as inbox:
        await inbox.append([event("a"), event("b")])
        batch = await inbox.claim()
        assert [item["id"] for item in batch["items"]] == ["a"]
        await inbox.ack(batch["token"], retain=retain)
    async with make_inbox(tmp_path, max_read_items=1) as inbox:
        assert (await inbox.receipts())[0].item_ids == ["a"]
        await inbox.append(event("a"))
        assert await inbox.pending_ids() == {"b"}


@pytest.mark.asyncio
async def test_notification_receipts_survive_display_limit_and_restart(tmp_path):
    refs = [
        NotificationRef.from_record("monitor", f"daily/2026-11-20/monitor_{index:06}.md", datetime.now(timezone.utc))
        for index in range(4)
    ]
    async with make_inbox(tmp_path, NotificationInbox, max_read_items=1) as inbox:
        for ref in refs:
            await inbox.append(ref)
            batch = await inbox.claim()
            await inbox.ack(batch["token"], metadata={"decision": "skip"})
        assert len(await inbox.receipts()) == 4
    async with make_inbox(tmp_path, NotificationInbox, max_read_items=1) as inbox:
        await inbox.append(refs)
        assert not await inbox.pending_ids()
        for ref in refs:
            assert await inbox.processed(ref.id)


@pytest.mark.asyncio
async def test_read_ledger_wins_after_crash_before_unread_rewrite(tmp_path, monkeypatch):
    async with make_inbox(tmp_path) as inbox:
        await inbox.append(event())
        batch = await inbox.claim()
        replace = inbox._replace

        def fail(path, raw):
            if path.name == "inbox_unread.jsonl":
                raise OSError("crash after read receipt")
            replace(path, raw)

        monkeypatch.setattr(inbox, "_replace", fail)
        with pytest.raises(OSError):
            await inbox.ack(batch["token"])
        with pytest.raises(RuntimeError):
            await inbox.append(event("next"))
    async with make_inbox(tmp_path) as inbox:
        assert not await inbox.pending_ids()
        assert (await inbox.receipts())[-1].item_ids == ["event-1"]


@pytest.mark.asyncio
async def test_cancellation_finishes_durable_append_before_releasing_lock(tmp_path, monkeypatch):
    async with make_inbox(tmp_path) as inbox:
        entered, release = ThreadEvent(), ThreadEvent()
        write = inbox._write

        def delayed(*args):
            entered.set()
            release.wait(2)
            write(*args)

        monkeypatch.setattr(inbox, "_write", delayed)
        task = asyncio.create_task(inbox.append(event()))
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await inbox.pending_ids() == {"event-1"}


@pytest.mark.asyncio
async def test_single_writer_and_symlink_containment(tmp_path, tmp_path_factory):
    async with make_inbox(tmp_path):
        with pytest.raises((OSError, RuntimeError, ValueError)):
            async with make_inbox(tmp_path):
                pass
    outside = tmp_path_factory.mktemp("outside-inbox")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        EventInbox(directory="escape", app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    assert not list(outside.iterdir())


@pytest.mark.asyncio
async def test_unterminated_tail_recovery_and_corruption_fail_closed(tmp_path):
    async with make_inbox(tmp_path) as inbox:
        await inbox.append(event())
    path = tmp_path / "inbox/inbox_unread.jsonl"
    with path.open("ab") as file:
        file.write(b'{"id":"partial')
    async with make_inbox(tmp_path) as inbox:
        assert await inbox.pending_ids() == {"event-1"}
        await inbox.append(event("next"))
    path.write_text('{"broken":true}\n')
    with pytest.raises(ValueError):
        async with make_inbox(tmp_path):
            pass
