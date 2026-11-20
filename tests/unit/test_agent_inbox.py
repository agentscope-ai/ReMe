"""Durable Agent inboxes: recovery, bounded buffering and cancellation-safe commits."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
import json
from threading import Event as ThreadEvent

import pytest
import pytest_asyncio
from pydantic import ValidationError

from reme import Application
from reme.components import AgentEventInbox, LocalAgentInbox
from reme.core.registry import create_application_registry
from reme.core.context import ApplicationContext
from reme.config import resolve_app_config
from reme.schema import Event


def event(identifier="e1", **kwargs):
    return {
        "id": identifier,
        "event_ts": "2026-11-11T14:30:00+08:00",
        "source": "news",
        "description": "A new observation",
        "paths": ["resource/news/report.md"],
        "metadata": {"url": "https://example.com/news", "tags": ["work"]},
        **kwargs,
    }


def make_inbox(tmp_path, **kwargs):
    return LocalAgentInbox(app_context=ApplicationContext(workspace_dir=str(tmp_path)), **kwargs)


@pytest_asyncio.fixture
async def inbox(tmp_path):
    async with make_inbox(tmp_path) as result:
        await result.create("news")
        yield result


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


async def consume(inbox, name="news", limit=None):
    batch = await inbox.claim(name, limit)
    await inbox.ack(name, batch["token"])
    return batch["items"]


@pytest.mark.asyncio
async def test_registry_config_and_application_lifecycle(tmp_path):
    assert create_application_registry().get("agent_inbox", "local") is LocalAgentInbox
    assert "agent_inbox" not in resolve_app_config(config="default")["components"]
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
        service={"backend": "cli"},
        components={"agent_inbox": {"default": {"backend": "local"}}},
        jobs={},
    )
    async with app:
        component = app.context.components["agent_inbox"]["default"]
        assert component.max_read_items == 1000
        await component.create("news")
        await component.append("news", event())
        assert (await consume(component))[0]["id"] == "e1"
    assert not component.is_started


@pytest.mark.asyncio
async def test_mount_restores_sources_and_forget_rejects_pending_work(tmp_path):
    async with make_inbox(tmp_path) as component:
        await component.mount("news", "agent_inbox/news")
        await component.mount("news", "agent_inbox/news")
        await component.append("news", event())
        with pytest.raises(ValueError, match="pending"):
            await component.forget("news")
        await consume(component)
        await component.forget("news")
        await component.mount("news", "agent_inbox/news")
        snapshot = await component.snapshot("news")
        assert snapshot["pending_count"] == 0
        assert snapshot["receipt"].item_ids == ["e1"]
        assert (await component.receipts())["news"][0].item_ids == ["e1"]
        with pytest.raises(ValueError, match="layout"):
            await component.mount("news", "other/news")


@pytest.mark.asyncio
async def test_result_handoffs_use_generic_retention_without_notification_coupling(tmp_path):
    async with make_inbox(tmp_path, max_read_items=1, append_timeout_seconds=0.02) as component:
        await component.create("news")
        await component.append("news", event())
        batch = await component.claim("news")
        await component.ack("news", batch["token"], metadata={"artifact": "daily/result.md"}, retain=True)
        await component.append("news", event("e2"))
        blocked = await component.claim("news")
        with pytest.raises(TimeoutError, match="handoffs"):
            await component.ack("news", blocked["token"])
        await component.nack("news", blocked["token"])
        with pytest.raises(ValueError, match="handoffs"):
            await component.forget("news")
        await component.retain("news", ["e1"], release=True)
        await component.append("news", event("e2"))
        assert (await component.snapshot("news"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_releasing_old_receipts_preserves_new_concurrent_handoffs(tmp_path):
    async with make_inbox(tmp_path, max_read_items=2, append_timeout_seconds=0.02) as component:
        await component.create("news")
        for identifier in ("e1", "e2"):
            await component.append("news", event(identifier))
            batch = await component.claim("news")
            await component.ack("news", batch["token"], retain=True)
        await component.retain("news", ["e1"], release=True)
        await component.append("news", event("e3"))
        await consume(component)
        await component.append("news", event("e4"))
        batch = await component.claim("news")
        with pytest.raises(TimeoutError, match="handoffs"):
            await component.ack("news", batch["token"])
        assert (await component.receipts())["news"][0].item_ids == ["e2"]
        await component.retain("news", ["e2"], release=True)
        await component.ack("news", batch["token"])
        assert [r.item_ids for r in (await component.receipts())["news"]] == [["e3"], ["e4"]]


@pytest.mark.asyncio
async def test_event_and_receipt_are_separate_and_claim_is_not_consumption(inbox, tmp_path):
    source = event()
    assert await inbox.append("news", source) == ["e1"]
    source["metadata"]["tags"].append("mutated")
    batch = await inbox.claim("news")
    assert batch["items"][0]["metadata"]["tags"] == ["work"]
    batch["items"][0]["metadata"]["tags"].append("also mutated")
    directory = tmp_path / "agent_inbox/news"
    persisted = lines(directory / "inbox_unread.jsonl")
    assert set(persisted[0]) == {"id", "event_ts", "source", "description", "paths", "metadata"}
    assert persisted[0]["metadata"]["tags"] == ["work"]
    assert not (directory / "inbox_read.jsonl").exists()
    assert await inbox.claim("news") is None
    summary = (await inbox.list())[0]
    assert summary["pending_count"] == summary["in_flight_count"] == 1
    assert summary["pending_bytes"] == (directory / "inbox_unread.jsonl").stat().st_size
    await inbox.nack("news", batch["token"])
    batch = await inbox.claim("news")
    assert batch["items"] == persisted
    timestamp = await inbox.ack("news", batch["token"])
    assert lines(directory / "inbox_read.jsonl") == [
        {"items": persisted, "consume_ts": timestamp.replace("+00:00", "Z"), "metadata": {}},
    ]
    assert lines(directory / "inbox_unread.jsonl") == []
    assert (await inbox.list())[0]["pending_count"] == 0


@pytest.mark.asyncio
async def test_recovery_restores_unacked_claims_and_empty_inboxes(inbox, tmp_path):
    await inbox.create("empty")
    await inbox.append("news", [event("e1"), event("e2")])
    await consume(inbox, limit=1)
    abandoned = await inbox.claim("news")
    await inbox.close()
    async with make_inbox(tmp_path) as restored:
        assert [item["name"] for item in await restored.list()] == ["empty", "news"]
        with pytest.raises(ValueError, match="token"):
            await restored.ack("news", abandoned["token"])
        assert [item["id"] for item in await consume(restored)] == ["e2"]
        assert await restored.claim("empty") is None


@pytest.mark.asyncio
async def test_idempotent_append_and_conflicting_ids(inbox, tmp_path):
    await inbox.append("news", [event(), event()])
    reordered = event(metadata={"tags": ["work"], "url": "https://example.com/news"})
    assert await inbox.append("news", reordered) == ["e1"]
    with pytest.raises(ValueError, match="conflicting"):
        await inbox.append("news", event(description="different content"))
    await consume(inbox)
    await inbox.close()
    async with make_inbox(tmp_path) as restored:
        assert await restored.append("news", event()) == ["e1"]
        assert await restored.claim("news") is None
        assert len(lines(tmp_path / "agent_inbox/news/inbox_read.jsonl")[0]["items"]) == 1


@pytest.mark.asyncio
async def test_read_history_trims_old_batches_without_discarding_unread_events(tmp_path):
    async with make_inbox(tmp_path, max_read_items=2) as store:
        await store.create("news")
        await store.append("news", [event(f"e{i}") for i in range(5)])
        assert [item["id"] for item in await consume(store)] == ["e0", "e1"]
        assert [item["id"] for item in await consume(store, limit=1)] == ["e2"]
        directory = tmp_path / "agent_inbox/news"
        assert [item["id"] for item in lines(directory / "inbox_unread.jsonl")] == ["e3", "e4"]
        assert [item["id"] for batch in lines(directory / "inbox_read.jsonl") for item in batch["items"]] == ["e2"]
        assert (await store.list())[0]["read_count"] == 1
    async with make_inbox(tmp_path, max_read_items=2) as restored:
        assert [item["id"] for item in await consume(restored)] == ["e3", "e4"]
        assert len((await restored.receipts())["news"]) == 1
        assert sorted(path.name for path in directory.glob("*.jsonl")) == ["inbox_read.jsonl", "inbox_unread.jsonl"]


@pytest.mark.asyncio
async def test_default_read_history_keeps_at_most_1000_events(tmp_path):
    async with make_inbox(tmp_path, claim_max_items=1000, claim_max_bytes=1024 * 1024) as store:
        await store.create("news")
        await store.append("news", [event(f"e{i}") for i in range(1001)])
        assert len(await consume(store)) == 1000
        assert len(await consume(store)) == 1
        assert (await store.list())[0]["read_count"] <= 1000
        await store.append("news", event("e0"))  # IDs expire with trimmed history.
        assert (await consume(store))[0]["id"] == "e0"


@pytest.mark.asyncio
async def test_global_capacity_includes_claims_and_ack_unblocks_another_stream(tmp_path):
    async with make_inbox(tmp_path, max_pending_items=1) as store:
        await store.create("news")
        await store.create("email")
        await store.append("news", event("e1"))
        batch = await store.claim("news")
        writing = asyncio.create_task(store.append("email", event("e2")))
        await asyncio.sleep(0.01)
        assert not writing.done()
        await store.nack("news", batch["token"])
        await asyncio.sleep(0)
        assert not writing.done()
        await consume(store)
        assert await asyncio.wait_for(writing, 1) == ["e2"]
        assert (await store.claim("email"))["items"][0]["id"] == "e2"


@pytest.mark.asyncio
async def test_byte_limits_bound_pending_inputs_and_claims(tmp_path):
    record = event(metadata={}, paths=[], description="x" * 90)
    size = len(LocalAgentInbox._encode(record))
    async with make_inbox(
        tmp_path,
        max_item_bytes=size,
        max_pending_bytes=size * 2,
        claim_max_bytes=size,
        append_timeout_seconds=0.01,
    ) as store:
        await store.create("news")
        await store.append("news", [record, {**record, "id": "e2"}])
        with pytest.raises(TimeoutError):
            await store.append("news", {**record, "id": "e3"})
        assert len(await consume(store)) == 1
        await store.append("news", {**record, "id": "e3"})
        assert (await store.list())[0]["pending_count"] == 2
        with pytest.raises(ValueError, match="max_item_bytes"):
            await store.append("news", event("large", description="x" * 1000))
        with pytest.raises(ValueError, match="pending budgets"):
            await store.append("news", [{**record, "id": f"x{i}"} for i in range(3)])


@pytest.mark.asyncio
async def test_wait_handles_arrival_claim_release_timeout_and_close(inbox):
    assert await inbox.wait("news", timeout=0.001) is False
    waiting = asyncio.create_task(inbox.wait(["news"], timeout=1))
    await inbox.append("news", event())
    assert await waiting is True
    batch = await inbox.claim("news")
    assert await inbox.wait("news", timeout=0.001) is False
    waiting = asyncio.create_task(inbox.wait(timeout=1))
    await inbox.nack("news", batch["token"])
    assert await waiting is True
    await consume(inbox)
    waiting = asyncio.create_task(inbox.wait())
    await asyncio.sleep(0)
    await inbox.close()
    with pytest.raises(RuntimeError, match="closed"):
        await waiting


@pytest.mark.asyncio
async def test_concurrent_append_and_claim_are_serialized(inbox):
    await asyncio.gather(*(inbox.append("news", event(f"e{i}")) for i in range(20)))
    batches = await asyncio.gather(*(inbox.claim("news") for _ in range(10)))
    claimed = [batch for batch in batches if batch is not None]
    assert len(claimed) == 1
    assert {item["id"] for item in claimed[0]["items"]} == {f"e{i}" for i in range(20)}
    # Arrivals during a claim belong to a later batch.
    await inbox.append("news", event("later"))
    await inbox.ack("news", claimed[0]["token"])
    assert [item["id"] for item in await consume(inbox)] == ["later"]
    with pytest.raises(ValueError, match="token"):
        await inbox.ack("news", claimed[0]["token"])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["append", "ack"])
async def test_cancelled_commit_finishes_disk_and_memory_together(inbox, operation, monkeypatch):
    batch = None
    if operation == "ack":
        await inbox.append("news", event())
        batch = await inbox.claim("news")
    entered, release = ThreadEvent(), ThreadEvent()
    original = inbox._write

    def blocked_write(path, raw):
        entered.set()
        assert release.wait(2)
        original(path, raw)

    monkeypatch.setattr(inbox, "_write", blocked_write)
    task = asyncio.create_task(
        inbox.append("news", event()) if operation == "append" else inbox.ack("news", batch["token"]),
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await inbox.list())[0]["pending_count"] == (1 if operation == "append" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["append", "ack"])
async def test_sync_failure_rolls_back_and_requires_restart(inbox, tmp_path, operation, monkeypatch):
    from reme.components.agent_inbox import local_agent_inbox as module

    await inbox.append("news", event())
    batch = await inbox.claim("news")
    original = module.sync_file
    calls = 0

    def fail_once(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk unavailable")
        original(fd)

    monkeypatch.setattr(module, "sync_file", fail_once)
    with pytest.raises(OSError, match="disk unavailable"):
        if operation == "append":
            await inbox.append("news", event("e2"))
        else:
            await inbox.ack("news", batch["token"])
    assert inbox._pending_count == 1
    with pytest.raises(RuntimeError, match="failed"):
        await inbox.append("news", event("e3"))
    await inbox.close()
    async with make_inbox(tmp_path) as restored:
        assert [item["id"] for item in await consume(restored)] == ["e1"]


@pytest.mark.asyncio
async def test_recovery_truncates_only_unterminated_tails(inbox, tmp_path):
    await inbox.append("news", [event("e1"), event("e2")])
    await consume(inbox, limit=1)
    await inbox.close()
    directory = tmp_path / "agent_inbox/news"
    for path in directory.glob("*.jsonl"):
        with path.open("ab") as file:
            file.write(b'{"interrupted":')
    async with make_inbox(tmp_path) as restored:
        assert [item["id"] for item in await consume(restored)] == ["e2"]
    assert all(not path.read_bytes() or path.read_bytes().endswith(b"\n") for path in directory.glob("*.jsonl"))


@pytest.mark.asyncio
async def test_corrupt_complete_record_fails_without_rewriting_source_and_releases_lock(inbox, tmp_path):
    await inbox.append("news", event())
    await inbox.close()
    path = tmp_path / "agent_inbox/news/inbox_unread.jsonl"
    with path.open("ab") as file:
        file.write(b'{"bad":true}\n')
    original = path.read_bytes()
    broken = make_inbox(tmp_path)
    with pytest.raises(ValidationError):
        await broken.start()
    assert path.read_bytes() == original
    assert broken._owner_fd is None
    # It must fail on the source error again, rather than a leaked writer lock.
    with pytest.raises(ValidationError):
        await make_inbox(tmp_path).start()


@pytest.mark.asyncio
async def test_startup_budget_changes_never_drop_persisted_events(inbox, tmp_path):
    await inbox.append("news", [event("e1"), event("e2")])
    await inbox.close()
    path = tmp_path / "agent_inbox/news/inbox_unread.jsonl"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="backlog"):
        await make_inbox(tmp_path, max_pending_items=1).start()
    assert path.read_bytes() == original
    async with make_inbox(tmp_path) as restored:
        assert len(await consume(restored)) == 2


@pytest.mark.asyncio
async def test_startup_rejects_pending_events_that_no_longer_fit_claims(inbox, tmp_path):
    await inbox.append("news", event(description="x" * 500))
    await inbox.close()
    restored = make_inbox(tmp_path, max_item_bytes=256, claim_max_bytes=256)
    try:
        with pytest.raises(ValueError, match="claim_max_bytes"):
            await restored.start()
    finally:
        await restored.close()
    async with make_inbox(tmp_path) as restored:
        assert (await consume(restored))[0]["id"] == "e1"


@pytest.mark.asyncio
async def test_duplicate_directory_is_not_a_component_io_failure(inbox, monkeypatch):
    original = inbox._path

    def case_insensitive_path(name="", filename=None):
        return original("news" if name == "NEWS" else name, filename)

    with monkeypatch.context() as patch:
        patch.setattr(inbox, "_path", case_insensitive_path)
        with pytest.raises(FileExistsError):
            await inbox.create("NEWS")
    assert [item["name"] for item in await inbox.list()] == ["news"]


@pytest.mark.asyncio
async def test_recovery_finishes_move_after_read_commit_before_unread_rewrite(inbox, tmp_path, monkeypatch):
    await inbox.append("news", [event("e1"), event("e2")])
    batch = await inbox.claim("news", limit=1)
    original = inbox._replace

    def interrupted(path, raw):
        if path.name == "inbox_unread.jsonl":
            raise OSError("interrupted move")
        original(path, raw)

    monkeypatch.setattr(inbox, "_replace", interrupted)
    with pytest.raises(OSError, match="interrupted move"):
        await inbox.ack("news", batch["token"], metadata={"result": "daily/report.md"})
    await inbox.close()
    async with make_inbox(tmp_path) as restored:
        receipt = (await restored.receipts())["news"][0]
        assert receipt.item_ids == ["e1"]
        assert receipt.metadata == {"result": "daily/report.md"}
        assert [item["id"] for item in await consume(restored)] == ["e2"]
        assert lines(tmp_path / "agent_inbox/news/inbox_unread.jsonl") == []


@pytest.mark.asyncio
async def test_model_input_metadata_is_detached_from_pending_events(inbox):
    source = Event.model_validate(event())
    await inbox.append("news", source)
    source.metadata["tags"].append("changed")
    assert (await inbox.claim("news"))["items"][0]["metadata"]["tags"] == ["work"]


@pytest.mark.asyncio
async def test_single_writer_lock_and_symlink_containment(inbox, tmp_path):
    with pytest.raises(RuntimeError, match="already has a writer"):
        await make_inbox(tmp_path).start()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "agent_inbox/escaped").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        await inbox.create("escaped")
    await inbox.append("news", event())
    target = tmp_path / "agent_inbox/news/inbox_read.jsonl"
    target.symlink_to(outside / "receipt.jsonl")
    batch = await inbox.claim("news")
    with pytest.raises(ValueError, match="symlinks"):
        await inbox.ack("news", batch["token"])
    assert not (outside / "receipt.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["", "..", "../escape", "a/b", "a\\b", "~user", "CON", " name", "/absolute", None])
async def test_invalid_stream_names_have_no_side_effects(inbox, name):
    with pytest.raises(ValueError, match="names"):
        await inbox.create(name)
    assert [item["name"] for item in await inbox.list()] == ["news"]


@pytest.mark.parametrize("path", ["../escape", "/absolute", "~/home", "a/../b", "a//b", "a\\b", "C:/outside", "."])
def test_event_paths_are_workspace_relative(path):
    with pytest.raises(ValidationError, match="workspace-relative"):
        Event.model_validate(event(paths=[path]))


@pytest.mark.parametrize(
    "options",
    [
        {"max_read_items": 0},
        {"max_pending_items": True},
        {"max_pending_bytes": 1.5},
        {"append_timeout_seconds": 0},
        {"append_timeout_seconds": float("inf")},
        {"append_timeout_seconds": True},
        {"max_item_bytes": 65537},
        {"directory": ".."},
        {"directory": "/tmp/escape"},
        {"directory": "~/events"},
    ],
)
def test_invalid_configuration(tmp_path, options):
    with pytest.raises(ValueError):
        make_inbox(tmp_path, **options)


@pytest.mark.asyncio
async def test_append_timeout_includes_waiting_for_shared_writer_lock(tmp_path):
    async with make_inbox(tmp_path, append_timeout_seconds=0.01) as inbox:
        await inbox.create("news")
        async with inbox._changed:
            pending = asyncio.create_task(inbox.append("news", event()))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(pending, 1)
        assert (await inbox.snapshot("news"))["pending_count"] == 0


@pytest.mark.asyncio
async def test_local_inbox_accepts_generic_inputs_without_an_agent_runtime(tmp_path):
    async with make_inbox(tmp_path) as inbox:
        await inbox.create("mail")
        await inbox.append("mail", [{"id": "ref", "file": "reminders/result.md"}, {"id": "task", "value": 42}])
        batch = await inbox.claim("mail")
        assert batch["items"] == [{"id": "ref", "file": "reminders/result.md"}, {"id": "task", "value": 42}]
        await inbox.ack("mail", batch["token"])
        assert (await inbox.snapshot("mail"))["receipt"].item_ids == ["ref", "task"]


@pytest.mark.asyncio
async def test_selected_claim_preserves_older_inputs_and_later_arrivals(tmp_path):
    async with make_inbox(tmp_path) as inbox:
        await inbox.create("mail")
        await inbox.append("mail", [{"id": "low"}, {"id": "urgent"}])
        assert await inbox.peek("mail") == {"id": "low"}
        assert await inbox.peek("mail", item_ids=["urgent", "low"]) == {"id": "low"}
        assert await inbox.peek("mail", item_ids=["urgent"]) == {"id": "urgent"}
        batch = await inbox.claim("mail", item_ids=["urgent"])
        assert await inbox.peek("mail") is None
        await inbox.append("mail", {"id": "new"})
        assert [item["item"]["id"] for item in (await inbox.read("mail", state="processing"))["items"]] == ["urgent"]
        assert (await inbox.snapshot("mail"))["in_flight_count"] == 1
        await inbox.ack("mail", batch["token"], metadata={"decision": "sent"})
        assert inbox._pending_count == 2
        assert (await inbox.snapshot("mail"))["receipt"].item_ids == ["urgent"]
    async with make_inbox(tmp_path) as inbox:
        batch = await inbox.claim("mail")
        assert [item["id"] for item in batch["items"]] == ["low", "new"]


@pytest.mark.asyncio
async def test_selected_commit_recovers_after_interrupted_unread_rewrite(tmp_path, monkeypatch):
    async with make_inbox(tmp_path) as inbox:
        await inbox.create("mail")
        await inbox.append("mail", [{"id": "low"}, {"id": "urgent"}, {"id": "new"}])
        batch = await inbox.claim("mail", item_ids=["urgent"])
        replace = inbox._replace

        def interrupted(path, raw):
            if path.name == "inbox_unread.jsonl":
                raise OSError("interrupted move")
            replace(path, raw)

        monkeypatch.setattr(inbox, "_replace", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            await inbox.ack("mail", batch["token"])
    async with make_inbox(tmp_path) as inbox:
        assert (await inbox.snapshot("mail"))["receipt"].item_ids == ["urgent"]
        assert [item["id"] for item in (await inbox.claim("mail"))["items"]] == ["low", "new"]


@pytest.mark.asyncio
async def test_event_inbox_validates_sources_and_supports_unicode_waits(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    async with AgentEventInbox(app_context=context) as inbox:
        await inbox.create("观察")
        await inbox.append("观察", event(metadata={"status": "ready"}))
        await inbox.append("观察", event(metadata={"status": "ready"}))
        assert (await inbox.snapshot("观察"))["pending_count"] == 1
        assert await inbox.wait("观察", timeout=0.01)
        with pytest.raises(ValidationError):
            await inbox.append("观察", {"id": "invalid"})
        with pytest.raises(ValidationError):
            await inbox.append("观察", {"id": "escape", "file": "../outside.md"})
        with pytest.raises(ValidationError):
            await inbox.append("观察", {"id": "old-reference", "file": "reminders/result.md"})
    async with AgentEventInbox(app_context=context) as inbox:
        await inbox.append("观察", event(metadata={"status": "ready"}))
        assert (await inbox.snapshot("观察"))["pending_count"] == 1
        batch = await inbox.claim("观察")
        assert batch["items"][0]["metadata"] == {"status": "ready"}
