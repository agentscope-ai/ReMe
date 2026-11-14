"""Durable event streams: recovery, bounded buffering and cancellation-safe commits."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
import json
from threading import Event as ThreadEvent

import pytest
import pytest_asyncio
from pydantic import ValidationError

from reme.application import Application
from reme.components import LocalEventStream, create_application_registry
from reme.components.application_context import ApplicationContext
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


def make_stream(tmp_path, **kwargs):
    return LocalEventStream(app_context=ApplicationContext(workspace_dir=str(tmp_path)), **kwargs)


@pytest_asyncio.fixture
async def stream(tmp_path):
    async with make_stream(tmp_path) as result:
        await result.create("news")
        yield result


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


async def consume(stream, name="news", limit=None):
    batch = await stream.claim(name, limit)
    await stream.ack(name, batch["token"])
    return batch["events"]


@pytest.mark.asyncio
async def test_registry_config_and_application_lifecycle(tmp_path):
    assert create_application_registry().get("event_stream", "local") is LocalEventStream
    assert "event_stream" not in resolve_app_config(config="default")["components"]
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
        service={"backend": "cli"},
        components={"event_stream": {"default": {"backend": "local"}}},
        jobs={},
    )
    async with app:
        component = app.context.components["event_stream"]["default"]
        assert component.max_read_events == 1000
        await component.create("news")
        await component.append("news", event())
        assert (await consume(component))[0]["id"] == "e1"
    assert not component.is_started


@pytest.mark.asyncio
async def test_mount_restores_sources_and_forget_rejects_pending_work(tmp_path):
    async with make_stream(tmp_path) as component:
        await component.mount("news", "event_stream/news")
        await component.mount("news", "event_stream/news")
        await component.append("news", event())
        with pytest.raises(ValueError, match="pending"):
            await component.forget("news")
        await consume(component)
        await component.forget("news")
        await component.mount("news", "event_stream/news")
        snapshot = await component.snapshot("news")
        assert snapshot["pending_count"] == 0
        assert snapshot["consumption"].event_ids == ["e1"]
        assert (await component.receipts())["news"][0].event_ids == ["e1"]
        with pytest.raises(ValueError, match="layout"):
            await component.mount("news", "other/news")


@pytest.mark.asyncio
async def test_result_handoffs_use_generic_retention_without_notification_coupling(tmp_path):
    async with make_stream(tmp_path, max_read_events=1, append_timeout_seconds=0.02) as component:
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
    async with make_stream(tmp_path, max_read_events=2, append_timeout_seconds=0.02) as component:
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
        assert (await component.receipts())["news"][0].event_ids == ["e2"]
        await component.retain("news", ["e2"], release=True)
        await component.ack("news", batch["token"])
        assert [r.event_ids for r in (await component.receipts())["news"]] == [["e3"], ["e4"]]


@pytest.mark.asyncio
async def test_event_and_receipt_are_separate_and_claim_is_not_consumption(stream, tmp_path):
    source = event()
    assert await stream.append("news", source) == ["e1"]
    source["metadata"]["tags"].append("mutated")
    batch = await stream.claim("news")
    assert batch["events"][0]["metadata"]["tags"] == ["work"]
    batch["events"][0]["metadata"]["tags"].append("also mutated")
    directory = tmp_path / "event_stream/news"
    persisted = lines(directory / "events_unread.jsonl")
    assert set(persisted[0]) == {"id", "event_ts", "source", "description", "paths", "metadata"}
    assert persisted[0]["metadata"]["tags"] == ["work"]
    assert not (directory / "events_read.jsonl").exists()
    assert await stream.claim("news") is None
    summary = (await stream.list())[0]
    assert summary["pending_count"] == summary["in_flight_count"] == 1
    assert summary["pending_bytes"] == (directory / "events_unread.jsonl").stat().st_size
    await stream.nack("news", batch["token"])
    batch = await stream.claim("news")
    assert batch["events"] == persisted
    timestamp = await stream.ack("news", batch["token"])
    assert lines(directory / "events_read.jsonl") == [
        {"events": persisted, "consume_ts": timestamp.replace("+00:00", "Z"), "metadata": {}},
    ]
    assert lines(directory / "events_unread.jsonl") == []
    assert (await stream.list())[0]["pending_count"] == 0


@pytest.mark.asyncio
async def test_recovery_restores_unacked_claims_and_empty_streams(stream, tmp_path):
    await stream.create("empty")
    await stream.append("news", [event("e1"), event("e2")])
    await consume(stream, limit=1)
    abandoned = await stream.claim("news")
    await stream.close()
    async with make_stream(tmp_path) as restored:
        assert [item["name"] for item in await restored.list()] == ["empty", "news"]
        with pytest.raises(ValueError, match="token"):
            await restored.ack("news", abandoned["token"])
        assert [item["id"] for item in await consume(restored)] == ["e2"]
        assert await restored.claim("empty") is None


@pytest.mark.asyncio
async def test_idempotent_append_and_conflicting_ids(stream, tmp_path):
    await stream.append("news", [event(), event()])
    reordered = event(metadata={"tags": ["work"], "url": "https://example.com/news"})
    assert await stream.append("news", reordered) == ["e1"]
    with pytest.raises(ValueError, match="conflicting"):
        await stream.append("news", event(description="different content"))
    await consume(stream)
    await stream.close()
    async with make_stream(tmp_path) as restored:
        assert await restored.append("news", event()) == ["e1"]
        assert await restored.claim("news") is None
        assert len(lines(tmp_path / "event_stream/news/events_read.jsonl")[0]["events"]) == 1


@pytest.mark.asyncio
async def test_read_history_trims_old_batches_without_discarding_unread_events(tmp_path):
    async with make_stream(tmp_path, max_read_events=2) as store:
        await store.create("news")
        await store.append("news", [event(f"e{i}") for i in range(5)])
        assert [item["id"] for item in await consume(store)] == ["e0", "e1"]
        assert [item["id"] for item in await consume(store, limit=1)] == ["e2"]
        directory = tmp_path / "event_stream/news"
        assert [item["id"] for item in lines(directory / "events_unread.jsonl")] == ["e3", "e4"]
        assert [item["id"] for batch in lines(directory / "events_read.jsonl") for item in batch["events"]] == ["e2"]
        assert (await store.list())[0]["read_count"] == 1
    async with make_stream(tmp_path, max_read_events=2) as restored:
        assert [item["id"] for item in await consume(restored)] == ["e3", "e4"]
        assert len((await restored.receipts())["news"]) == 1
        assert sorted(path.name for path in directory.glob("*.jsonl")) == ["events_read.jsonl", "events_unread.jsonl"]


@pytest.mark.asyncio
async def test_default_read_history_keeps_at_most_1000_events(tmp_path):
    async with make_stream(tmp_path, claim_max_events=1000, claim_max_bytes=1024 * 1024) as store:
        await store.create("news")
        await store.append("news", [event(f"e{i}") for i in range(1001)])
        assert len(await consume(store)) == 1000
        assert len(await consume(store)) == 1
        assert (await store.list())[0]["read_count"] <= 1000
        await store.append("news", event("e0"))  # IDs expire with trimmed history.
        assert (await consume(store))[0]["id"] == "e0"


@pytest.mark.asyncio
async def test_global_capacity_includes_claims_and_ack_unblocks_another_stream(tmp_path):
    async with make_stream(tmp_path, max_pending_events=1) as store:
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
        assert (await store.claim("email"))["events"][0]["id"] == "e2"


@pytest.mark.asyncio
async def test_byte_limits_bound_pending_inputs_and_claims(tmp_path):
    record = event(metadata={}, paths=[], description="x" * 90)
    size = len(LocalEventStream._encode(record))
    async with make_stream(
        tmp_path,
        max_event_bytes=size,
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
        with pytest.raises(ValueError, match="max_event_bytes"):
            await store.append("news", event("large", description="x" * 1000))
        with pytest.raises(ValueError, match="pending budgets"):
            await store.append("news", [{**record, "id": f"x{i}"} for i in range(3)])


@pytest.mark.asyncio
async def test_wait_handles_arrival_claim_release_timeout_and_close(stream):
    assert await stream.wait("news", timeout=0.001) is False
    waiting = asyncio.create_task(stream.wait(["news"], timeout=1))
    await stream.append("news", event())
    assert await waiting is True
    batch = await stream.claim("news")
    assert await stream.wait("news", timeout=0.001) is False
    waiting = asyncio.create_task(stream.wait(timeout=1))
    await stream.nack("news", batch["token"])
    assert await waiting is True
    await consume(stream)
    waiting = asyncio.create_task(stream.wait())
    await asyncio.sleep(0)
    await stream.close()
    with pytest.raises(RuntimeError, match="closed"):
        await waiting


@pytest.mark.asyncio
async def test_concurrent_append_and_claim_are_serialized(stream):
    await asyncio.gather(*(stream.append("news", event(f"e{i}")) for i in range(20)))
    batches = await asyncio.gather(*(stream.claim("news") for _ in range(10)))
    claimed = [batch for batch in batches if batch is not None]
    assert len(claimed) == 1
    assert {item["id"] for item in claimed[0]["events"]} == {f"e{i}" for i in range(20)}
    # Arrivals during a claim belong to a later batch.
    await stream.append("news", event("later"))
    await stream.ack("news", claimed[0]["token"])
    assert [item["id"] for item in await consume(stream)] == ["later"]
    with pytest.raises(ValueError, match="token"):
        await stream.ack("news", claimed[0]["token"])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["append", "ack"])
async def test_cancelled_commit_finishes_disk_and_memory_together(stream, operation, monkeypatch):
    batch = None
    if operation == "ack":
        await stream.append("news", event())
        batch = await stream.claim("news")
    entered, release = ThreadEvent(), ThreadEvent()
    original = stream._write

    def blocked_write(path, raw):
        entered.set()
        assert release.wait(2)
        original(path, raw)

    monkeypatch.setattr(stream, "_write", blocked_write)
    task = asyncio.create_task(
        stream.append("news", event()) if operation == "append" else stream.ack("news", batch["token"]),
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
    assert (await stream.list())[0]["pending_count"] == (1 if operation == "append" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["append", "ack"])
async def test_sync_failure_rolls_back_and_requires_restart(stream, tmp_path, operation, monkeypatch):
    from reme.components.event_stream import local_event_stream as module

    await stream.append("news", event())
    batch = await stream.claim("news")
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
            await stream.append("news", event("e2"))
        else:
            await stream.ack("news", batch["token"])
    assert stream._pending_count == 1
    with pytest.raises(RuntimeError, match="failed"):
        await stream.append("news", event("e3"))
    await stream.close()
    async with make_stream(tmp_path) as restored:
        assert [item["id"] for item in await consume(restored)] == ["e1"]


@pytest.mark.asyncio
async def test_recovery_truncates_only_unterminated_tails(stream, tmp_path):
    await stream.append("news", [event("e1"), event("e2")])
    await consume(stream, limit=1)
    await stream.close()
    directory = tmp_path / "event_stream/news"
    for path in directory.glob("*.jsonl"):
        with path.open("ab") as file:
            file.write(b'{"interrupted":')
    async with make_stream(tmp_path) as restored:
        assert [item["id"] for item in await consume(restored)] == ["e2"]
    assert all(not path.read_bytes() or path.read_bytes().endswith(b"\n") for path in directory.glob("*.jsonl"))


@pytest.mark.asyncio
async def test_corrupt_complete_record_fails_without_rewriting_source_and_releases_lock(stream, tmp_path):
    await stream.append("news", event())
    await stream.close()
    path = tmp_path / "event_stream/news/events_unread.jsonl"
    with path.open("ab") as file:
        file.write(b'{"bad":true}\n')
    original = path.read_bytes()
    broken = make_stream(tmp_path)
    with pytest.raises(ValidationError):
        await broken.start()
    assert path.read_bytes() == original
    assert broken._owner_fd is None
    # It must fail on the source error again, rather than a leaked writer lock.
    with pytest.raises(ValidationError):
        await make_stream(tmp_path).start()


@pytest.mark.asyncio
async def test_startup_budget_changes_never_drop_persisted_events(stream, tmp_path):
    await stream.append("news", [event("e1"), event("e2")])
    await stream.close()
    path = tmp_path / "event_stream/news/events_unread.jsonl"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="backlog"):
        await make_stream(tmp_path, max_pending_events=1).start()
    assert path.read_bytes() == original
    async with make_stream(tmp_path) as restored:
        assert len(await consume(restored)) == 2


@pytest.mark.asyncio
async def test_startup_rejects_pending_events_that_no_longer_fit_claims(stream, tmp_path):
    await stream.append("news", event(description="x" * 500))
    await stream.close()
    restored = make_stream(tmp_path, max_event_bytes=256, claim_max_bytes=256)
    try:
        with pytest.raises(ValueError, match="claim_max_bytes"):
            await restored.start()
    finally:
        await restored.close()
    async with make_stream(tmp_path) as restored:
        assert (await consume(restored))[0]["id"] == "e1"


@pytest.mark.asyncio
async def test_numbered_logs_require_explicit_migration_and_remain_untouched(stream, tmp_path):
    await stream.close()
    path = tmp_path / "event_stream/news/00000001.events.jsonl"
    original = LocalEventStream._encode(event())
    path.write_bytes(original)
    with pytest.raises(ValueError, match="migration"):
        await make_stream(tmp_path).start()
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_duplicate_directory_is_not_a_component_io_failure(stream, monkeypatch):
    original = stream._path

    def case_insensitive_path(name="", filename=None):
        return original("news" if name == "NEWS" else name, filename)

    with monkeypatch.context() as patch:
        patch.setattr(stream, "_path", case_insensitive_path)
        with pytest.raises(FileExistsError):
            await stream.create("NEWS")
    assert [item["name"] for item in await stream.list()] == ["news"]


@pytest.mark.asyncio
async def test_recovery_finishes_move_after_read_commit_before_unread_rewrite(stream, tmp_path, monkeypatch):
    await stream.append("news", [event("e1"), event("e2")])
    batch = await stream.claim("news", limit=1)
    original = stream._replace

    def interrupted(path, raw):
        if path.name == "events_unread.jsonl":
            raise OSError("interrupted move")
        original(path, raw)

    monkeypatch.setattr(stream, "_replace", interrupted)
    with pytest.raises(OSError, match="interrupted move"):
        await stream.ack("news", batch["token"], metadata={"result": "daily/report.md"})
    await stream.close()
    async with make_stream(tmp_path) as restored:
        receipt = (await restored.receipts())["news"][0]
        assert receipt.event_ids == ["e1"]
        assert receipt.metadata == {"result": "daily/report.md"}
        assert [item["id"] for item in await consume(restored)] == ["e2"]
        assert lines(tmp_path / "event_stream/news/events_unread.jsonl") == []


@pytest.mark.asyncio
async def test_model_input_metadata_is_detached_from_pending_events(stream):
    source = Event.model_validate(event())
    await stream.append("news", source)
    source.metadata["tags"].append("changed")
    assert (await stream.claim("news"))["events"][0]["metadata"]["tags"] == ["work"]


@pytest.mark.asyncio
async def test_single_writer_lock_and_symlink_containment(stream, tmp_path):
    with pytest.raises(RuntimeError, match="already has a writer"):
        await make_stream(tmp_path).start()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "event_stream/escaped").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        await stream.create("escaped")
    await stream.append("news", event())
    target = tmp_path / "event_stream/news/events_read.jsonl"
    target.symlink_to(outside / "receipt.jsonl")
    batch = await stream.claim("news")
    with pytest.raises(ValueError, match="symlinks"):
        await stream.ack("news", batch["token"])
    assert not (outside / "receipt.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["", "..", "../escape", "a/b", "a\\b", "~user", "CON", " name", "/absolute", None])
async def test_invalid_stream_names_have_no_side_effects(stream, name):
    with pytest.raises(ValueError, match="names"):
        await stream.create(name)
    assert [item["name"] for item in await stream.list()] == ["news"]


@pytest.mark.parametrize("path", ["../escape", "/absolute", "~/home", "a/../b", "a//b", "a\\b", "C:/outside", "."])
def test_event_paths_are_workspace_relative(path):
    with pytest.raises(ValidationError, match="workspace-relative"):
        Event.model_validate(event(paths=[path]))


@pytest.mark.parametrize(
    "options",
    [
        {"max_read_events": 0},
        {"max_pending_events": True},
        {"max_pending_bytes": 1.5},
        {"append_timeout_seconds": 0},
        {"append_timeout_seconds": float("inf")},
        {"append_timeout_seconds": True},
        {"max_event_bytes": 65537},
        {"directory": ".."},
        {"directory": "/tmp/escape"},
        {"directory": "~/events"},
    ],
)
def test_invalid_configuration(tmp_path, options):
    with pytest.raises(ValueError):
        make_stream(tmp_path, **options)
