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
        assert component.segment_max_events == 1000 and component.max_segments == 10
        await component.create("news")
        await component.append("news", event())
        assert (await consume(component))[0]["id"] == "e1"
    assert not component.is_started


@pytest.mark.asyncio
async def test_event_and_receipt_are_separate_and_claim_is_not_consumption(stream, tmp_path):
    source = event()
    assert await stream.append("news", source) == ["e1"]
    source["metadata"]["tags"].append("mutated")
    batch = await stream.claim("news")
    assert batch["events"][0]["metadata"]["tags"] == ["work"]
    batch["events"][0]["metadata"]["tags"].append("also mutated")
    directory = tmp_path / "event_stream/news"
    persisted = lines(directory / "00000001.events.jsonl")
    assert set(persisted[0]) == {"id", "event_ts", "source", "description", "paths", "metadata"}
    assert persisted[0]["metadata"]["tags"] == ["work"]
    assert not list(directory.glob("*.consumed.jsonl"))
    assert await stream.claim("news") is None
    summary = (await stream.list())[0]
    assert summary["pending_count"] == summary["in_flight_count"] == 1
    assert summary["pending_bytes"] == (directory / "00000001.events.jsonl").stat().st_size
    await stream.nack("news", batch["token"])
    batch = await stream.claim("news")
    assert batch["events"] == persisted
    timestamp = await stream.ack("news", batch["token"])
    assert lines(directory / "00000001.consumed.jsonl") == [
        {"event_ids": ["e1"], "consume_ts": timestamp.replace("+00:00", "Z")},
    ]
    assert lines(directory / "00000001.events.jsonl") == persisted
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
        assert len(lines(tmp_path / "event_stream/news/00000001.events.jsonl")) == 1


@pytest.mark.asyncio
async def test_segment_retention_never_discards_partially_consumed_events(tmp_path):
    async with make_stream(tmp_path, segment_max_events=2, max_segments=2, append_timeout_seconds=0.01) as store:
        await store.create("news")
        await store.append("news", [event("e1"), event("e2")])
        await consume(store, limit=1)
        await store.append("news", [event("e3"), event("e4")])
        with pytest.raises(TimeoutError):
            await store.append("news", event("e5"))
        directory = tmp_path / "event_stream/news"
        assert len(list(directory.glob("*.events.jsonl"))) == 2
        assert [item["id"] for item in await consume(store)] == ["e2"]
        await store.append("news", event("e5"))
        assert not (directory / "00000001.events.jsonl").exists()
        assert not (directory / "00000001.consumed.jsonl").exists()
        assert (directory / "00000002.events.jsonl").exists()
        assert (directory / "00000003.events.jsonl").exists()
        # Claims stop at segment boundaries, keeping each acknowledgement to one write.
        assert [item["id"] for item in await consume(store)] == ["e3", "e4"]
        assert [item["id"] for item in await consume(store)] == ["e5"]
        assert (await store.list())[0]["segments"] == 2


@pytest.mark.asyncio
async def test_one_segment_can_rotate_after_consumption(tmp_path):
    async with make_stream(tmp_path, segment_max_events=1, max_segments=1) as store:
        await store.create("news")
        await store.append("news", event("e1"))
        await consume(store)
        await store.append("news", event("e2"))
        assert [path.name for path in (tmp_path / "event_stream/news").glob("*.jsonl")] == ["00000002.events.jsonl"]


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
async def test_byte_limits_rotate_and_bound_claims(tmp_path):
    record = event(metadata={}, paths=[], description="x" * 90)
    size = len(LocalEventStream._encode(record))
    async with make_stream(
        tmp_path,
        max_event_bytes=size,
        segment_max_bytes=size * 2,
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
        assert (await store.list())[0]["segments"] == 2
        with pytest.raises(ValueError, match="max_event_bytes"):
            await store.append("news", event("large", description="x" * 1000))
        with pytest.raises(ValueError, match="one segment"):
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
    import reme.components.event_stream as module

    await stream.append("news", event())
    batch = await stream.claim("news")
    original = module._sync
    calls = 0

    def fail_once(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk unavailable")
        original(fd)

    monkeypatch.setattr(module, "_sync", fail_once)
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
    assert all(path.read_bytes().endswith(b"\n") for path in directory.glob("*.jsonl"))


@pytest.mark.asyncio
async def test_corrupt_complete_record_fails_without_rewriting_source_and_releases_lock(stream, tmp_path):
    await stream.append("news", event())
    await stream.close()
    path = tmp_path / "event_stream/news/00000001.events.jsonl"
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
    path = tmp_path / "event_stream/news/00000001.events.jsonl"
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
async def test_noncanonical_receipt_name_is_not_silently_deleted(stream, tmp_path):
    await stream.close()
    path = tmp_path / "event_stream/news/000000001.consumed.jsonl"
    path.write_text('{"event_ids":["e1"],"consume_ts":"2026-11-11T00:00:00Z"}\n')
    original = path.read_bytes()
    restored = make_stream(tmp_path)
    try:
        with pytest.raises(ValueError, match="segment name"):
            await restored.start()
    finally:
        await restored.close()
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_duplicate_directory_is_not_a_component_io_failure(stream, monkeypatch):
    original = stream._path

    def case_insensitive_path(name="", segment=None, **kwargs):
        return original("news" if name == "NEWS" else name, segment, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(stream, "_path", case_insensitive_path)
        with pytest.raises(FileExistsError):
            await stream.create("NEWS")
    assert [item["name"] for item in await stream.list()] == ["news"]


@pytest.mark.asyncio
async def test_recovery_orders_segments_numerically_across_filename_widths(stream, tmp_path):
    await stream.append("news", event("e1"))
    await stream.close()
    directory = tmp_path / "event_stream/news"
    (directory / "00000001.events.jsonl").rename(directory / "99999999.events.jsonl")
    (directory / "100000000.events.jsonl").write_bytes(LocalEventStream._encode(event("e2")))
    async with make_stream(tmp_path) as restored:
        assert (await consume(restored))[0]["id"] == "e1"
        assert (await consume(restored))[0]["id"] == "e2"


@pytest.mark.asyncio
async def test_recovery_finishes_interrupted_consumed_segment_cleanup(stream, tmp_path):
    await stream.append("news", event())
    await consume(stream)
    await stream.close()
    directory = tmp_path / "event_stream/news"
    (directory / "00000001.events.jsonl").unlink()
    async with make_stream(tmp_path) as restored:
        assert await restored.claim("news") is None
        assert not list(directory.glob("*.jsonl"))


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
    target = tmp_path / "event_stream/news/00000001.consumed.jsonl"
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
        {"max_segments": 0},
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
