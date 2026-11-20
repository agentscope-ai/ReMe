"""Tests for the action-item contract and its local store."""

# pylint: disable=redefined-outer-name,protected-access

from contextlib import chdir as temp_chdir
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

import pytest
from pydantic import ValidationError

from reme.components.action_item_store import BaseActionItemStore, LocalActionItemStore
from reme.schema import (
    ActionItem,
    ActionItemEvidence,
    ActionItemQuery,
    ActionItemState,
    action_item_id,
)

BACKENDS = [LocalActionItemStore]
NOTE = "daily/2026-11-13/release.md"
BASE = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)


def evidence(observed_at: datetime, locator: str = "## Next") -> ActionItemEvidence:
    """Build one evidence row pointing at the shared note."""
    return ActionItemEvidence(path=NOTE, locator=locator, observed_at=observed_at)


def make_item(
    description: str = "Cut the 2.3.1 release",
    state: ActionItemState = ActionItemState.UNRESOLVED,
    observed_at: datetime = BASE,
    updated_at: datetime | None = None,
    note: str = NOTE,
    aliases: list[str] | None = None,
) -> ActionItem:
    """Build an action item with a stable id for the given note and text."""
    return ActionItem(
        id=action_item_id(note, description),
        note=note,
        description=description,
        state=state,
        evidence=[evidence(observed_at)],
        aliases=list(aliases or []),
        first_seen=observed_at,
        last_updated=updated_at or observed_at,
    )


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_upsert_and_query_roundtrip(backend_cls):
    """Stored items come back with the same id and state."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()

        item = make_item()
        stored = await store.upsert([item])

        assert [s.id for s in stored] == [item.id]
        assert await store.get(item.id) == item
        assert [i.id for i in await store.query()] == [item.id]
        assert await store.get("missing") is None

        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_records_survive_restart(backend_cls):
    """The JSONL file reloads and ids stay identical across restarts."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        first: BaseActionItemStore = backend_cls()
        await first.start()
        item = make_item(state=ActionItemState.BLOCKED)
        await first.upsert([item])
        await first.close()

        second: BaseActionItemStore = backend_cls()
        await second.start()
        reloaded = await second.query()

        assert len(reloaded) == 1
        assert reloaded[0].id == item.id
        assert reloaded[0].state is ActionItemState.BLOCKED
        assert reloaded[0].evidence[0].locator == "## Next"
        assert reloaded[0].aliases == []
        await second.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_later_evidence_resolves_the_item(backend_cls):
    """A newer observation updates the state of the same item."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()

        await store.upsert([make_item(state=ActionItemState.UNRESOLVED, observed_at=BASE)])
        later = BASE + timedelta(days=2)
        await store.upsert(
            [
                make_item(
                    state=ActionItemState.COMPLETED,
                    observed_at=later,
                    updated_at=later,
                ),
            ],
        )

        stored = await store.get(action_item_id(NOTE, "Cut the 2.3.1 release"))
        assert stored is not None
        assert stored.state is ActionItemState.COMPLETED
        assert stored.first_seen == BASE
        assert [e.observed_at for e in stored.evidence] == [BASE, later]
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_older_evidence_cannot_override_newer_resolution(backend_cls):
    """Evidence observed earlier but processed later keeps the newer state."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()

        resolved_at = BASE + timedelta(days=5)
        await store.upsert(
            [
                make_item(
                    state=ActionItemState.COMPLETED,
                    observed_at=resolved_at,
                    updated_at=resolved_at,
                ),
            ],
        )
        stale_at = BASE
        await store.upsert(
            [
                make_item(
                    state=ActionItemState.UNRESOLVED,
                    observed_at=stale_at,
                    updated_at=stale_at,
                ),
            ],
        )

        stored = await store.get(action_item_id(NOTE, "Cut the 2.3.1 release"))
        assert stored is not None
        assert stored.state is ActionItemState.COMPLETED
        assert stored.last_updated == resolved_at
        assert len(stored.evidence) == 2
        assert stored.evidence[0].observed_at == stale_at
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_duplicate_evidence_is_merged(backend_cls):
    """Re-observing the same evidence never grows the list."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()

        item = make_item()
        await store.upsert([item])
        await store.upsert([item])

        stored = await store.get(item.id)
        assert stored is not None
        assert len(stored.evidence) == 1
        await store.close()


def test_state_other_than_uncertain_requires_evidence():
    """A state must never be asserted without evidence."""

    with pytest.raises(ValidationError):
        ActionItem(
            id="ai_x",
            note=NOTE,
            description="Ship it",
            state=ActionItemState.UNRESOLVED,
            first_seen=BASE,
            last_updated=BASE,
        )


def test_uncertain_state_allows_missing_evidence():
    """Unknown material maps to uncertain and may carry no evidence yet."""

    item = ActionItem(
        id="ai_y",
        note=NOTE,
        description="Maybe follow up",
        state=ActionItemState.UNCERTAIN,
        first_seen=BASE,
        last_updated=BASE,
    )
    assert item.state is ActionItemState.UNCERTAIN
    assert item.evidence == []


def test_naive_datetimes_are_rejected():
    """Timestamps without a timezone are ambiguous and must not be stored."""

    naive = datetime(2026, 11, 1, 8, 0)
    with pytest.raises(ValidationError):
        ActionItem(id="ai_z", note=NOTE, description="No zone", first_seen=naive, last_updated=naive)
    with pytest.raises(ValidationError):
        ActionItemEvidence(path=NOTE, observed_at=naive)
    with pytest.raises(ValidationError):
        ActionItemQuery(updated_since=naive)
    with pytest.raises(ValidationError):
        make_item().model_copy(update={"last_updated": naive}).model_validate(
            make_item().model_copy(update={"last_updated": naive}).model_dump(),
        )


def test_non_normalized_paths_are_rejected():
    """Only normalized workspace-relative POSIX paths identify a note."""

    for bad in ["", "/abs.md", "dir\\file.md", "a/../b.md", "./a.md", "a//b.md", "C:/x.md"]:
        with pytest.raises(ValidationError):
            ActionItemEvidence(path=bad, observed_at=BASE)


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_query_filters_by_state_time_and_limit(backend_cls):
    """state, updated_since and limit narrow the result deterministically."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()

        older = make_item(
            description="Old unresolved",
            state=ActionItemState.UNRESOLVED,
            observed_at=BASE,
        )
        newer = make_item(
            description="New blocked",
            state=ActionItemState.BLOCKED,
            observed_at=BASE + timedelta(days=3),
        )
        await store.upsert([older, newer])

        only_blocked = await store.query(ActionItemQuery(states=[ActionItemState.BLOCKED]))
        assert [i.id for i in only_blocked] == [newer.id]

        recent = await store.query(ActionItemQuery(updated_since=BASE + timedelta(days=1)))
        assert [i.id for i in recent] == [newer.id]

        limited = await store.query(ActionItemQuery(limit=1))
        assert [i.id for i in limited] == [newer.id]

        everything = await store.query()
        assert [i.id for i in everything] == [newer.id, older.id]
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_query_without_limit_returns_every_match(backend_cls):
    """No implicit cap: only an explicit limit truncates the result."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()

        items = [make_item(description=f"Task {index}") for index in range(120)]
        await store.upsert(items)

        assert len(await store.query()) == 120
        assert len(await store.query(ActionItemQuery(limit=5))) == 5
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_alias_merges_a_reworded_item(backend_cls):
    """A reworded description keeps one record instead of forking it."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()

        original = make_item(description="Cut the 2.3.1 release")
        await store.upsert([original])

        later = BASE + timedelta(days=1)
        reworded_description = "Cut the 2.3.1 release and announce it"
        reworded = make_item(
            description=reworded_description,
            observed_at=later,
            updated_at=later,
            aliases=[original.id],
        )
        assert reworded.id != original.id

        stored = await store.upsert([reworded])

        assert len(await store.query()) == 1
        assert stored[0].id == original.id
        assert stored[0].description == reworded_description
        assert reworded.id in stored[0].aliases
        assert original.id not in stored[0].aliases
        assert (await store.get(reworded.id)) is not None
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_corrupt_line_is_skipped(backend_cls):
    """One bad record does not hide the valid ones on reload."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        first: BaseActionItemStore = backend_cls()
        await first.start()
        item = make_item()
        await first.upsert([item])
        await first.close()

        target = first._file
        with target.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")

        second: BaseActionItemStore = backend_cls()
        await second.start()
        assert [i.id for i in await second.query()] == [item.id]
        await second.close()

        raw = target.read_text(encoding="utf-8")
        assert "{not json}" in raw.splitlines()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_unreadable_lines_are_preserved_verbatim(backend_cls):
    """A record this version cannot parse is kept, never deleted by a rewrite."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        first: BaseActionItemStore = backend_cls()
        await first.start()
        await first.upsert([make_item()])
        await first.close()

        target = first._file
        unknown = json.dumps(
            {
                "id": "ai_future",
                "note": NOTE,
                "description": "Written by a newer version",
                "state": "deferred",
                "first_seen": "2026-11-02T08:00:00Z",
                "last_updated": "2026-11-02T08:00:00Z",
            },
        )
        with target.open("a", encoding="utf-8") as handle:
            handle.write(unknown + "\n")

        second: BaseActionItemStore = backend_cls()
        await second.start()
        assert len(await second.query()) == 1
        await second.upsert([make_item(description="Another work")])
        await second.close()

        raw = target.read_text(encoding="utf-8")
        assert unknown in raw.splitlines()
        assert len([line for line in raw.splitlines() if line.strip()]) == 3

        third: BaseActionItemStore = backend_cls()
        await third.start()
        await third.close()

        again = target.read_text(encoding="utf-8")
        assert again.splitlines().count(unknown) == 1
        assert len([line for line in again.splitlines() if line.strip()]) == 3


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_legacy_record_without_timezone_is_preserved_not_dropped(backend_cls):
    """A record the previous schema accepted is kept, never silently deleted."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        seed: BaseActionItemStore = backend_cls()
        await seed.start()
        await seed.close()

        target = seed._file
        legacy = json.dumps(
            {
                "id": "ai_legacy",
                "note": NOTE,
                "description": "Written before timestamps carried a timezone",
                "state": "unresolved",
                "evidence": [
                    {"path": NOTE, "locator": "## Next", "observed_at": "2026-11-01T08:00:00"},
                ],
                "first_seen": "2026-11-01T08:00:00",
                "last_updated": "2026-11-01T08:00:00",
            },
        )
        target.write_text(legacy + "\n", encoding="utf-8")

        store: BaseActionItemStore = backend_cls()
        await store.start()
        assert await store.query() == []
        await store.close()

        assert target.read_text(encoding="utf-8").splitlines() == [legacy]


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_unreadable_file_is_never_overwritten(backend_cls):
    """A file that cannot be decoded disables the store instead of erasing it."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        seed: BaseActionItemStore = backend_cls()
        await seed.start()
        await seed.close()

        target = seed._file
        original = b"\xff\xfe not valid utf-8\n"
        target.write_bytes(original)

        broken: BaseActionItemStore = backend_cls()
        await broken.start()

        with pytest.raises(RuntimeError):
            await broken.query()
        with pytest.raises(RuntimeError):
            await broken.get("anything")
        with pytest.raises(RuntimeError):
            await broken.upsert([make_item()])
        await broken.close()

        assert target.read_bytes() == original


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_missing_file_starts_empty(backend_cls):
    """A workspace without records queries as an empty list."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()
        assert await store.query() == []
        await store.close()


def test_id_is_stable_across_whitespace_and_case():
    """The id depends on normalized text, so reruns reuse the same record."""

    first = action_item_id(NOTE, "Cut the 2.3.1 release")
    second = action_item_id(NOTE, "  cut   the 2.3.1 RELEASE ")
    assert first == second
    assert action_item_id(NOTE, "Different work") != first


def test_construction_touches_no_files():
    """Building the store must not create directories as a side effect."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore()
        assert not store.component_metadata_path.exists()


@pytest.mark.asyncio
async def test_written_file_is_jsonl():
    """The durable record is plain JSONL: one object per line, no BOM, LF, id-sorted."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore()
        await store.start()
        await store.upsert([make_item(description="Beta work"), make_item(description="Alpha work")])
        await store.close()

        assert store.component_metadata_path.is_relative_to(store.workspace_path)
        raw = store._file.read_bytes()

        assert not raw.startswith(b"\xef\xbb\xbf")
        assert raw.endswith(b"\n")
        assert b"\r\n" not in raw

        lines = raw.decode("utf-8").splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert [record["id"] for record in records] == sorted(record["id"] for record in records)
        assert {record["description"] for record in records} == {"Alpha work", "Beta work"}
        assert all("aliases" in record for record in records)


@pytest.mark.asyncio
async def test_write_leaves_no_temporary_file():
    """The atomic replace consumes its staging file."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore()
        await store.start()
        await store.upsert([make_item()])
        await store.close()

        assert not list(store._file.parent.glob("*.tmp"))


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_older_evidence_does_not_reopen_completed_item(backend_cls):
    """A reworded description backed by older evidence must not resurrect resolved work.

    ``last_updated`` moves with any edit, so the state has to be decided by the
    newest supporting observation instead of by the record timestamp.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()
        resolved_at = BASE + timedelta(days=5)
        completed = make_item(state=ActionItemState.COMPLETED, observed_at=resolved_at)
        await store.upsert([completed])

        edited_at = resolved_at + timedelta(days=5)
        reworded = make_item(
            description="Cut the 2.3.1 release and announce it",
            observed_at=BASE,  # observed before the completion
            updated_at=edited_at,  # only the wording changed
            aliases=[completed.id],
        )
        await store.upsert([reworded])

        stored = await store.get(completed.id)
        assert stored is not None
        assert stored.state is ActionItemState.COMPLETED
        assert stored.description == "Cut the 2.3.1 release and announce it"
        assert stored.last_updated == edited_at
        await store.close()


@pytest.mark.asyncio
async def test_named_stores_do_not_share_a_file():
    """Two independently named stores keep separate durable records."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        work = LocalActionItemStore(name="work")
        personal = LocalActionItemStore(name="personal")
        await work.start()
        await personal.start()
        await work.upsert([make_item(description="Ship the work review")])
        await personal.upsert([make_item(description="Book the personal trip")])

        assert work._file != personal._file
        await work.close()
        await personal.close()

        reopened = LocalActionItemStore(name="work")
        await reopened.start()
        assert [item.description for item in await reopened.query()] == ["Ship the work review"]
        await reopened.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_alias_link_reconciles_two_stored_records(backend_cls):
    """An alias pointing at another stored record folds both into one identity.

    Keeping them apart leaves two states for the same work: the alias-linked
    copy would never see the later resolution.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()
        earlier = make_item(description="Task A")
        later = make_item(description="Task B", observed_at=BASE + timedelta(hours=1))
        await store.upsert([earlier])
        await store.upsert([later])

        resolved = make_item(
            description="Task B",
            state=ActionItemState.COMPLETED,
            observed_at=BASE + timedelta(days=2),
            aliases=[earlier.id],
        )
        await store.upsert([resolved])

        rows = await store.query()
        assert len(rows) == 1
        assert rows[0].id == earlier.id  # the oldest identity keeps the record
        assert rows[0].state is ActionItemState.COMPLETED
        assert later.id in rows[0].aliases
        assert await store.get(earlier.id) is not None
        assert await store.get(later.id) is not None
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_failed_write_leaves_memory_unchanged(backend_cls, monkeypatch):
    """A write that cannot reach the disk must not publish the new state in memory."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()
        item = make_item()
        await store.upsert([item])

        def failing_replace(*_args, **_kwargs):
            raise OSError("simulated rename failure")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            await store.upsert([make_item(state=ActionItemState.COMPLETED, observed_at=BASE + timedelta(days=1))])
        monkeypatch.undo()

        stored = await store.get(item.id)
        assert stored is not None
        assert stored.state is ActionItemState.UNRESOLVED
        assert not list(store._file.parent.glob("*.tmp"))
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_query_returns_detached_records(backend_cls):
    """Mutating a returned record must not reach the stored one."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store: BaseActionItemStore = backend_cls()
        await store.start()
        item = make_item()
        await store.upsert([item])

        rows = await store.query()
        rows[0].state = ActionItemState.COMPLETED
        rows[0].evidence.clear()
        rows[0].aliases.append("ai_injected")

        stored = await store.get(item.id)
        assert stored is not None
        assert stored.state is ActionItemState.UNRESOLVED
        assert stored.evidence
        assert stored.aliases == []
        await store.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_unknown_fields_survive_a_round_trip(backend_cls):
    """Fields written by a newer version survive read-modify-write.

    ``close`` always rewrites the file, so parsing lossily would erase them on
    the first plain start-query-close cycle.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        seed: BaseActionItemStore = backend_cls()
        await seed.start()
        await seed.upsert([make_item()])
        await seed.close()

        target = seed._file
        payload = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
        payload["sla_hint"] = "two weeks"
        payload["evidence"][0]["upstream_ref"] = "workflow/notes#12"
        target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

        reader: BaseActionItemStore = backend_cls()
        await reader.start()
        loaded = (await reader.query())[0]
        assert loaded.sla_hint == "two weeks"
        assert loaded.evidence[0].upstream_ref == "workflow/notes#12"
        await reader.close()

        rewritten = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
        assert rewritten["sla_hint"] == "two weeks"
        assert rewritten["evidence"][0]["upstream_ref"] == "workflow/notes#12"


@pytest.mark.parametrize(
    "file_name",
    ["../action_items.jsonl", "nested/action_items.jsonl", "/absolute/action_items.jsonl"],
)
def test_store_rejects_path_like_file_name(file_name):
    """A file name that could leave the metadata directory is refused up front."""

    with pytest.raises(ValueError):
        LocalActionItemStore(file_name=file_name)


@pytest.mark.asyncio
async def test_staging_symlink_cannot_redirect_the_write():
    """A symlink pre-created at the staging path must not become the write target.

    The staging file is created exclusively under a random name, so an existing
    link at a predictable name is never opened. Following it would truncate and
    rewrite a file outside the workspace before the rename moved the link into
    place.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore()
        await store.start()
        await store.upsert([make_item()])

        outside = Path(tmpdir) / "outside.txt"
        outside.write_text("user-owned content", encoding="utf-8")
        staging = store._file.with_name(f"{store._file.name}.tmp")
        try:
            staging.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are unavailable on this platform")

        await store.upsert([make_item(description="Ship the follow-up")])

        assert outside.read_text(encoding="utf-8") == "user-owned content"
        assert store._file.is_file()
        assert not store._file.is_symlink()
        await store.close()


@pytest.mark.asyncio
async def test_reader_close_does_not_erase_newer_records():
    """Closing an unchanged reader must not rewrite its stale snapshot.

    A reader that only loaded the file holds no new information; replacing the
    file on close would silently drop records another instance appended after
    that load.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        writer = LocalActionItemStore()
        await writer.start()
        first = make_item(description="Ship the review")
        await writer.upsert([first])

        reader = LocalActionItemStore()
        await reader.start()
        assert [item.description for item in await reader.query()] == ["Ship the review"]

        second = make_item(description="Book the venue", observed_at=BASE + timedelta(hours=1))
        await writer.upsert([second])
        await reader.close()

        reopened = LocalActionItemStore()
        await reopened.start()
        assert [item.description for item in await reopened.query()] == [
            "Book the venue",
            "Ship the review",
        ]
        await reopened.close()
        await writer.close()


@pytest.mark.asyncio
async def test_writer_refuses_to_overwrite_a_file_changed_since_load():
    """A write over a file another instance changed must fail loudly, not lose rows."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        writer = LocalActionItemStore()
        await writer.start()
        first = make_item(description="Ship the review")
        await writer.upsert([first])

        stale = LocalActionItemStore()
        await stale.start()
        second = make_item(description="Book the venue", observed_at=BASE + timedelta(hours=1))
        await writer.upsert([second])

        with pytest.raises(RuntimeError):
            await stale.upsert([make_item(description="Update the changelog")])

        reopened = LocalActionItemStore()
        await reopened.start()
        assert [item.description for item in await reopened.query()] == [
            "Book the venue",
            "Ship the review",
        ]
        await reopened.close()
        await stale.close()
        await writer.close()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "sub/dir", "C:\\drive"])
@pytest.mark.asyncio
async def test_component_name_must_be_a_single_path_segment(name):
    """A component name that could select another directory is refused at start.

    The store directory is scoped by name, so an absolute or relative name would
    point reads and writes outside the metadata root without touching the
    file-name guard.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore(name=name)
        with pytest.raises(ValueError):
            await store.start()


@pytest.mark.asyncio
async def test_symlinked_metadata_root_is_rejected():
    """A metadata directory redirected outside the workspace is not trusted as the root."""

    with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as elsewhere:
        with temp_chdir(workspace):
            root = Path("metadata") / "action_item_store"
            root.parent.mkdir(parents=True)
            outside = Path(elsewhere)
            try:
                root.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks are unavailable on this platform")

            store = LocalActionItemStore()
            with pytest.raises(ValueError):
                await store.start()
            assert not list(outside.iterdir())


@pytest.mark.asyncio
async def test_duplicate_evidence_keeps_new_extension_fields():
    """Merging a repeated observation must not drop metadata only it supplies.

    The evidence entry is deduplicated by its identity fields, so discarding the
    incoming row as a whole would lose extension fields a later writer added.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore()
        await store.start()
        item = make_item()
        await store.upsert([item])

        enriched = ActionItemEvidence(
            path=NOTE,
            locator="## Next",
            observed_at=BASE,
            source_event_id="event-123",
            owner="release-team",
        )
        await store.upsert([item.model_copy(update={"evidence": [enriched]})])

        stored = await store.get(item.id)
        assert stored is not None
        assert len(stored.evidence) == 1
        assert stored.evidence[0].source_event_id == "event-123"
        assert stored.evidence[0].owner == "release-team"

        rewritten = json.loads(store._file.read_text(encoding="utf-8").splitlines()[0])
        assert rewritten["evidence"][0]["source_event_id"] == "event-123"
        await store.close()


# ---------------------------------------------------------------------------
# Coordination between writers, and read-target containment
# (review: ployts, 2026-11-20)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_writer_refuses_instead_of_dropping_records():
    """Two stores over one file must not silently write past each other.

    Both load the same snapshot, then each adds a record. A check that ends
    before the other writer's replace cannot see that write, and the second
    commit would drop it while reporting success. The replacement is now
    serialised with the check, so the loser fails loudly instead.
    """

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        first = LocalActionItemStore()
        await first.start()
        await first.upsert([make_item(description="Base")])

        second = LocalActionItemStore()
        await second.start()

        await first.upsert([make_item(description="B")])
        with pytest.raises(RuntimeError, match="changed on disk"):
            await second.upsert([make_item(description="C")])

        reader = LocalActionItemStore()
        await reader.start()
        descriptions = {item.description for item in await reader.query()}
        assert {"Base", "B"} <= descriptions
        assert "C" not in descriptions

        await first.close()
        await second.close()
        await reader.close()


@pytest.mark.asyncio
async def test_a_file_rewritten_after_load_is_not_overwritten():
    """A version that arrived after this instance parsed the file survives."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        writer = LocalActionItemStore()
        await writer.start()
        await writer.upsert([make_item(description="A")])

        reader = LocalActionItemStore()
        await reader.start()

        await writer.upsert([make_item(description="B")])

        with pytest.raises(RuntimeError, match="changed on disk"):
            await reader.upsert([make_item(description="C")])

        check = LocalActionItemStore()
        await check.start()
        descriptions = {item.description for item in await check.query()}
        assert {"A", "B"} <= descriptions
        assert "C" not in descriptions

        await writer.close()
        await reader.close()
        await check.close()


@pytest.mark.asyncio
async def test_a_symlinked_store_directory_is_not_read():
    """A named store directory that resolves outside the workspace is refused."""

    with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as elsewhere:
        with temp_chdir(workspace):
            root = Path("metadata") / "action_item_store"
            root.mkdir(parents=True)
            outside = Path(elsewhere)
            (outside / "action_items.jsonl").write_text(
                json.dumps(make_item().model_dump(mode="json")) + "\n",
                encoding="utf-8",
            )
            try:
                (root / "work").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks are unavailable on this platform")

            store = LocalActionItemStore(name="work")
            with pytest.raises(ValueError):
                await store.start()


@pytest.mark.asyncio
async def test_a_symlinked_store_file_is_not_read():
    """A regular directory holding a linked JSONL file is refused as well."""

    with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as elsewhere:
        with temp_chdir(workspace):
            store_dir = Path("metadata") / "action_item_store" / "work"
            store_dir.mkdir(parents=True)
            outside = Path(elsewhere) / "action_items.jsonl"
            outside.write_text(
                json.dumps(make_item().model_dump(mode="json")) + "\n",
                encoding="utf-8",
            )
            try:
                (store_dir / "action_items.jsonl").symlink_to(outside)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks are unavailable on this platform")

            store = LocalActionItemStore(name="work")
            with pytest.raises(ValueError):
                await store.start()


@pytest.mark.asyncio
async def test_the_check_and_the_replace_run_inside_the_store_lock(monkeypatch):
    """Validation and replacement must be one protected operation.

    Two stores over one file each hold their own snapshot. If the check ends
    before the other writer's replace, it cannot see that write and the commit
    that follows drops it. The order below is what keeps that window shut.
    """

    from contextlib import contextmanager

    from reme.components.action_item_store import local_action_item_store as store_module

    events: list[str] = []
    real_lock = store_module._exclusive_lock
    real_check = LocalActionItemStore._assert_unchanged
    real_replace = store_module.os.replace

    @contextmanager
    def tracked_lock(path):
        events.append("lock-enter")
        with real_lock(path):
            yield
        events.append("lock-exit")

    def tracked_check(self, destination):
        events.append("check")
        return real_check(self, destination)

    def tracked_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(store_module, "_exclusive_lock", tracked_lock)
    monkeypatch.setattr(LocalActionItemStore, "_assert_unchanged", tracked_check)
    monkeypatch.setattr(store_module.os, "replace", tracked_replace)

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore()
        await store.start()
        await store.upsert([make_item()])
        await store.close()

    assert events == ["lock-enter", "check", "replace", "lock-exit"]
