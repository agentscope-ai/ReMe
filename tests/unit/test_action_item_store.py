"""Tests for the action-item contract and its local store."""

# pylint: disable=redefined-outer-name

from contextlib import chdir as temp_chdir
from datetime import datetime, timedelta, timezone
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
) -> ActionItem:
    """Build an action item with a stable id for the given note and text."""
    return ActionItem(
        id=action_item_id(note, description),
        note=note,
        description=description,
        state=state,
        evidence=[evidence(observed_at)],
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
async def test_corrupt_line_is_skipped(backend_cls):
    """One bad record does not hide the valid ones on reload."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        first: BaseActionItemStore = backend_cls()
        await first.start()
        item = make_item()
        await first.upsert([item])
        await first.close()

        target = first._file  # pylint: disable=protected-access
        with target.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")

        second: BaseActionItemStore = backend_cls()
        await second.start()
        assert [i.id for i in await second.query()] == [item.id]
        await second.close()


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


def test_written_file_is_jsonl():
    """The durable record stays a plain JSONL file in the workspace metadata."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        store = LocalActionItemStore()
        assert store.component_metadata_path.is_relative_to(store.workspace_path)
