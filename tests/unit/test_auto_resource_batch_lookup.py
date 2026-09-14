"""Reuse initial ownership for distinct resources and live lookup for repeated sources."""

import asyncio
import os
from collections import Counter
from unittest.mock import patch

import frontmatter
import pytest

from reme.components import R
from reme.components.runtime_context import RuntimeContext
from reme.schema import Response
from reme.steps.evolve import base_auto_resource as resource_module
from reme.steps.evolve.auto_resource import AutoResourceStep
from reme.steps.evolve.auto_text_resource import AutoTextResourceStep
from reme.steps.evolve.base_auto_resource import BaseAutoResourceStep

from .auto_resource_test_support import FakeVisionModel, caption_json, image_bytes

pytest_plugins = ("unit.auto_resource_test_plugin",)
pytestmark = pytest.mark.asyncio

_HISTORY_DAYS = ("2025-12-01", "2025-12-02", "2025-12-03")


def _seed_history(env) -> None:
    """Create older days untouched by the resource changes under test."""
    for day in _HISTORY_DAYS:
        env.write_note(f"daily/{day}/archive.md", f"[[resource/archive-{day}.txt]]")


def _observe_history(env, monkeypatch) -> Counter:
    """Observe directory walks, parsing, and daily_list, not every possible standalone stat call."""
    calls = Counter()
    daily_root = (env.workspace / "daily").resolve()
    history_text = {
        path.read_text(encoding="utf-8"): day
        for day in _HISTORY_DAYS
        if (path := daily_root / day / "archive.md").is_file()
    }
    watched_directories = {
        str(daily_root): "history_enumerations",
        **{str(daily_root / day): ("enumerate", day) for day in _HISTORY_DAYS},
    }
    original_listdir = os.listdir
    original_scandir = os.scandir
    original_loads = frontmatter.loads
    daily_list = env.app_context.jobs["daily_list"]

    def record_enumeration(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            key = watched_directories.get(os.fsdecode(path))
            if key is not None:
                calls[key] += 1

    def counted_listdir(path="."):
        record_enumeration(path)
        return original_listdir(path)

    def counted_scandir(path="."):
        record_enumeration(path)
        return original_scandir(path)

    def counted_loads(text, *args, **kwargs):
        if isinstance(text, str) and text in history_text:
            calls[("parse", history_text[text])] += 1
        return original_loads(text, *args, **kwargs)

    async def counted_daily_list(**kwargs):
        calls[("daily_list", kwargs["date"])] += 1
        return await daily_list(**kwargs)

    # Path.iterdir uses listdir or scandir depending on the Python version.
    # Return the original iterator unchanged, without wrapping DirEntry objects.
    monkeypatch.setattr(os, "listdir", counted_listdir)
    monkeypatch.setattr(os, "scandir", counted_scandir)
    monkeypatch.setattr(frontmatter, "loads", counted_loads)
    monkeypatch.setitem(env.app_context.jobs, "daily_list", counted_daily_list)
    return calls


def _assert_history_reads(calls, count=1) -> None:
    assert calls["history_enumerations"] == count
    for day in _HISTORY_DAYS:
        assert calls[("enumerate", day)] == count
        assert calls[("daily_list", day)] == count
        assert calls[("parse", day)] == count


def _model() -> FakeVisionModel:
    return FakeVisionModel(caption_json("caption", "A sample image", "A red square."))


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
@pytest.mark.parametrize("resource_count", [1, 8])
@pytest.mark.parametrize("write_notes", [False, True], ids=["missing-deletions", "real-updates"])
async def test_loose_batch_reads_history_once(
    routed,
    resource_count,
    write_notes,
    auto_resource_env,
    monkeypatch,
):
    """More resources, including actual note writes, must not cause another history walk."""
    env = auto_resource_env
    _seed_history(env)
    changes = []
    notes = []
    for index in range(resource_count):
        source = f"resource/photo-{index}.png"
        if write_notes:
            env.write_binary(source, image_bytes())
            notes.append(env.write_note(f"daily/2026-01-01/photo-{index}.md", f"[[{source}]]"))
        changes.append({"change": "modified" if write_notes else "deleted", "path": source})
    before = {note: note.read_bytes() for note in notes}
    calls = _observe_history(env, monkeypatch)

    with patch.object(BaseAutoResourceStep, "_today", return_value="2026-01-01"):
        response = await env.run(env.processor(_model(), routed=routed), changes)

    assert response.success is True
    assert response.metadata["processed"] == resource_count
    results = response.metadata["results"]
    assert [result["path"] for result in results] == [item["path"] for item in changes]
    assert all(result["metadata"]["action"] == ("modified" if write_notes else "skipped") for result in results)
    _assert_history_reads(calls)
    if write_notes:
        assert all(note.read_bytes() != previous for note, previous in before.items())
        assert all(result["metadata"]["modified"] for result in results)
        # All existing notes share one already-read day. Pre-write lookup must
        # reuse it; normal day-index rebuilds are not daily_list calls.
        assert calls[("daily_list", "2026-01-01")] == 1


async def test_mixed_processors_share_history_and_keep_result_order(auto_resource_env, monkeypatch):
    """Both processors reuse the batch lookup while returning results in input order."""
    env = auto_resource_env
    _seed_history(env)
    text_note = env.write_note("daily/2026-01-01/report.md", "[[resource/report.txt]]")
    image_note = env.write_note("daily/2026-01-01/photo.md", "[[resource/photo.png]]")
    calls = _observe_history(env, monkeypatch)
    changes = [
        {"change": "deleted", "path": "resource/report.txt"},
        {"change": "deleted", "path": "resource/photo.png"},
    ]

    response = await env.run(env.processor(_model(), routed=True), changes)

    assert response.success is True
    results = response.metadata["results"]
    assert [result["path"] for result in results] == [item["path"] for item in changes]
    assert all(result["metadata"]["action"] == "deleted" for result in results)
    assert not text_note.exists()
    assert not image_note.exists()
    _assert_history_reads(calls)
    assert calls[("daily_list", "2026-01-01")] == 1


@pytest.mark.parametrize("routed", [False, True], ids=["text-step", "text-only-router"])
async def test_text_only_batch_uses_history_without_image_processor(routed, auto_resource_env, monkeypatch):
    """A text-only pipeline can reuse history without configuring an image processor."""
    env = auto_resource_env
    _seed_history(env)
    env.app_context.registry = R.copy()
    notes = [env.write_note(f"daily/2026-01-01/report-{i}.md", f"[[resource/report-{i}.txt]]") for i in range(3)]
    calls = _observe_history(env, monkeypatch)
    options = {"app_context": env.app_context, "file_store": env.file_store}
    step = (
        AutoResourceStep(**options, dispatch_steps=["auto_text_resource_step"])
        if routed
        else AutoTextResourceStep(**options)
    )

    response = await env.run(step, [{"change": "deleted", "path": f"resource/report-{i}.txt"} for i in range(3)])

    assert response.success is True
    assert all(not note.exists() for note in notes)
    _assert_history_reads(calls)
    assert calls[("daily_list", "2026-01-01")] == 1


@pytest.mark.parametrize("dated", [False, True], ids=["loose-root", "dated-resource"])
async def test_new_image_uses_live_post_write_resolution(dated, auto_resource_env, monkeypatch):
    """Only dated input needs a pre-write daily_list; both paths read the newly written note."""
    env = auto_resource_env
    _seed_history(env)
    prefix = "resource/2026-01-01" if dated else "resource"
    source = env.write_binary(f"{prefix}/caption.png", image_bytes())
    calls = _observe_history(env, monkeypatch)
    plain_call = FakeVisionModel.__call__

    async def observe_pre_write(model, messages, **kwargs):
        assert calls[("daily_list", "2026-01-01")] == (1 if dated else 0)
        return await plain_call(model, messages, **kwargs)

    monkeypatch.setattr(FakeVisionModel, "__call__", observe_pre_write)
    with patch.object(BaseAutoResourceStep, "_today", return_value="2026-01-01"):
        response = await env.run(
            env.processor(_model(), routed=True),
            [{"change": "added", "path": str(source)}],
        )

    assert response.success is True
    path = response.metadata["results"][0]["metadata"]["path"]
    assert path == "daily/2026-01-01/caption.md"
    assert (env.workspace / path).is_file()
    _assert_history_reads(calls, 0 if dated else 1)
    assert calls[("daily_list", "2026-01-01")] == (2 if dated else 1)


async def test_reused_step_and_context_rebuild_history_next_invocation(auto_resource_env):
    """External edits between invocations are visible; batch state never survives the call."""
    env = auto_resource_env
    original = env.write_note("daily/2026-01-01/original.md", "[[resource/photo.png]]")
    step = env.processor(_model(), routed=True)
    changes = [{"change": "deleted", "path": "resource/photo.png"}]
    context = RuntimeContext(changes=changes, user_value="preserve")
    context_keys = set(context.data)

    first = await step(context)
    assert first.success is True
    assert first.metadata["results"][0]["metadata"]["path"] == "daily/2026-01-01/original.md"
    assert not original.exists()
    assert set(context.data) == context_keys
    assert context["changes"] is changes
    assert context["user_value"] == "preserve"

    recreated = env.write_note("daily/2026-01-02/recreated.md", "[[resource/photo.png]]")
    second = await step(context)

    assert second.success is True
    assert second.metadata["results"][0]["metadata"]["path"] == "daily/2026-01-02/recreated.md"
    assert second.metadata["results"][0]["metadata"]["action"] == "deleted"
    assert not recreated.exists()
    assert set(context.data) == context_keys
    assert context["changes"] is changes
    assert context["user_value"] == "preserve"


async def test_create_update_delete_recreate_maintains_ownership_across_midnight(
    auto_resource_env,
    monkeypatch,
):
    """A card keeps its original day until deletion permits a fresh card on today's date."""
    env = auto_resource_env
    env.write_binary("resource/caption.png", image_bytes())
    clock = {"day": "2026-01-01"}
    refresh = resource_module.refresh_day_index

    async def refresh_and_advance(*args, **kwargs):
        result = await refresh(*args, **kwargs)
        clock["day"] = "2026-01-02"
        return result

    monkeypatch.setattr(BaseAutoResourceStep, "_today", lambda _step: clock["day"])
    monkeypatch.setattr(resource_module, "refresh_day_index", refresh_and_advance)
    response = await env.run(
        env.processor(_model(), routed=True),
        [{"change": change, "path": "resource/caption.png"} for change in ("added", "modified", "deleted", "added")],
    )

    assert response.success is True
    metadata = [result["metadata"] for result in response.metadata["results"]]
    assert [item["action"] for item in metadata] == ["added", "modified", "deleted", "added"]
    assert [item["path"] for item in metadata] == [
        "daily/2026-01-01/caption.md",
        "daily/2026-01-01/caption.md",
        "daily/2026-01-01/caption.md",
        "daily/2026-01-02/caption.md",
    ]
    assert not (env.workspace / "daily/2026-01-01/caption.md").exists()
    assert (env.workspace / "daily/2026-01-02/caption.md").is_file()


@pytest.mark.parametrize("first_change", ["added", "deleted"])
async def test_repeated_source_retries_after_partial_write_or_delete(
    first_change,
    auto_resource_env,
    monkeypatch,
):
    """A repeated source uses live lookup even when its previous event failed after mutation."""
    env = auto_resource_env
    env.write_binary("resource/caption.png", image_bytes())
    if first_change == "deleted":
        env.write_note("daily/2026-01-01/caption.md", "[[resource/caption.png]]")
    clock = {"day": "2026-01-01"}
    refresh = resource_module.refresh_day_index
    refresh_calls = 0

    async def fail_first_refresh(*args, **kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            clock["day"] = "2026-01-02"
            raise RuntimeError("day index unavailable after disk change")
        return await refresh(*args, **kwargs)

    monkeypatch.setattr(BaseAutoResourceStep, "_today", lambda _step: clock["day"])
    monkeypatch.setattr(resource_module, "refresh_day_index", fail_first_refresh)
    next_change = "modified" if first_change == "added" else "added"
    response = await env.run(
        env.processor(_model(), routed=True),
        [{"change": change, "path": "resource/caption.png"} for change in (first_change, next_change)],
    )

    failed, recovered = response.metadata["results"]
    assert response.success is False
    assert response.metadata["modified"] is True
    assert failed["success"] is False
    assert failed["metadata"]["modified"] is True
    assert "day index unavailable" in failed["metadata"]["error"]
    assert recovered["success"] is True
    day = "2026-01-01" if first_change == "added" else "2026-01-02"
    assert recovered["metadata"]["path"] == f"daily/{day}/caption.md"
    assert recovered["metadata"]["created"] is (first_change == "deleted")
    assert (env.workspace / f"daily/{day}/caption.md").is_file()
    assert len(list((env.workspace / "daily").glob("*/caption.md"))) == 1


async def test_repeated_existing_updates_keep_original_day_and_path(
    auto_resource_env,
    monkeypatch,
):
    """Non-consecutive updates recognize the same source in absolute and relative form."""
    env = auto_resource_env
    _seed_history(env)
    source = env.write_binary("resource/photo.png", image_bytes())
    note = env.write_note("daily/2026-01-01/original.md", "[[resource/photo.png]]")
    env.write_binary("resource/other.png", image_bytes())
    other = env.write_note("daily/2026-01-01/other.md", "[[resource/other.png]]")
    before = note.read_bytes()
    calls = _observe_history(env, monkeypatch)
    model = _model()
    changes = [
        {"change": "modified", "path": str(source)},
        {"change": "modified", "path": "resource/other.png"},
        {"change": "modified", "path": "resource/photo.png"},
    ]

    with patch.object(BaseAutoResourceStep, "_today", return_value="2026-01-02"):
        response = await env.run(env.processor(model, routed=True), changes)

    assert response.success is True
    metadata = [result["metadata"] for result in response.metadata["results"]]
    assert [item["path"] for item in metadata] == [
        "daily/2026-01-01/original.md",
        "daily/2026-01-01/other.md",
        "daily/2026-01-01/original.md",
    ]
    assert all(item["action"] == "modified" and item["created"] is False for item in metadata)
    assert len(model.calls) == 3
    assert note.read_bytes() != before
    assert other.is_file()
    assert not (env.workspace / "daily/2026-01-02").exists()
    # Unchanged paths alone cannot prove that the repeated source used live lookup.
    _assert_history_reads(calls, 2)


async def test_cross_day_duplicate_only_fails_ambiguous_resource(auto_resource_env):
    """Ambiguous ownership prevents mutation without blocking another valid resource."""
    env = auto_resource_env
    first = env.write_note("daily/2026-01-01/first.md", "[[resource/duplicate.png]]")
    second = env.write_note("daily/2026-01-02/second.md", "[[resource/duplicate.png]]")
    valid = env.write_note("daily/2026-01-02/valid.md", "[[resource/valid.png]]")
    before = {note: note.read_bytes() for note in (first, second)}
    response = await env.run(
        env.processor(_model(), routed=True),
        [
            {"change": "deleted", "path": "resource/duplicate.png"},
            {"change": "deleted", "path": "resource/valid.png"},
        ],
    )

    ambiguous, deleted = response.metadata["results"]
    assert response.success is False
    assert ambiguous["success"] is False
    assert "Multiple daily resource notes claim resource/duplicate.png" in ambiguous["metadata"]["error"]
    assert ambiguous["metadata"]["modified"] is False
    assert deleted["success"] is True
    assert deleted["metadata"]["action"] == "deleted"
    assert not valid.exists()
    assert all(note.read_bytes() == contents for note, contents in before.items())


async def test_same_day_duplicates_keep_first_match_after_deletion(auto_resource_env):
    """Deleting the first same-day match exposes the next one, preserving existing behavior."""
    env = auto_resource_env
    first = env.write_note("daily/2026-01-01/first.md", "[[resource/photo.png]]")
    second = env.write_note("daily/2026-01-01/second.md", "[[resource/photo.png]]")
    with patch.object(BaseAutoResourceStep, "_today", return_value="2026-01-02"):
        response = await env.run(
            env.processor(_model(), routed=True),
            [{"change": "deleted", "path": "resource/photo.png"} for _ in range(3)],
        )

    assert response.success is True
    metadata = [result["metadata"] for result in response.metadata["results"]]
    assert [item["action"] for item in metadata] == ["deleted", "deleted", "skipped"]
    assert [item["path"] for item in metadata[:2]] == ["daily/2026-01-01/first.md", "daily/2026-01-01/second.md"]
    assert not first.exists()
    assert not second.exists()


async def test_loose_and_dated_changes_keep_separate_lookup_modes(auto_resource_env):
    """A dated event must not inherit a loose event's cached path, including on repeated deletion."""
    env = auto_resource_env
    sources = ["resource/first.png", "resource/2026-01-02/dated.png", "resource/last.png"]
    notes = [
        env.write_note("daily/2026-01-01/first.md", f"[[{sources[0]}]]"),
        env.write_note("daily/2026-01-02/first.md", f"[[{sources[1]}]]"),
        env.write_note("daily/2026-01-02/second.md", f"[[{sources[1]}]]"),
        env.write_note("daily/2026-01-01/last.md", f"[[{sources[2]}]]"),
    ]

    response = await env.run(
        env.processor(_model(), routed=True),
        [{"change": "deleted", "path": source} for source in (sources[0], sources[1], sources[1], sources[2])],
    )

    assert response.success is True
    metadata = [result["metadata"] for result in response.metadata["results"]]
    assert [item["action"] for item in metadata] == ["deleted"] * 4
    assert [item["path"] for item in metadata] == [note.relative_to(env.workspace).as_posix() for note in notes]
    assert all(not note.exists() for note in notes)


async def test_initial_read_failure_does_not_publish_partial_history(auto_resource_env, monkeypatch):
    """A transient history read failure makes the next item retry the initial scan."""
    env = auto_resource_env
    _seed_history(env)
    owned_note = env.write_note("daily/2026-01-01/original.md", "[[resource/photo.png]]")
    daily_list = env.app_context.jobs["daily_list"]
    failed_once = False

    async def fail_once_mid_scan(**kwargs):
        nonlocal failed_once
        if kwargs["date"] == _HISTORY_DAYS[1] and not failed_once:
            failed_once = True
            return Response(success=False, answer="temporary history read failure")
        return await daily_list(**kwargs)

    monkeypatch.setitem(env.app_context.jobs, "daily_list", fail_once_mid_scan)
    response = await env.run(
        env.processor(_model(), routed=True),
        [
            {"change": "deleted", "path": "resource/missing.png"},
            {"change": "deleted", "path": "resource/photo.png"},
        ],
    )

    failed, recovered = response.metadata["results"]
    assert response.success is False
    assert failed["success"] is False
    assert failed["metadata"]["modified"] is False
    assert "temporary history read failure" in failed["metadata"]["error"]
    assert recovered["success"] is True
    assert recovered["metadata"]["path"] == "daily/2026-01-01/original.md"
    assert recovered["metadata"]["action"] == "deleted"
    assert not owned_note.exists()


async def test_empty_history_is_initialized_once(auto_resource_env, monkeypatch):
    """An empty ownership table is a successful scan, not an uninitialized cache."""
    env = auto_resource_env
    (env.workspace / "daily").mkdir(exist_ok=True)
    calls = _observe_history(env, monkeypatch)

    response = await env.run(
        env.processor(_model(), routed=True),
        [{"change": "deleted", "path": f"resource/missing-{index}.png"} for index in range(8)],
    )

    assert response.success is True
    assert all(result["metadata"]["action"] == "skipped" for result in response.metadata["results"])
    assert calls["history_enumerations"] == 1


async def test_router_exception_restores_context_and_next_call_rebuilds(auto_resource_env):
    """Even an exception after dispatch cannot leak batch state into a reused context."""
    env = auto_resource_env
    original = env.write_note("daily/2026-01-01/original.md", "[[resource/photo.png]]")
    step = env.processor(_model(), routed=True)
    changes = [{"change": "deleted", "path": "resource/photo.png"}]
    context = RuntimeContext(changes=changes, user_value="preserve")
    context_keys = set(context.data)
    dispatch = step._dispatch_processor  # pylint: disable=protected-access

    async def dispatch_then_fail(*args, **kwargs):
        await dispatch(*args, **kwargs)
        raise RuntimeError("dispatch interrupted after processor completion")

    with patch.object(step, "_dispatch_processor", side_effect=dispatch_then_fail):
        with pytest.raises(RuntimeError, match="dispatch interrupted"):
            await step(context)

    assert not original.exists()
    assert set(context.data) == context_keys
    assert context["changes"] is changes
    assert context["user_value"] == "preserve"
    recreated = env.write_note("daily/2026-01-02/recreated.md", "[[resource/photo.png]]")
    response = await step(context)

    assert response.success is True
    assert response.metadata["results"][0]["metadata"]["path"] == "daily/2026-01-02/recreated.md"
    assert not recreated.exists()
    assert set(context.data) == context_keys


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
async def test_cancelled_invocation_restores_context_and_rebuilds(routed, auto_resource_env):
    """Cancel at the model boundary after lookup, without mocking the resource processing path."""
    env = auto_resource_env
    env.write_binary("resource/photo.png", image_bytes())
    original = env.write_note("daily/2026-01-01/original.md", "[[resource/photo.png]]")
    before = original.read_bytes()
    entered = asyncio.Event()

    # The inherited structured method deliberately raises to exercise plain-call fallback.
    class WaitingModel(FakeVisionModel):  # pylint: disable=abstract-method
        """Keep the image request pending until the invocation is cancelled."""

        async def __call__(self, messages, **kwargs):
            del messages, kwargs
            entered.set()
            await asyncio.Event().wait()

    step = env.processor(WaitingModel("unused"), routed=routed)
    changes = [{"change": "modified", "path": "resource/photo.png"}]
    context = RuntimeContext(changes=changes, user_value="preserve")
    context_keys = set(context.data)
    task = asyncio.create_task(step(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert original.read_bytes() == before
    assert set(context.data) == context_keys
    assert context["changes"] is changes
    assert context["user_value"] == "preserve"
    original.unlink()
    recreated = env.write_note("daily/2026-01-02/recreated.md", "[[resource/photo.png]]")
    context["changes"] = [{"change": "deleted", "path": "resource/photo.png"}]
    response = await step(context)

    assert response.success is True
    assert response.metadata["results"][0]["metadata"]["path"] == "daily/2026-01-02/recreated.md"
    assert not recreated.exists()
    assert set(context.data) == context_keys
