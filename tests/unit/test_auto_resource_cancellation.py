"""Cancellation retains real tool writes and completed resource results without retrying."""

# pylint: disable=protected-access

import asyncio
import uuid
from contextlib import asynccontextmanager
from unittest.mock import patch

import frontmatter
import pytest

from reme.components.runtime_context import RuntimeContext
from reme.steps.evolve.auto_image_resource import AutoImageResourceStep
from reme.steps.evolve.auto_text_resource import AutoTextResourceStep
from reme.steps.file_io import MoveStep
from reme.utils.wikilink_handler import WikilinkHandler

from .auto_resource_test_support import FakeAgentWrapper, FakeImageAgentWrapper, caption_fields, image_bytes

pytest_plugins = ("unit.auto_resource_test_plugin",)
pytestmark = pytest.mark.asyncio


class PausingImageAgentWrapper(FakeImageAgentWrapper):
    """Pause one real file-tool invocation immediately before or after its write."""

    def __init__(self, sources, *, after_write=True, pause_at=1):
        super().__init__(caption_fields("", "New description", "New caption."))
        self.sources = sources
        self.after_write = after_write
        self.pause_at = pause_at
        self.paused = asyncio.Event()

    async def reply(self, inputs, **kwargs):
        source = self.sources[len(self.calls)]
        self.note_metadata = {"source_resource": f"[[{source}]]"}
        if len(self.calls) + 1 == self.pause_at:
            if self.after_write:
                await super().reply(inputs, **kwargs)
            else:
                self.calls.append((inputs, kwargs))
            self.paused.set()
            await asyncio.Event().wait()
        return await super().reply(inputs, **kwargs)


@asynccontextmanager
async def running_step(step, context):
    """Never leave the test's own task running after an assertion failure."""
    task = asyncio.create_task(step(context))
    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert "_auto_resource_lookup_table" not in context


async def assert_cancelled(task):
    """The original cancellation must reach the caller within the cleanup bound."""
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)
    assert task.cancelled()


@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
@pytest.mark.parametrize("after_write", [False, True], ids=["before-write", "after-write"])
async def test_cancelled_image_reports_actual_note_state(existing, after_write, auto_resource_env):
    """Cancellation preserves actual new/update writes and still reaches the caller."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/photo.png"
    source = env.write_binary(source_path, image_bytes())
    note_path = "daily/2026-01-01/photo.md"
    note = env.workspace / note_path
    before = env.write_note(note_path, f"[[{source_path}]]").read_bytes() if existing else None
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    wrapper = PausingImageAgentWrapper([source_path], after_write=after_write)
    changes = [{"change": "modified" if existing else "added", "path": str(source)}]
    context = RuntimeContext(changes=changes)
    async with running_step(env.processor(wrapper, routed=True), context) as task:
        await asyncio.wait_for(wrapper.paused.wait(), timeout=5)
        task.cancel()
        await assert_cancelled(task)
    metadata = context.response.metadata
    result = metadata["results"][0]
    assert context.response.success is False
    assert metadata["cancelled"] is True
    assert metadata["unprocessed"] == []
    assert metadata["modified"] is after_write
    assert result["success"] is False
    assert result["metadata"]["action"] == "failed"
    assert result["metadata"]["reason"] == "cancelled"
    assert result["metadata"]["modified"] is after_write
    assert len(wrapper.calls) == 1
    assert len(hooks) == int(after_write)
    if after_write:
        assert note.read_bytes() != before
        post = frontmatter.load(note)
        assert post["source_resource"] == f"[[{source_path}]]"
        assert post["kind"] == "image" and post["media_type"] == "image/png"
        assert "New description" in (env.workspace / "daily/2026-01-01.md").read_text(encoding="utf-8")
    else:
        assert (note.read_bytes() if note.exists() else None) == before
        assert not (env.workspace / "daily/2026-01-01.md").exists()


async def test_cancelled_batch_keeps_completed_results_and_never_starts_remaining_changes(auto_resource_env):
    """Keep completed items when a later image stops its processor and the remaining routes."""
    env = auto_resource_env
    sources = [f"resource/2026-01-01/{name}.png" for name in ("first", "paused", "later")]
    for source in sources:
        env.write_binary(source, image_bytes())
    text_path = "resource/2026-01-01/later.txt"
    env.write_binary(text_path, b"must not be processed")
    changes = [{"change": "added", "path": source} for source in [*sources, text_path]]
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    wrapper = PausingImageAgentWrapper(sources, pause_at=2)
    step = env.processor(wrapper, routed=True)
    context = RuntimeContext(changes=changes)
    async with running_step(step, context) as task:
        await asyncio.wait_for(wrapper.paused.wait(), timeout=5)
        task.cancel()
        await assert_cancelled(task)
    metadata = context.response.metadata
    assert not context.response.success
    assert metadata["cancelled"] is True
    assert metadata["modified"] is True
    assert metadata["processed"] == 2
    assert [item["path"] for item in metadata["results"]] == sources[:2]
    assert [item["success"] for item in metadata["results"]] == [True, False]
    assert metadata["results"][1]["metadata"]["reason"] == "cancelled"
    assert metadata["unprocessed"] == changes[2:]
    assert context.get("changes") == changes
    assert len(hooks) == 1
    assert hooks[0]["kwargs"]["changes"] == changes
    assert hooks[0]["metadata"]["results"] == metadata["results"]
    assert len(wrapper.calls) == 2
    assert not step.dispatch_step_specs[-1]["agent_wrapper"].inputs
    assert (env.workspace / "daily/2026-01-01/first.md").exists()
    assert (env.workspace / "daily/2026-01-01/paused.md").exists()
    assert not (env.workspace / "daily/2026-01-01/later.md").exists()


async def test_text_processor_cancellation_preserves_completed_image_results(auto_resource_env):
    """A cancelled text route cannot overwrite the preceding image route's completed results."""
    env = auto_resource_env
    image_path = "resource/2026-01-01/photo.png"
    text_path = "resource/2026-01-01/waiting.txt"
    later_path = "resource/2026-01-01/later.txt"
    env.write_binary(image_path, image_bytes())
    env.write_binary(text_path, "中文文本".encode())
    env.write_binary(later_path, b"must not be processed")
    paused = asyncio.Event()
    text_calls = []

    class WaitingTextAgentWrapper(FakeAgentWrapper):
        """Keep the real text input contract while waiting at the external reply boundary."""

        async def reply(self, inputs, **kwargs):
            self.inputs = inputs
            text_calls.append(kwargs)
            paused.set()
            await asyncio.Event().wait()

    image_wrapper = FakeImageAgentWrapper()
    text_wrapper = WaitingTextAgentWrapper()
    step = env.processor(image_wrapper, routed=True)
    step.dispatch_step_specs[-1]["agent_wrapper"] = text_wrapper
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    changes = [{"change": "added", "path": path} for path in (image_path, text_path, later_path)]
    context = RuntimeContext(changes=changes)
    async with running_step(step, context) as task:
        await asyncio.wait_for(paused.wait(), timeout=5)
        task.cancel()
        await assert_cancelled(task)
    metadata = context.response.metadata
    assert not context.response.success
    assert metadata["cancelled"] is True
    assert metadata["modified"] is True
    assert [item["path"] for item in metadata["results"]] == [image_path, text_path]
    assert [item["success"] for item in metadata["results"]] == [True, False]
    assert metadata["results"][1]["metadata"]["reason"] == "cancelled"
    assert metadata["unprocessed"] == changes[2:]
    assert len(hooks) == len(image_wrapper.calls) == len(text_calls) == 1
    assert isinstance(text_wrapper.inputs, str) and "中文文本" in text_wrapper.inputs
    text_step = AutoTextResourceStep(app_context=env.app_context, file_store=env.file_store)
    assert text_calls[0] == {
        "system_prompt": text_step.prompt_format("system_prompt"),
        "job_tools": ["write"],
        "session_id": str(uuid.uuid5(uuid.NAMESPACE_URL, text_path)),
    }
    assert (env.workspace / "daily/2026-01-01/photo.md").exists()
    assert not (env.workspace / "daily/2026-01-01/waiting.md").exists()
    assert not (env.workspace / "daily/2026-01-01/later.md").exists()


@pytest.mark.parametrize("failure", ["repeat-cancel", "timeout", "error"])
async def test_cancelled_cleanup_is_bounded_and_preserves_the_primary_cancellation(failure, auto_resource_env):
    """Cleanup errors, timeouts, and repeated cancellation cannot hide the original cancellation."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/photo.png"
    source = env.write_binary(source_path, image_bytes())
    wrapper = PausingImageAgentWrapper([source_path])
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    cleanup_started = asyncio.Event()
    cleanup_calls = []

    async def fail_cleanup(_step, day):
        cleanup_calls.append(day)
        cleanup_started.set()
        if failure == "error":
            raise RuntimeError("secondary cleanup error")
        await asyncio.Event().wait()

    context = RuntimeContext(changes=[{"change": "added", "path": str(source)}])
    with (
        patch.object(AutoImageResourceStep, "_refresh_day_index", new=fail_cleanup),
        patch("reme.steps.evolve.base_auto_resource._RESOURCE_CLEANUP_TIMEOUT", 0.05),
    ):
        async with running_step(env.processor(wrapper, routed=True), context) as task:
            await asyncio.wait_for(wrapper.paused.wait(), timeout=5)
            task.cancel()
            if failure == "repeat-cancel":
                await asyncio.wait_for(cleanup_started.wait(), timeout=5)
                task.cancel()
            await assert_cancelled(task)
    metadata = context.response.metadata
    assert not context.response.success
    assert metadata["cancelled"] is True
    assert metadata["modified"] is True
    assert metadata["results"][0]["metadata"]["reason"] == "cancelled"
    assert len(wrapper.calls) == len(cleanup_calls) == 1
    assert len(hooks) == (0 if failure == "repeat-cancel" else 1)
    assert (env.workspace / "daily/2026-01-01/photo.md").exists()
    assert not (env.workspace / "daily/2026-01-01.md").exists()


async def test_cancellation_during_finalize_keeps_the_renamed_note_and_refreshes_its_index(auto_resource_env):
    """Resolve the actual renamed note when cancellation interrupts its index refresh."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("renamed-photo", "New description", "New caption."))
    wrapper.note_metadata = {"source_resource": "[[resource/2026-01-01/photo.png]]"}
    hooks = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    paused = asyncio.Event()
    refresh = AutoImageResourceStep._refresh_day_index
    calls = []

    async def pause_once(step, day):
        calls.append(day)
        if len(calls) == 1:
            paused.set()
            await asyncio.Event().wait()
        return await refresh(step, day)

    context = RuntimeContext(changes=[{"change": "added", "path": str(source)}])
    with patch.object(AutoImageResourceStep, "_refresh_day_index", new=pause_once):
        async with running_step(env.processor(wrapper, routed=True), context) as task:
            await asyncio.wait_for(paused.wait(), timeout=5)
            task.cancel()
            await assert_cancelled(task)
    result = context.response.metadata["results"][0]["metadata"]
    assert result["reason"] == "cancelled"
    assert result["path"] == "daily/2026-01-01/renamed-photo.md"
    assert result["modified"] is True
    assert len(wrapper.calls) == len(hooks) == 1
    assert len(calls) == 2
    assert not (env.workspace / "daily/2026-01-01/photo.md").exists()
    assert (env.workspace / result["path"]).exists()
    assert "renamed-photo.md" in (env.workspace / "daily/2026-01-01.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("phase", ["retarget", "move-result"])
async def test_cancellation_during_move_preserves_its_actual_files_without_a_second_move(phase, auto_resource_env):
    """An interrupted copy/retarget/move keeps its real paths without starting another rename."""
    env = auto_resource_env
    source_path = "resource/2026-01-01/photo.png"
    env.write_binary(source_path, image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("renamed-photo", "New description", "New caption."))
    wrapper.note_metadata = {"source_resource": f"[[{source_path}]]"}
    src_path = "daily/2026-01-01/photo.md"
    dst_path = "daily/2026-01-01/renamed-photo.md"
    paused = asyncio.Event()
    before = {}
    hooks = []
    retarget_calls = []
    recovery_list_calls = []
    env.app_context.metadata = {"qwenpaw_memory_result_hook": lambda **kwargs: hooks.append(kwargs)}
    original_move = MoveStep._move
    original_retarget = WikilinkHandler.retarget_links
    original_list = AutoImageResourceStep._list_daily_notes

    async def pause():
        before.update({path.name: path.read_bytes() for path in (env.workspace / "daily/2026-01-01").glob("*.md")})
        paused.set()
        await asyncio.Event().wait()

    async def retarget(file_store, *, src, dst, **kwargs):
        retarget_calls.append((src, dst))
        if phase == "retarget" and len(retarget_calls) == 1:
            await pause()
        return await original_retarget(file_store, src=src, dst=dst, **kwargs)

    async def move(step, *args, **kwargs):
        result = await original_move(step, *args, **kwargs)
        if phase == "move-result":
            await pause()
        return result

    async def list_daily_notes(step, day):
        if paused.is_set():
            recovery_list_calls.append(day)
            raise RuntimeError("Recovery should use the known move paths, without listing notes")
        return await original_list(step, day)

    context = RuntimeContext(changes=[{"change": "added", "path": source_path}])
    with (
        patch.object(MoveStep, "_move", new=move),
        patch.object(WikilinkHandler, "retarget_links", new=retarget),
        patch.object(AutoImageResourceStep, "_list_daily_notes", new=list_daily_notes),
    ):
        async with running_step(env.processor(wrapper, routed=True), context) as task:
            await asyncio.wait_for(paused.wait(), timeout=5)
            task.cancel("original move cancellation")
            with pytest.raises(asyncio.CancelledError, match="original move cancellation"):
                await asyncio.wait_for(task, timeout=10)
    result = context.response.metadata["results"][0]["metadata"]
    assert task.cancelled()
    assert context.response.success is False
    assert context.response.metadata["cancelled"] is True
    assert context.response.metadata["modified"] is True
    assert result["reason"] == "cancelled" and result["modified"] is True
    assert result["path"] == (src_path if phase == "retarget" else dst_path)
    assert result["interrupted_move"] == {"src_path": src_path, "dst_path": dst_path}
    assert retarget_calls == [(src_path, dst_path)]
    assert not recovery_list_calls
    assert len(hooks) == len(wrapper.calls) == 1
    assert sorted(before) == (["photo.md", "renamed-photo.md"] if phase == "retarget" else ["renamed-photo.md"])
    after = {path.name: path.read_bytes() for path in (env.workspace / "daily/2026-01-01").glob("*.md")}
    assert after == before
    index = (env.workspace / "daily/2026-01-01.md").read_text(encoding="utf-8")
    assert all(f"[[daily/2026-01-01/{name}]]" in index for name in after)


async def test_cancellation_inside_result_hook_never_calls_the_hook_again(auto_resource_env):
    """A cancelled hook remains one notification attempt, with completed writes retained."""
    env = auto_resource_env
    source = env.write_binary("resource/2026-01-01/photo.png", image_bytes())
    wrapper = FakeImageAgentWrapper(caption_fields("photo", "Photo", "A brown coat."))
    hook_started = asyncio.Event()
    hooks = []

    async def hook(**kwargs):
        hooks.append(kwargs)
        hook_started.set()
        await asyncio.Event().wait()

    env.app_context.metadata = {"qwenpaw_memory_result_hook": hook}
    changes = [{"change": "added", "path": str(source)}]
    context = RuntimeContext(changes=changes)
    async with running_step(env.processor(wrapper, routed=True), context) as task:
        await asyncio.wait_for(hook_started.wait(), timeout=5)
        task.cancel()
        await assert_cancelled(task)
    assert len(hooks) == len(wrapper.calls) == 1
    assert context.response.success is False
    assert context.response.metadata["cancelled"] is True
    assert context.response.metadata["unprocessed"] == []
    assert context.response.metadata["modified"] is True
    assert context.response.metadata["results"][0]["success"] is True
    assert context.get("changes") == changes
