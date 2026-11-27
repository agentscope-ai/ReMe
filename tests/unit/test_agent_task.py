"""Named task editing preserves workspace ownership and optimistic file writes."""

# pylint: disable=missing-function-docstring

import os

import pytest

from reme.core.context import RuntimeContext
from reme.schema.agent_view import TaskDocumentRequest
from reme.steps.proactive.api import TaskDocumentStep
from .proactive_helpers import running_system as running_manager


@pytest.mark.asyncio
async def test_task_editor_preserves_text_and_rejects_stale_writes(tmp_path):
    async with running_manager(tmp_path) as (runtime, context, wrapper):
        await runtime.create({"agent_type": "event", "agent_name": "monitor", "task_description": "Original\n"})
        step = TaskDocumentStep(app_context=context)
        opened = (await step(action="read", agent_name="monitor")).answer
        assert opened["content"] == "Original\n"
        assert opened["path"] == "reme_agents/monitor/TASK.md"
        saved = await step(
            action="save",
            agent_name="monitor",
            content="# Task\n\n保留原样。\n",
            expected_mtime=opened["mtime"],
        )
        assert saved.success and "content" not in saved.answer
        target = tmp_path / opened["path"]
        assert target.read_text() == "# Task\n\n保留原样。\n"
        reopened = (await step(action="read", agent_name="monitor")).answer
        target.write_text("External edit\n")
        os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 10))
        conflict = await step(
            action="save",
            agent_name="monitor",
            content="Do not overwrite",
            expected_mtime=reopened["mtime"],
        )
        assert not conflict.success and conflict.metadata["code"] == "file_conflict"
        assert target.read_text() == "External edit\n"
        assert not wrapper.turns


@pytest.mark.asyncio
async def test_task_editor_rejects_remy_deleted_agents_and_outside_paths(tmp_path):
    async with running_manager(tmp_path) as (runtime, context, _):
        step = TaskDocumentStep(app_context=context)
        await runtime.create({"agent_type": "event", "agent_name": "monitor", "task_description": "Original"})
        for agent_name in ("Remy", "missing"):
            with pytest.raises(ValueError):
                await step(action="read", agent_name=agent_name)
        with pytest.raises(PermissionError):
            await step(RuntimeContext(_allowed_paths=["daily"]), action="read", agent_name="monitor")
        with pytest.raises(ValueError):
            await step(action="read", agent_name="monitor", path="daily/other.md")
        target = runtime.registry.task_path(await runtime.definition("monitor"))
        outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
        outside.write_text("Outside")
        target.unlink()
        target.symlink_to(outside)
        try:
            with pytest.raises(ValueError):
                await step(action="read", agent_name="monitor")
            assert outside.read_text() == "Outside"
        finally:
            outside.unlink()
        target.unlink()
        target.write_text("Original")
        await runtime.delete("monitor")
        with pytest.raises(ValueError):
            await step(action="read", agent_name="monitor")


@pytest.mark.parametrize(
    "fields",
    [
        {"action": "save", "agent_name": "task", "content": "text"},
        {"action": "save", "agent_name": "task", "content": " ", "expected_mtime": "revision"},
        {"action": "read", "agent_name": "task", "expected_mtime": None},
        {"action": "read", "agent_name": "../task"},
    ],
)
def test_task_editor_requires_a_revision_and_action_specific_fields(fields):
    with pytest.raises(ValueError):
        TaskDocumentRequest.model_validate(fields)
