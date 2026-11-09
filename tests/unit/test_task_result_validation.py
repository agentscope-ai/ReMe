"""Invalid task snapshots stay untouched and cannot block or pollute valid results."""

# pylint: disable=missing-function-docstring,protected-access

import json

import frontmatter
import pytest

from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.steps.proactive import CronAgentStep
from reme.steps.proactive.api import ProactiveInboxViewStep, TaskResultsStep
from reme.steps.proactive.tools import ReadCronResultsStep

from .pa_helpers import configure, snapshot, stub_model, task


@pytest.fixture(
    name="invalid_result",
    params=[
        b"# Result\n\nBody without frontmatter.\n",
        b"---\ntask_id: other\nstatus: unread\n---\nWrong identity.\n",
        b"---\ntask_id: one\nstatus: unread\n---\n   \n",
        b"---\ntask_id: [\n---\nMalformed YAML.\n",
        b"---\n- not a mapping\n---\nInvalid metadata.\n",
        b"\xff\xfe invalid UTF-8",
    ],
    ids=["missing-metadata", "wrong-task", "empty-body", "bad-yaml", "non-mapping", "bad-encoding"],
)
def invalid_result_file(pa_workspace, request):
    env = pa_workspace
    path = env.runtime.tasks.task_path("one") / "2026-11-08/182230.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(request.param)
    return path, request.param


@pytest.mark.asyncio
async def test_result_list_isolates_invalid_documents_and_preserves_pagination(pa_workspace, invalid_result):
    env = pa_workspace
    bad, original = invalid_result
    valid = [snapshot(env, time="110000"), snapshot(env, identifier="two", time="120000")]
    definition = resolve_app_config(config="proactive", log_config=False)["jobs"]["task_results"]
    job = BaseJob(name="task_results", app_context=env.context, **definition)
    async with job:
        response = await job(action="list", offset=1, limit=1)
    assert response.success
    page = response.answer
    assert page["total"] == 2 and len(page["items"]) == 1
    assert page["task_ids"] == ["one", "two"] and not page["complete"]
    assert len(page["warnings"]) == 1 and bad.name in page["warnings"][0]
    assert page["items"][0]["path"] in {env.runtime.to_workspace_relative(path) for path in valid}
    assert bad.read_bytes() == original


@pytest.mark.asyncio
async def test_inbox_view_reports_the_same_invalid_results(pa_workspace, invalid_result):
    env = pa_workspace
    configure(env, [task()])
    valid = snapshot(env, time="110000")
    bad, original = invalid_result
    response = await ProactiveInboxViewStep(app_context=env.context)(action="list", kind="task_result")
    assert response.success and response.answer["total"] == 1
    assert response.answer["items"][0]["path"] == env.runtime.to_workspace_relative(valid)
    assert not response.answer["complete"] and len(response.answer["warnings"]) == 1
    assert bad.name in response.answer["warnings"][0] and bad.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("has_previous", [False, True])
async def test_cron_summary_uses_latest_valid_history(pa_workspace, monkeypatch, invalid_result, has_previous):
    env = pa_workspace
    valid = snapshot(env, time="110000") if has_previous else None
    bad, original = invalid_result
    captured = stub_model(env, monkeypatch)
    step = CronAgentStep(app_context=env.context)
    response = await step(task=task())
    expected = str(valid) if valid else step.get_prompt("no_previous")
    assert response.success and captured["previous_path"] == expected
    assert bad.name in response.metadata["task_result_warnings"][0]
    assert bad.read_bytes() == original


@pytest.mark.asyncio
async def test_consumption_skips_invalid_results_without_using_the_limit(pa_workspace, invalid_result):
    env = pa_workspace
    configure(env, [task()])
    old = snapshot(env, time="100000")
    selected = [snapshot(env, time=time) for time in ("110000", "120000", "130000")]
    bad, original = invalid_result
    async with env.runtime.proactive_round() as current:
        response = await ReadCronResultsStep(app_context=env.context)(pa_round_id=current.id)
    assert [row["path"] for row in response.answer] == [
        env.runtime.to_workspace_relative(path) for path in reversed(selected)
    ]
    assert all(frontmatter.load(path)["status"] == "read" for path in selected)
    assert frontmatter.load(old)["status"] == "superseded"
    assert response.metadata["complete"] is False and len(response.metadata["warnings"]) == 1
    assert bad.read_bytes() == original
    records = [json.loads(line) for line in env.runtime.proactive_context_path.read_text().splitlines()]
    consumed = next(record["value"] for record in records if record["kind"] == "task_results")
    assert len(consumed) == 3 and all(row["path"] != env.runtime.to_workspace_relative(bad) for row in consumed)


@pytest.mark.asyncio
async def test_direct_read_rejects_invalid_result_without_repairing_it(pa_workspace, invalid_result):
    env = pa_workspace
    bad, original = invalid_result
    expected_error = "utf-8" if original.startswith(b"\xff") else "Invalid task result"
    with pytest.raises(ValueError, match=expected_error):
        await TaskResultsStep(app_context=env.context)(
            action="read",
            path=env.runtime.to_workspace_relative(bad),
            mark_read=True,
        )
    assert bad.read_bytes() == original


@pytest.mark.asyncio
async def test_consumption_revalidates_files_changed_after_selection(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    changed = snapshot(env, time="120000")
    valid = snapshot(env, time="110000")
    paths = env.runtime.tasks.paths

    def select(*args, **kwargs):
        selected = paths(*args, **kwargs)
        changed.write_text("No longer a valid result.\n", encoding="utf-8")
        return selected

    monkeypatch.setattr(env.runtime.tasks, "paths", select)
    warnings = []
    results = await env.runtime.tasks.consume(3, "round", warnings=warnings)
    assert results == [{"task_id": "one", "path": env.runtime.to_workspace_relative(valid)}]
    assert len(warnings) == 1 and changed.name in warnings[0]
    assert changed.read_text() == "No longer a valid result.\n"


@pytest.mark.asyncio
async def test_valid_result_lists_remain_complete(pa_workspace):
    env = pa_workspace
    snapshot(env)
    response = await TaskResultsStep(app_context=env.context)(action="list")
    assert response.answer["complete"] and response.answer["warnings"] == []
    assert response.answer["total"] == 1


@pytest.mark.asyncio
async def test_unreadable_file_does_not_hide_readable_results(pa_workspace, monkeypatch):
    env = pa_workspace
    valid = snapshot(env, time="110000")
    unreadable = snapshot(env, time="120000")
    original = unreadable.read_bytes()
    read = env.runtime.files.read

    def checked_read(path):
        if path == unreadable:
            raise OSError("Cannot read this file")
        return read(path)

    monkeypatch.setattr(env.runtime.files, "read", checked_read)
    response = await TaskResultsStep(app_context=env.context)(action="list")
    assert response.success and response.answer["total"] == 1
    assert response.answer["items"][0]["path"] == env.runtime.to_workspace_relative(valid)
    assert len(response.answer["warnings"]) == 1 and "Cannot read this file" in response.answer["warnings"][0]
    assert unreadable.read_bytes() == original
