"""Failed Dream integration units report the digest changes they leave on disk (#759)."""

# pylint: disable=protected-access

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from reme.core.context import ApplicationContext, RuntimeContext
from reme.schema import DreamState
from reme.steps.evolve.dream.finish import DreamFinishStep
from reme.steps.evolve.dream.integrate import DreamIntegrateStep

from .test_auto_dream import _Catalog, _FileStore, _ReplyAgent, _SequenceAgent, _touch

_UNIT = {"name": "git-commit-text", "bucket": "procedure", "summary": "summary", "paths": ["daily/a.md"]}
_OUT_OF_BUCKET = "digest/workflow/git-commit-text-workflow.md"
_REPORT_LINE = "- Changed during failed units (left in place, review): "
_MISSING_LINE = "- Changed during failed units, missing at end of run (review): "


def _receipt(target: str, action: str = "REFINE") -> dict:
    return {"result": json.dumps({"action": action, "target_path": target, "note": "edited"})}


async def _run(workspace: Path, agent, units=None, answers=None):
    """Run integrate + finish the way auto_dream does and return the finished context.

    ``answers`` collects the integration answer, which the finish step replaces with the summary.
    """
    app_context = ApplicationContext(workspace_dir=str(workspace))
    state = DreamState(units=units or [_UNIT], workspace=str(workspace))
    context = RuntimeContext(dream=state.model_dump(), file_catalog=_Catalog(), file_store=_FileStore(workspace))
    with patch("reme.steps.evolve.dream.integrate.llm_available", return_value=True):
        await DreamIntegrateStep(app_context=app_context, agent_wrapper=agent)(context)
    if answers is not None:
        answers.append(context.response.answer)
    await DreamFinishStep(app_context=app_context)(context)
    return context


@pytest.mark.asyncio
async def test_rejected_out_of_bucket_edit_is_left_in_place_and_reported(tmp_path):
    """The #759 scenario: an existing note outside the buckets is edited, then the receipt is rejected."""
    target = _touch(tmp_path / _OUT_OF_BUCKET, "original workflow\n")
    edited = "original workflow\nplus an unreviewed model edit\n"

    def edit_out_of_bucket():
        target.write_text(edited, encoding="utf-8")
        return _receipt(_OUT_OF_BUCKET)

    agent = _SequenceAgent(edit_out_of_bucket, edit_out_of_bucket)
    context = await _run(tmp_path, agent)

    dream = context.response.metadata["dream"]
    assert context.response.success is False
    assert agent.calls == 2
    assert target.read_text(encoding="utf-8") == edited
    assert dream["failed_unit_changed_paths"] == [_OUT_OF_BUCKET]
    assert dream["failed_units"][0]["changed_paths"] == [_OUT_OF_BUCKET]
    assert _OUT_OF_BUCKET in dream["modified_paths"]
    assert context.response.metadata["modified"] is True
    assert "- Failed paths: daily/a.md" in dream["summary"]
    assert f"{_REPORT_LINE}{_OUT_OF_BUCKET}" in dream["summary"]


@pytest.mark.asyncio
async def test_files_saved_by_the_user_during_a_failed_unit_are_never_removed(tmp_path):
    """The job cannot tell who wrote a file, so concurrent saves are kept and only listed."""
    existing = _touch(tmp_path / "digest" / "workflow" / "note.md", "v0")
    created = tmp_path / "digest" / "workflow" / "new" / "user-note.md"

    def user_saves_while_agent_runs():
        existing.write_text("user edit", encoding="utf-8")
        _touch(created, "user created")
        return {"result": "{}"}

    agent = _SequenceAgent(user_saves_while_agent_runs, {"result": "{}"})
    context = await _run(tmp_path, agent)

    dream = context.response.metadata["dream"]
    assert existing.read_text(encoding="utf-8") == "user edit"
    assert created.read_text(encoding="utf-8") == "user created"
    assert dream["failed_unit_changed_paths"] == ["digest/workflow/new/user-note.md", "digest/workflow/note.md"]


@pytest.mark.asyncio
async def test_files_removed_during_a_failed_unit_are_reported_as_missing(tmp_path):
    """A pre-existing file that is gone after a failed unit is listed as missing, not as left in place."""
    removed = _touch(tmp_path / _OUT_OF_BUCKET, "original")
    edited = _touch(tmp_path / "digest" / "workflow" / "kept.md", "v0")

    def remove_and_edit():
        removed.unlink()
        edited.write_text("v1", encoding="utf-8")
        return {"result": "{}"}

    context = await _run(tmp_path, _SequenceAgent(remove_and_edit, {"result": "{}"}))

    dream = context.response.metadata["dream"]
    assert not removed.exists()
    assert dream["failed_unit_changed_paths"] == ["digest/workflow/kept.md"]
    assert dream["failed_unit_missing_paths"] == [_OUT_OF_BUCKET]
    assert dream["failed_units"][0]["changed_paths"] == [_OUT_OF_BUCKET, "digest/workflow/kept.md"]
    assert f"{_REPORT_LINE}digest/workflow/kept.md" in dream["summary"]
    assert f"{_MISSING_LINE}{_OUT_OF_BUCKET}" in dream["summary"]
    assert "Removed during failed units" not in dream["summary"]
    assert dream["summary"].count(_OUT_OF_BUCKET) == 1


@pytest.mark.asyncio
async def test_in_bucket_edits_from_a_failed_unit_are_kept_and_listed(tmp_path):
    """Existing behaviour is kept: durable in-bucket output stays, and is now listed too."""
    kept = "digest/wiki/kept.md"  # a cross-bucket create: never recovered as this unit's result

    def create_in_other_bucket():
        _touch(tmp_path / kept, "durable output from a failed unit")
        return {"result": "{}"}

    context = await _run(tmp_path, _SequenceAgent(create_in_other_bucket, {"result": "{}"}))

    dream = context.response.metadata["dream"]
    assert (tmp_path / kept).read_text(encoding="utf-8") == "durable output from a failed unit"
    assert dream["failed_unit_changed_paths"] == [kept]
    assert dream["modified_paths"] == [kept]


@pytest.mark.asyncio
async def test_symlinked_digest_root_outside_the_workspace_is_not_walked_or_touched(tmp_path):
    """Nothing outside the workspace is listed or modified, even if the digest folder links there."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside_file = _touch(outside / "workflow" / "secret.md", "outside")
    workspace.mkdir()
    try:
        os.symlink(outside, workspace / "digest", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")

    def edit_outside():
        outside_file.write_text("changed through the link", encoding="utf-8")
        return {"result": "{}"}

    context = await _run(workspace, _SequenceAgent(edit_outside, {"result": "{}"}))

    dream = context.response.metadata["dream"]
    assert outside_file.read_text(encoding="utf-8") == "changed through the link"
    assert dream["failed_unit_changed_paths"] == []
    assert len(dream["failed_units"]) == 1


@pytest.mark.asyncio
async def test_summary_lists_each_path_once_and_only_while_it_still_differs(tmp_path):
    """Two failed units editing one file list it once; a file a later unit removes is not listed."""
    shared = _touch(tmp_path / _OUT_OF_BUCKET, "v0")
    temp = tmp_path / "digest" / "workflow" / "scratch.md"
    target = "digest/procedure/ok.md"

    def first_fails():
        shared.write_text("v1", encoding="utf-8")
        _touch(temp, "temporary")
        return {"result": "{}"}

    def second_fails():
        shared.write_text("v2 longer", encoding="utf-8")
        return {"result": "{}"}

    def third_succeeds():
        temp.unlink()
        _touch(tmp_path / target, "new")
        return _receipt(target, action="CREATE")

    units = [
        {"name": "one", "bucket": "procedure", "summary": "s", "paths": ["daily/1.md"]},
        {"name": "two", "bucket": "procedure", "summary": "s", "paths": ["daily/2.md"]},
        {"name": "three", "bucket": "procedure", "summary": "s", "paths": ["daily/3.md"]},
    ]
    agent = _SequenceAgent(first_fails, {"result": "{}"}, second_fails, {"result": "{}"}, third_succeeds)
    context = await _run(tmp_path, agent, units)

    dream = context.response.metadata["dream"]
    assert dream["failed_unit_changed_paths"] == [_OUT_OF_BUCKET]
    assert dream["summary"].count(_OUT_OF_BUCKET) == 1
    assert [unit["changed_paths"] for unit in dream["failed_units"]] == [
        ["digest/workflow/git-commit-text-workflow.md", "digest/workflow/scratch.md"],
        ["digest/workflow/git-commit-text-workflow.md"],
    ]
    assert len(dream["integrate_results"]) == 1


@pytest.mark.asyncio
async def test_file_edited_by_a_failed_unit_and_removed_later_is_reported_as_missing(tmp_path):
    """A failed unit edits a file and a later successful unit removes it.

    The file still existed when the failed unit finished, so the report says only what the
    job can establish: the path changed during a failed unit and is missing at the end.
    """
    shared = _touch(tmp_path / _OUT_OF_BUCKET, "v0")
    target = "digest/procedure/ok.md"

    def first_edits_and_fails():
        shared.write_text("v1 from the failed unit", encoding="utf-8")
        return {"result": "{}"}

    def second_removes_and_succeeds():
        shared.unlink()
        _touch(tmp_path / target, "new")
        return _receipt(target, action="CREATE")

    units = [
        {"name": "one", "bucket": "procedure", "summary": "s", "paths": ["daily/1.md"]},
        {"name": "two", "bucket": "procedure", "summary": "s", "paths": ["daily/2.md"]},
    ]
    agent = _SequenceAgent(first_edits_and_fails, {"result": "{}"}, second_removes_and_succeeds)
    answers = []
    context = await _run(tmp_path, agent, units, answers)

    dream = context.response.metadata["dream"]
    assert not shared.exists()
    assert [unit["changed_paths"] for unit in dream["failed_units"]] == [[_OUT_OF_BUCKET]]
    assert len(dream["integrate_results"]) == 1
    assert dream["failed_unit_changed_paths"] == []
    assert dream["failed_unit_missing_paths"] == [_OUT_OF_BUCKET]
    assert f"{_MISSING_LINE}{_OUT_OF_BUCKET}" in dream["summary"]
    assert "Removed during failed units" not in dream["summary"]
    assert "removed while failed units ran" not in answers[0]
    assert "1 digest file(s) changed during failed units are missing at the end of the run" in answers[0]


@pytest.mark.asyncio
async def test_agent_error_after_retry_reports_partial_edits(tmp_path):
    """The exception path is reported too, not only the invalid-receipt path."""
    first = _touch(tmp_path / "digest" / "workflow" / "a.md", "a0")
    second = _touch(tmp_path / "digest" / "workflow" / "b.md", "b0")

    def partial_write():
        first.write_text("a-partial", encoding="utf-8")
        second.write_text("b-partial", encoding="utf-8")
        raise RuntimeError("agent crashed mid-edit")

    context = await _run(tmp_path, _SequenceAgent(partial_write, partial_write))

    dream = context.response.metadata["dream"]
    assert first.read_text(encoding="utf-8") == "a-partial"
    assert dream["failed_unit_changed_paths"] == ["digest/workflow/a.md", "digest/workflow/b.md"]
    assert dream["errors"]


@pytest.mark.asyncio
async def test_successful_units_are_unaffected(tmp_path):
    """Nothing is listed when every unit succeeds."""
    target = "digest/procedure/ok.md"
    stray = _touch(tmp_path / _OUT_OF_BUCKET, "stray")

    def succeed():
        _touch(tmp_path / target, "new")
        stray.write_text("incidental edit", encoding="utf-8")
        return _receipt(target, action="CREATE")

    context = await _run(tmp_path, _SequenceAgent(succeed))

    dream = context.response.metadata["dream"]
    assert context.response.success is True
    assert dream["failed_unit_changed_paths"] == []
    assert "Changed during failed units" not in dream["summary"]


@pytest.mark.asyncio
async def test_direct_integrate_one_callers_get_the_report(tmp_path):
    """Calling _integrate_one directly still records what a failed unit left behind."""
    target = _touch(tmp_path / _OUT_OF_BUCKET, "original")

    def edit():
        target.write_text("edited", encoding="utf-8")

    state = DreamState(units=[_UNIT])
    step = DreamIntegrateStep(agent_wrapper=_ReplyAgent(_receipt(_OUT_OF_BUCKET), on_reply=edit))

    await step._integrate_one(state, _UNIT, 1, tmp_path, "digest")

    assert target.read_text(encoding="utf-8") == "edited"
    assert state.failed_unit_changed_paths == [_OUT_OF_BUCKET]
    assert state.failed_units[0]["changed_paths"] == [_OUT_OF_BUCKET]
