"""Dream integration only recovers its own writes, and reports writes left by failed attempts."""

# pylint: disable=protected-access

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from reme.core.context import ApplicationContext, RuntimeContext
from reme.schema import DreamState
from reme.steps.evolve.dream.integrate import DreamIntegrateStep
from reme.steps.file_io.save import SaveStep
from reme.utils.files import write_ledger

from .test_auto_dream import _FileStore, _SequenceAgent, _tool_write, _touch
from .test_dream_failed_unit_report import _OUT_OF_BUCKET, _REPORT_LINE, _UNIT, _receipt, _run

_ATTEMPT_LINE = "- Changed during failed attempts before a successful retry (review): "
_TARGET = "digest/procedure/git-commit-text.md"
_USER_NOTE = "digest/wiki/user-note.md"


async def _user_save(workspace: Path, path: str, content: str) -> None:
    """Save the way the editor does, through the real SaveStep."""
    context = RuntimeContext(path=path, content=content)
    app_context = ApplicationContext(workspace_dir=str(workspace))
    await SaveStep(app_context=app_context, file_store=_FileStore(workspace))(context)
    assert context.response.success, context.response.answer


@pytest.mark.asyncio
async def test_concurrent_user_save_is_not_accepted_as_the_integration_result(tmp_path):
    """ployts's repro: invalid receipt, no agent write, user saves a supported digest note meanwhile."""
    _touch(tmp_path / _USER_NOTE, "v0")
    paused, resume = asyncio.Event(), asyncio.Event()

    async def invalid_receipt_while_user_saves():
        paused.set()
        await resume.wait()
        return {"result": "{}"}

    agent = _SequenceAgent(invalid_receipt_while_user_saves, {"result": "{}"})
    run = asyncio.create_task(_run(tmp_path, agent))
    await asyncio.wait_for(paused.wait(), 5)
    await _user_save(tmp_path, _USER_NOTE, "user edit")
    resume.set()
    context = await asyncio.wait_for(run, 10)

    dream = context.response.metadata["dream"]
    assert agent.calls == 2
    assert context.response.success is False
    assert dream["integrate_results"] == []
    assert dream["nodes_updated"] == []
    assert dream["failed_paths"] == _UNIT["paths"]
    assert dream["checkpoint_paths"] == []
    assert (tmp_path / _USER_NOTE).read_text(encoding="utf-8") == "user edit"
    assert dream["failed_unit_changed_paths"] == [_USER_NOTE]
    assert f"{_REPORT_LINE}{_USER_NOTE}" in dream["summary"]
    assert not write_ledger._LEDGERS


@pytest.mark.asyncio
async def test_a_write_from_an_earlier_attempt_is_not_evidence_for_the_retry(tmp_path):
    """Each attempt has its own ledger, so attempt 1's tool write cannot vouch for a later user save."""
    _touch(tmp_path / _USER_NOTE, "v0")
    stray = "digest/wiki/stray.md"

    async def attempt_one():
        # Two changes, so attempt 1 itself is not recoverable and the unit retries.
        await _tool_write(tmp_path, agent.requests[-1][1], _USER_NOTE, "agent edit")
        _touch(tmp_path / stray, "stray")
        return {"result": "{}"}

    async def attempt_two_while_user_saves():
        await _user_save(tmp_path, _USER_NOTE, "user edit")
        return {"result": "{}"}

    agent = _SequenceAgent(attempt_one, attempt_two_while_user_saves)
    context = await _run(tmp_path, agent)

    dream = context.response.metadata["dream"]
    first_ledger = agent.requests[0][1]["injected_job_kwargs"][write_ledger.WRITE_LEDGER_KWARG]
    second_ledger = agent.requests[1][1]["injected_job_kwargs"][write_ledger.WRITE_LEDGER_KWARG]
    assert first_ledger != second_ledger
    assert context.response.success is False
    assert dream["integrate_results"] == []
    assert dream["failed_paths"] == _UNIT["paths"]
    assert dream["failed_unit_changed_paths"] == [stray, _USER_NOTE]


@pytest.mark.asyncio
async def test_concurrent_save_to_the_file_the_agent_wrote_still_recovers(tmp_path):
    """Evidence is per path: the agent's own tool write recovers even if the user also saved it."""
    _touch(tmp_path / _TARGET, "v0")

    async def agent_writes_then_user_saves():
        await _tool_write(tmp_path, agent.requests[-1][1], _TARGET, "agent edit")
        await _user_save(tmp_path, _TARGET, "agent edit, then a user tweak")
        return {"result": "{}"}

    agent = _SequenceAgent(agent_writes_then_user_saves)
    context = await _run(tmp_path, agent)

    dream = context.response.metadata["dream"]
    assert context.response.success is True
    assert agent.calls == 1
    assert dream["nodes_updated"] == [_TARGET]
    assert dream["failed_attempt_changed_paths"] == []


@pytest.mark.asyncio
async def test_out_of_bucket_edit_from_a_failed_attempt_is_reported_when_the_retry_succeeds(tmp_path):
    """ployts's second repro: attempt 1 edits outside the buckets, attempt 2 succeeds with a valid target."""
    outside = _touch(tmp_path / _OUT_OF_BUCKET, "original workflow\n")
    _touch(tmp_path / _TARGET, "v0")

    async def attempt_one():
        await _tool_write(tmp_path, agent.requests[-1][1], _OUT_OF_BUCKET, "unreviewed model edit\n")
        # A partial edit of the final target (here without tool evidence, so attempt 1 is not
        # recovered) is part of the unit's result, not a stray write.
        _touch(tmp_path / _TARGET, "partial")
        return _receipt(_OUT_OF_BUCKET)

    async def attempt_two():
        await _tool_write(tmp_path, agent.requests[-1][1], _TARGET, "integrated")
        return _receipt(_TARGET)

    agent = _SequenceAgent(attempt_one, attempt_two)
    answers: list[str] = []
    context = await _run(tmp_path, agent, answers=answers)

    dream = context.response.metadata["dream"]
    assert context.response.success is True
    assert agent.calls == 2
    assert dream["failed_units"] == []
    assert dream["nodes_updated"] == [_TARGET]
    assert "unreviewed model edit" in outside.read_text(encoding="utf-8")
    assert dream["failed_attempt_changed_paths"] == [_OUT_OF_BUCKET]
    assert _OUT_OF_BUCKET in dream["modified_paths"]
    assert f"{_ATTEMPT_LINE}{_OUT_OF_BUCKET}" in dream["summary"]
    assert dream["summary"].count(_OUT_OF_BUCKET) == 1
    assert "1 digest file(s) changed during failed attempts before a successful retry" in answers[0]


@pytest.mark.asyncio
async def test_failed_attempt_write_undone_later_in_the_run_is_not_reported(tmp_path):
    """Only paths that still differ from the start of the run are listed."""
    _touch(tmp_path / _TARGET, "v0")
    scratch = tmp_path / "digest" / "workflow" / "scratch.md"

    def attempt_one():
        _touch(scratch, "scratch")
        return {"result": "{}"}

    def attempt_two():
        scratch.unlink()
        return _receipt(_TARGET)

    context = await _run(tmp_path, _SequenceAgent(attempt_one, attempt_two))

    dream = context.response.metadata["dream"]
    assert context.response.success is True
    assert dream["failed_attempt_changed_paths"] == []
    assert _ATTEMPT_LINE not in dream["summary"]


@pytest.mark.asyncio
async def test_path_flagged_by_a_later_failed_unit_is_listed_once(tmp_path):
    """A failed-unit report takes precedence over the failed-attempt report for the same path."""
    outside = _touch(tmp_path / _OUT_OF_BUCKET, "v0")
    _touch(tmp_path / _TARGET, "v0")
    second_unit = {**_UNIT, "name": "second", "paths": ["daily/b.md"]}

    def edit_outside():
        outside.write_text(outside.read_text(encoding="utf-8") + "+", encoding="utf-8")
        return {"result": "{}"}

    agent = _SequenceAgent(edit_outside, _receipt(_TARGET), edit_outside, edit_outside)
    context = await _run(tmp_path, agent, units=[_UNIT, second_unit])

    dream = context.response.metadata["dream"]
    assert len(dream["integrate_results"]) == 1
    assert len(dream["failed_units"]) == 1
    assert dream["failed_unit_changed_paths"] == [_OUT_OF_BUCKET]
    assert dream["failed_attempt_changed_paths"] == []
    assert dream["summary"].count(_OUT_OF_BUCKET) == 1


@pytest.mark.asyncio
async def test_per_unit_warning_uses_a_neutral_label_for_missing_files(tmp_path):
    """The per-unit log line must not say a removed file was left in place."""
    removed = _touch(tmp_path / _OUT_OF_BUCKET, "original")

    def remove():
        removed.unlink()
        return {"result": "{}"}

    state = DreamState(units=[_UNIT], workspace=str(tmp_path))
    step = DreamIntegrateStep(agent_wrapper=_SequenceAgent(remove, {"result": "{}"}))
    step.logger = Mock()
    await step._integrate_one(state, _UNIT, 1, tmp_path, "digest")

    warnings = [str(call.args[0]) for call in step.logger.warning.call_args_list]
    failed = [line for line in warnings if "failed; digest files changed while it ran" in line]
    assert failed and all("were not reverted" in line for line in failed)
    assert not any("left in place" in line for line in warnings)
    assert _OUT_OF_BUCKET in failed[0]


def test_write_ledger_ignores_unknown_and_closed_ids(tmp_path):
    """Tool calls outside an open ledger record nothing and never raise."""
    write_ledger.record_write(None, tmp_path / "a.md")
    write_ledger.record_write("not-a-ledger", tmp_path / "a.md")
    with write_ledger.open_write_ledger() as (ledger_id, written):
        write_ledger.record_write(ledger_id, tmp_path / "a.md")
    write_ledger.record_write(ledger_id, tmp_path / "b.md")
    assert written == {(tmp_path / "a.md").resolve()}
    assert ledger_id not in write_ledger._LEDGERS


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["write", "edit", "frontmatter_update"])
async def test_each_dream_write_tool_records_its_target(tmp_path, tool):
    """All three write tools the integrate agent gets feed the ledger."""
    from reme.steps.file_io.edit import EditStep  # pylint: disable=import-outside-toplevel
    from reme.steps.file_io.frontmatter_update import (  # pylint: disable=import-outside-toplevel
        FrontmatterUpdateStep,
    )
    from reme.steps.file_io.write import WriteStep  # pylint: disable=import-outside-toplevel

    path = "digest/wiki/note.md"
    _touch(tmp_path / path, "---\nname: note\n---\nold body\n")
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    store = _FileStore(tmp_path)
    args = {
        "write": (WriteStep, {"content": "new body"}),
        "edit": (EditStep, {"old": "old body", "new": "new body"}),
        "frontmatter_update": (FrontmatterUpdateStep, {"metadata": {"memory_tags": ["x"]}}),
    }
    step_cls, kwargs = args[tool]
    with write_ledger.open_write_ledger() as (ledger_id, written):
        context = RuntimeContext(path=path, **kwargs, **{write_ledger.WRITE_LEDGER_KWARG: ledger_id})
        await step_cls(app_context=app_context, file_store=store)(context)
    assert context.response.success, context.response.answer
    assert written == {(tmp_path / path).resolve()}
