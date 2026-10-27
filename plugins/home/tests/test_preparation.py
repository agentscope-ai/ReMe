"""One-time preparation through the real application lifecycle and Git objects."""

import asyncio
import copy
from threading import Event
from importlib.metadata import EntryPoint
import json
from pathlib import Path
import subprocess
import shutil

import pytest
import pytest_asyncio
import yaml
from reme_home.preparation_spec import PreparationSpec
from reme_home.preparation_transaction import TRANSACTION, recover_preparation
from reme_home.workspace import _initialization_lock

from reme.application import Application


def command(root, *args):
    """Git fixtures never depend on a developer's identity or hooks."""
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false", "-C", str(root), *args],
        text=True,
    ).strip()


@pytest.fixture(name="inputs")
def preparation_inputs(tmp_path):
    """A committed tiny ReMe project plus both benchmark formats."""
    repository = tmp_path / "source"
    repository.mkdir()
    files = {
        "pyproject.toml": '[project]\nname="fixture-reme"\nversion="0.0.1"\n',
        "reme/__init__.py": '"""Fixture."""\n',
        "reme/config/benchmark.yaml": "jobs: {}\n",
        "reme/candidate.py": "baseline = 1\n",
        "plugins/beam/pyproject.toml": '[project.entry-points."reme.plugins"]\nbeam="reme_beam"\n',
        "plugins/beam/src/reme_beam/plugin.yaml": "backends: {}\n",
        "plugins/beam/src/reme_beam/__init__.py": "value = 1\n",
    }
    for name, value in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    command(repository, "init", "--initial-branch=main", "--template=")
    command(repository, "add", ".")
    command(repository, "-c", "user.name=Fixture", "-c", "user.email=fixture@localhost", "commit", "-m", "fixture")
    revision = command(repository, "rev-parse", "HEAD")
    beam = tmp_path / "beam"
    for case_id in ("1", "2"):
        directory = beam / "chats/100K" / case_id
        (directory / "probing_questions").mkdir(parents=True)
        (directory / "chat.json").write_text(
            json.dumps(
                [
                    {
                        "batch_number": 1,
                        "time_anchor": "January-1-2024",
                        "turns": [[{"id": 1, "role": "user", "content": "hello"}]],
                    },
                ],
            ),
        )
        (directory / "probing_questions/probing_questions.json").write_text(
            json.dumps(
                {
                    "fact": [{"question": "What?", "ideal_answer": "hello", "source_chat_ids": [1]}],
                },
            ),
        )
    lme = tmp_path / "lme.json"
    lme.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "What?",
                    "answer": "hello",
                    "question_type": "single-session-user",
                    "question_date": "2024/01/02 (Tue) 00:00",
                    "haystack_dates": ["2024/01/01 (Mon) 00:00"],
                    "haystack_session_ids": ["s1"],
                    "haystack_sessions": [[{"role": "user", "content": "hello"}]],
                },
            ],
        ),
    )
    spec = {
        "benchmark": {"name": "beam", "source": str(beam), "variant": "100K", "case_ids": ["2", "1"]},
        "runtime": {"source_repo": str(repository), "source_ref": revision, "plugins": ["beam"]},
        "candidate": {"paths": ["plugins/beam/src"]},
    }
    return spec, lme


@pytest_asyncio.fixture(name="app")
async def home_application(tmp_path, monkeypatch):
    """Enable Home by entry point, as an installed plugin would be enabled."""
    entry = EntryPoint(name="home", value="reme_home", group="reme.plugins")
    monkeypatch.setattr("reme.plugin.find_entry_points", lambda *args: [entry])
    application = Application(
        workspace_dir=str(tmp_path / "workspace"),
        plugins=["home"],
        service={"backend": "cli"},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )
    await application.start()
    try:
        yield application
    finally:
        await application.close()


def snapshot(root):
    """Compare durable paths, bytes and modes, ignoring atime/mtime bookkeeping."""
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mode, path.read_bytes() if path.is_file() else None)
        for path in root.rglob("*")
    }


@pytest.mark.asyncio
async def test_python_job_prepares_once(app, inputs):
    """Dict input prepares selected cases and only committed source bytes."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    (Path(spec["runtime"]["source_repo"]) / "plugins/beam/src/reme_beam/__init__.py").write_text("dirty = True")
    response = await app.run_job("home_prepare", spec=spec)
    assert response.success, response.answer
    assert response.answer["dataset"]["case_ids"] == ["2", "1"]
    assert response.answer["dataset"]["query_count"] == 2
    assert (root / "code/repo/plugins/beam/src/reme_beam/__init__.py").read_text() == "value = 1\n"
    case = json.loads((root / "datasets/training-set/cases/000000.json").read_text())
    assert case["sessions"][0]["messages"][0]["id"] == "1"
    assert case["queries"][0]["metadata"]["evidence_session_ids"] == ["beam_100K_2_batch1"]
    assert (root / "datasets/debug-cases/derivation.json").is_file()
    assert command(root / "code/repo", "branch", "--show-current") == "init"
    assert not command(root / "code/repo", "status", "--porcelain")
    before = snapshot(root)
    again = await app.run_job("home_prepare", spec=spec)
    assert not again.success
    assert "already prepared" in str(again.answer)
    assert snapshot(root) == before


@pytest.mark.asyncio
async def test_yaml_lme_core_candidate(app, inputs):
    """Workspace YAML and core candidates use the same job as Python dictionaries."""
    spec, lme = inputs
    spec["benchmark"] = {"name": "longmemeval", "source": str(lme), "all_cases": True}
    spec["runtime"]["plugins"] = []
    spec["candidate"]["paths"] = ["reme/candidate.py"]
    spec["debug"] = {"enabled": False}
    root = Path(app.config.workspace_dir)
    (root / "prepare.yaml").write_text(yaml.safe_dump(spec))
    response = await app.run_job("home_prepare", spec_path="prepare.yaml")
    assert response.success, response.answer
    domain = yaml.safe_load((root / "domain_spec.yaml").read_text())
    assert domain["benchmark"]["case_ids"] == ["q1"]
    assert not domain["benchmark"]["all_cases"]
    assert (root / "code/repo/reme/candidate.py").is_file()
    case = json.loads((root / "datasets/training-set/cases/000000.json").read_text())
    assert case["queries"][0]["metadata"]["question_date"] == "2024/01/02 (Tue) 00:00"
    assert case["sessions"][0]["messages"][0]["created_at"] == "2024-01-01T00:00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["convert", "code", "absent-outputs", *[f"publish-{i}" for i in range(6)], *[f"after-{i}" for i in range(6)]],
)
async def test_failures_restore_original_workspace_and_allow_retry(app, inputs, monkeypatch, failure):
    """Failures during conversion, code export, and every publication phase roll back."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    (root / "daily/keep.md").write_text("user memory")
    if failure == "absent-outputs":
        for relative in ("code/runtime", "code/repo", "datasets/training-set", "datasets/debug-cases"):
            shutil.rmtree(root / relative)
        failure = "after-3"
    before = snapshot(root)
    with monkeypatch.context() as patch:
        if failure == "convert":
            patch.setattr(
                "reme_home.preparation.prepare_dataset",
                lambda *a, **kw: (_ for _ in ()).throw(OSError("test")),
            )
        elif failure == "code":
            patch.setattr("reme_home.preparation.prepare_code", lambda *a, **kw: (_ for _ in ()).throw(OSError("test")))
        else:
            original = __import__("os").replace

            def replace(source, destination):
                if Path(source).name == failure.replace("publish-", "output-"):
                    raise OSError("test publication failure")
                result = original(source, destination)
                if failure.startswith("after-") and Path(source).name == failure.replace("after-", "output-"):
                    raise OSError("test failure after rename")
                return result

            patch.setattr("reme_home.preparation_transaction.os.replace", replace)
        response = await app.run_job("home_prepare", spec=spec)
    assert not response.success
    assert snapshot(root) == before
    assert (await app.run_job("home_prepare", spec=spec)).success


@pytest.mark.asyncio
async def test_interrupted_publication_recovers_at_startup(app, inputs, monkeypatch):
    """A durable journal survives process interruption and is recovered on startup."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    preparation = app.context.components["home.preparation"]["default"]
    with monkeypatch.context() as patch:
        original = __import__("os").replace

        def replace(source, destination):
            if Path(source).name == "output-3":
                raise RuntimeError("simulated crash")
            return original(source, destination)

        patch.setattr("reme_home.preparation_transaction.os.replace", replace)
        patch.setattr("reme_home.preparation.rollback", lambda *args: None)
        with pytest.raises(RuntimeError, match="simulated crash"):
            await preparation.prepare(spec=spec)
    workspace = app.context.components["home.workspace"]["default"]
    await workspace.close()
    await workspace.start()
    assert snapshot(root) == before
    assert (await app.run_job("home_prepare", spec=spec)).success


@pytest.mark.asyncio
async def test_success_survives_cleanup_failure_and_restart(app, inputs, monkeypatch):
    """Committed completion is retained when cleanup was interrupted."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    original = __import__("shutil").rmtree
    with monkeypatch.context() as patch:

        def remove(path, *args, **kwargs):
            if Path(path).name == TRANSACTION:
                raise OSError("cleanup")
            return original(path, *args, **kwargs)

        patch.setattr("reme_home.preparation.shutil.rmtree", remove)
        response = await app.run_job("home_prepare", spec=spec)
    assert response.success
    assert (root / "metadata/home" / TRANSACTION).exists()
    workspace = app.context.components["home.workspace"]["default"]
    await workspace.close()
    await workspace.start()
    assert not (root / "metadata/home" / TRANSACTION).exists()
    assert not (await app.run_job("home_prepare", spec=spec)).success


@pytest.mark.asyncio
async def test_conflicts_and_lock_preserve_user_content(app, inputs):
    """Competing jobs and user-owned output directories cannot be overwritten."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    with _initialization_lock(root / "metadata/home/workspace.lock"):
        response = await app.run_job("home_prepare", spec=spec)
    assert not response.success
    assert snapshot(root) == before
    (root / "code/repo/keep.py").write_text("keep")
    before = snapshot(root)
    assert not (await app.run_job("home_prepare", spec=spec)).success
    assert snapshot(root) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"spec": {}, "spec_path": "prepare.yaml"},
        {"spec_path": "../outside.yaml"},
        {"spec_path": "~/prepare.yaml"},
        {"spec": {}, "_allowed_paths": []},
    ],
)
async def test_bad_job_inputs_do_not_mutate_workspace(app, kwargs):
    """Path escape and restricted invocation fail closed."""
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    assert not (await app.run_job("home_prepare", **kwargs)).success
    assert snapshot(root) == before


@pytest.mark.parametrize(
    "change",
    [
        {"case_ids": []},
        {"case_ids": ["1", "1"]},
        {"case_ids": [1]},
        {"case_ids": None},
        {"all_cases": True},
        {"variant": "bad"},
        {"evidence_source": "annotations.jsonl"},
    ],
)
def test_strict_case_selection(inputs, change):
    """Do not silently select everything or accept misspelled dataset options."""
    spec, _ = inputs
    spec = copy.deepcopy(spec)
    spec["benchmark"].update(change)
    with pytest.raises(ValueError):
        PreparationSpec.model_validate(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["unknown-case", "unknown-plugin", "bad-ref", "bad-candidate", "symlink"])
async def test_invalid_sources_roll_back(app, inputs, mutation):
    """Input and revision failures leave no partially prepared dataset."""
    spec, _ = inputs
    if mutation == "unknown-case":
        spec["benchmark"]["case_ids"] = ["missing"]
    elif mutation == "unknown-plugin":
        spec["runtime"]["plugins"] = ["missing"]
    elif mutation == "bad-ref":
        spec["runtime"]["source_ref"] = "--bad-ref"
    elif mutation == "bad-candidate":
        spec["candidate"]["paths"] = ["../outside"]
    else:
        repository = Path(spec["runtime"]["source_repo"])
        (repository / "reme/link").symlink_to("__init__.py")
        command(repository, "add", ".")
        command(repository, "-c", "user.name=Fixture", "-c", "user.email=fixture@localhost", "commit", "-m", "link")
        spec["runtime"]["source_ref"] = "HEAD"
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    assert not (await app.run_job("home_prepare", spec=spec)).success
    assert snapshot(root) == before


def test_bad_recovery_journal_does_not_remove_outside_paths(tmp_path):
    """Recovery accepts no path names from disk."""
    state = tmp_path / "metadata/home"
    transaction = state / TRANSACTION
    transaction.mkdir(parents=True)
    (transaction / "journal.json").write_text('{"existed": [], "paths": ["../../outside"]}')
    with pytest.raises(ValueError):
        recover_preparation(tmp_path, state)


@pytest.mark.asyncio
async def test_cancellation_waits_for_rollback(app, inputs, monkeypatch):
    """The event loop stays available, and cancellation waits for the worker to restore state."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    component = app.context.components["home.preparation"]["default"]
    stage = component._stage  # pylint: disable=protected-access
    entered = Event()
    proceed = Event()

    def blocked(*args):
        entered.set()
        if not proceed.wait(10):
            raise RuntimeError("test worker timed out")
        return stage(*args)

    monkeypatch.setattr(component, "_stage", blocked)
    task = asyncio.create_task(app.run_job("home_prepare", spec=spec))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
    assert snapshot(root) == before
    monkeypatch.setattr(component, "_stage", stage)
    assert (await app.run_job("home_prepare", spec=spec)).success


@pytest.mark.asyncio
async def test_duplicate_lme_ids_fail_without_mutation(app, inputs):
    """Upstream duplicate IDs cannot silently replace selected cases."""
    spec, lme = inputs
    items = json.loads(lme.read_text())
    lme.write_text(json.dumps(items + items))
    spec["benchmark"] = {"name": "longmemeval", "source": str(lme), "all_cases": True}
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    response = await app.run_job("home_prepare", spec=spec)
    assert not response.success
    assert "Duplicate LongMemEval" in str(response.answer)
    assert snapshot(root) == before


@pytest.mark.asyncio
async def test_lme_evidence_and_time_cutoff_debug_cases(app, inputs):
    """Evidence survives conversion and debug data obeys question-time cutoffs."""
    spec, lme = inputs
    items = json.loads(lme.read_text())
    item = items[0]
    item["haystack_dates"].append("2024/01/03 (Wed) 00:00")
    item["haystack_session_ids"].append("future")
    item["haystack_sessions"].append([{"role": "user", "content": "future information"}])
    lme.write_text(json.dumps(items))
    evidence = lme.with_suffix(".jsonl")
    evidence.write_text(json.dumps({"case_id": "q1", "evidence_sessions": ["s1"], "is_certain": True}) + "\n")
    spec["benchmark"] = {
        "name": "longmemeval",
        "source": str(lme),
        "case_ids": ["q1"],
        "evidence_source": str(evidence),
    }
    response = await app.run_job("home_prepare", spec=spec)
    assert response.success, response.answer
    root = Path(app.config.workspace_dir)
    case = json.loads((root / "datasets/training-set/cases/000000.json").read_text())
    assert case["queries"][0]["metadata"]["evidence_session_ids"] == ["s1"]
    debug = json.loads((root / "datasets/debug-cases/cases/000000.json").read_text())
    assert [session["session_id"] for session in debug["sessions"]] == ["s1"]


@pytest.mark.asyncio
async def test_concurrent_preparation_rejects_second_writer(app, inputs, monkeypatch):
    """Two active Jobs share the process lock rather than racing to publish."""
    spec, _ = inputs
    component = app.context.components["home.preparation"]["default"]
    stage = component._stage  # pylint: disable=protected-access
    entered = Event()
    proceed = Event()

    def blocked(*args):
        entered.set()
        if not proceed.wait(10):
            raise RuntimeError("test worker timed out")
        return stage(*args)

    monkeypatch.setattr(component, "_stage", blocked)
    first = asyncio.create_task(app.run_job("home_prepare", spec=spec))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        second = await app.run_job("home_prepare", spec=spec)
        assert not second.success
        assert "already in progress" in str(second.answer)
        proceed.set()
        assert (await first).success
    finally:
        proceed.set()
        await asyncio.gather(first, return_exceptions=True)
