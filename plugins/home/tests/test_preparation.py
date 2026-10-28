"""One-time preparation through the real application lifecycle and Git objects."""

import asyncio
import copy
from threading import Event
from importlib.metadata import EntryPoint
from importlib.resources import files as resource_files
import json
from pathlib import Path
import subprocess
import shutil

import pytest
import pytest_asyncio
import yaml
from reme_home.preparation_spec import PreparationSpec
from reme_home.code_preparation import tree_fingerprint
from reme_home.preparation_transaction import OUTPUTS, TRANSACTION, recover_preparation
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
        "plugins/beam/README.md": "Plugin installation instructions\n",
        "plugins/beam/config/default.yaml": "enabled: true\n",
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
async def home_application(tmp_path, monkeypatch, request):
    """Enable Home by entry point, as an installed plugin would be enabled."""
    judge_name = getattr(request, "param", "beam-judge")
    for name in ("beam-judge", "lme-judge"):
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / name / "src"))
    entries = {
        name: EntryPoint(name=name, value=value, group="reme.plugins")
        for name, value in [("home", "reme_home"), ("beam-judge", "judge_beam"), ("lme-judge", "judge_lme")]
    }
    monkeypatch.setattr("reme.plugin.find_entry_points", lambda group, name: [entries[name]])
    monkeypatch.setenv("LLM_API_KEY", "test")
    application = Application(
        workspace_dir=str(tmp_path / "workspace"),
        plugins=["home", judge_name],
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
    assert (root / "home/code/repo/plugins/beam/src/reme_beam/__init__.py").read_text() == "value = 1\n"
    case = json.loads((root / "home/datasets/training-set/cases/000000.json").read_text())
    assert case["sessions"][0]["messages"][0]["id"] == "1"
    assert case["queries"][0]["metadata"]["evidence_session_ids"] == ["beam_100K_2_batch1"]
    assert (root / "home/datasets/debug-cases/derivation.json").is_file()
    docs = root / "home/docs"
    resources = resource_files("reme_home").joinpath("runtime_docs")
    assert sorted(path.name for path in docs.iterdir()) == ["architecture-guide.md", "validation-artifact-guide.md"]
    for path in docs.iterdir():
        assert path.read_bytes() == resources.joinpath(path.name).read_bytes()
    assert command(root / "home/code/repo", "branch", "--show-current") == "init"
    assert not command(root / "home/code/repo", "status", "--porcelain")
    before = snapshot(root)
    again = await app.run_job("home_prepare", spec=spec)
    assert not again.success
    assert "already prepared" in str(again.answer)
    assert snapshot(root) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate_path",
    ["plugins/beam/src", "plugins/beam", "plugins/beam/src/reme_beam/__init__.py"],
)
async def test_complete_plugins_live_only_in_candidate(app, inputs, tmp_path, candidate_path):
    """An editable subset still yields an installable plugin and a Git baseline."""
    spec, _ = inputs
    spec["candidate"]["paths"] = [candidate_path, "reme/candidate.py"]
    source = Path(spec["runtime"]["source_repo"])
    (source / "plugins/beam/README.md").write_text("uncommitted documentation")
    response = await app.run_job("home_prepare", spec=spec)
    assert response.success, response.answer
    root = Path(app.config.workspace_dir)
    runtime, candidate = root / "home/code/runtime", root / "home/code/repo"
    assert not (runtime / "plugins").exists()
    for relative in (
        "pyproject.toml",
        "README.md",
        "config/default.yaml",
        "src/reme_beam/plugin.yaml",
        "src/reme_beam/__init__.py",
    ):
        name = f"plugins/beam/{relative}"
        committed = command(source, "show", f"{spec['runtime']['source_ref']}:{name}")
        assert (candidate / name).read_text().strip() == committed
        assert command(candidate, "show", f"init:{name}") == committed
    assert (candidate / "reme/candidate.py").read_bytes() == (runtime / "reme/candidate.py").read_bytes()
    code = response.answer["code"]
    assert code["candidate_paths"] == spec["candidate"]["paths"]
    assert code["runtime_fingerprint"] == tree_fingerprint(runtime)
    assert code["candidate_fingerprint"] == tree_fingerprint(candidate)
    # Future evaluators combine these trees; plugin installation uses repo bytes.
    evaluation = tmp_path / "evaluation"
    shutil.copytree(runtime, evaluation)
    shutil.copytree(candidate, evaluation, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
    for project in code["install_projects"]:
        assert (evaluation / project / "pyproject.toml").is_file()
    assert not command(candidate, "status", "--porcelain")


@pytest.mark.asyncio
@pytest.mark.parametrize("spec_path", ["prepare.yaml", "home/prepare.yaml"])
@pytest.mark.parametrize("app", ["lme-judge"], indirect=True)
async def test_yaml_lme_core_candidate(app, inputs, spec_path):
    """Workspace YAML and core candidates use the same job as Python dictionaries."""
    spec, lme = inputs
    spec["benchmark"] = {"name": "longmemeval", "source": str(lme), "all_cases": True}
    spec["runtime"]["plugins"] = []
    spec["candidate"]["paths"] = ["reme/candidate.py"]
    spec["debug"] = {"enabled": False}
    root = Path(app.config.workspace_dir)
    (root / spec_path).write_text(yaml.safe_dump(spec))
    response = await app.run_job("home_prepare", spec_path=spec_path)
    assert response.success, response.answer
    domain = yaml.safe_load((root / "home/domain_spec.yaml").read_text())
    assert domain["benchmark"]["case_ids"] == ["q1"]
    assert not domain["benchmark"]["all_cases"]
    assert (root / "home/code/repo/reme/candidate.py").is_file()
    case = json.loads((root / "home/datasets/training-set/cases/000000.json").read_text())
    assert case["queries"][0]["metadata"]["question_date"] == "2024/01/02 (Tue) 00:00"
    assert case["sessions"][0]["messages"][0]["created_at"] == "2024-01-01T00:00:00"


@pytest.mark.asyncio
async def test_relative_inputs_still_resolve_against_outer_workspace(app, inputs):
    """Nesting outputs must not change existing source-path configuration."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    shutil.copytree(spec["benchmark"]["source"], root / "inputs/beam")
    shutil.copytree(spec["runtime"]["source_repo"], root / "inputs/source")
    spec["benchmark"]["source"] = "inputs/beam"
    spec["runtime"]["source_repo"] = "inputs/source"
    response = await app.run_job("home_prepare", spec=spec)
    assert response.success, response.answer
    domain = yaml.safe_load((root / "home/domain_spec.yaml").read_text())
    assert domain["benchmark"]["source"] == str(root / "inputs/beam")
    assert domain["runtime"]["source_repo"] == str(root / "inputs/source")
    assert not (root / "code").exists()
    assert not (root / "datasets").exists()
    assert not (root / "metadata/home").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "convert",
        "code",
        "docs",
        "absent-outputs",
        *[f"publish-{i}" for i in range(len(OUTPUTS) + 1)],
        *[f"after-{i}" for i in range(len(OUTPUTS) + 1)],
    ],
)
async def test_failures_restore_original_workspace_and_allow_retry(app, inputs, monkeypatch, failure):
    """Failures during conversion, code export, and every publication phase roll back."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    (root / "daily/keep.md").write_text("user memory")
    if failure == "absent-outputs":
        for relative in ("code/runtime", "code/repo", "datasets/training-set", "datasets/debug-cases", "docs"):
            shutil.rmtree(root / "home" / relative)
        failure = "after-5"
    before = snapshot(root)
    with monkeypatch.context() as patch:
        if failure == "convert":
            patch.setattr(
                "reme_home.preparation.prepare_dataset",
                lambda *a, **kw: (_ for _ in ()).throw(OSError("test")),
            )
        elif failure == "code":
            patch.setattr("reme_home.preparation.prepare_code", lambda *a, **kw: (_ for _ in ()).throw(OSError("test")))
        elif failure == "docs":
            original_read = Path.read_bytes

            def read_bytes(path):
                if path.name == "validation-artifact-guide.md":
                    raise FileNotFoundError("missing packaged document")
                return original_read(path)

            patch.setattr(Path, "read_bytes", read_bytes)
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
async def test_preparation_preserves_existing_docs(app, inputs):
    """Existing user documentation is a conflict, never overwritten or removed."""
    root = Path(app.config.workspace_dir)
    (root / "home/docs/architecture-guide.md").write_text("user-owned documentation")
    before = snapshot(root)
    response = await app.run_job("home_prepare", spec=inputs[0])
    assert not response.success
    assert "not empty" in str(response.answer)
    assert snapshot(root) == before


@pytest.mark.asyncio
async def test_legacy_preparation_journal_recovery_preserves_docs(app):
    """Version 1 recovery uses the original output indices and leaves docs alone."""
    root = Path(app.config.workspace_dir) / "home"
    state = root / "metadata"
    (root / "docs/keep.md").write_text("existing documentation")
    before = snapshot(root)
    transaction = state / TRANSACTION
    transaction.mkdir()
    (transaction / "journal.json").write_text(
        json.dumps({"schema_version": 1, "existed": [True, True, True, True, False, False]}),
    )
    (root / "code/runtime").rename(transaction / "backup-0")
    (root / "code/runtime").mkdir()
    (root / "code/runtime/partial.py").write_text("partial runtime")
    (root / "code/repo").rename(transaction / "backup-1")
    (transaction / "output-1").mkdir()
    recover_preparation(root, state)
    assert snapshot(root) == before


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
    assert (root / "home/metadata" / TRANSACTION).exists()
    workspace = app.context.components["home.workspace"]["default"]
    await workspace.close()
    await workspace.start()
    assert not (root / "home/metadata" / TRANSACTION).exists()
    assert not (await app.run_job("home_prepare", spec=spec)).success


@pytest.mark.asyncio
async def test_conflicts_and_lock_preserve_user_content(app, inputs):
    """Competing jobs and user-owned output directories cannot be overwritten."""
    spec, _ = inputs
    root = Path(app.config.workspace_dir)
    before = snapshot(root)
    with _initialization_lock(root / "home/metadata/workspace.lock"):
        response = await app.run_job("home_prepare", spec=spec)
    assert not response.success
    assert snapshot(root) == before
    (root / "home/code/repo/keep.py").write_text("keep")
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
    state = tmp_path / "home/metadata"
    transaction = state / TRANSACTION
    transaction.mkdir(parents=True)
    (transaction / "journal.json").write_text('{"existed": [], "paths": ["../../outside"]}')
    with pytest.raises(ValueError):
        recover_preparation(tmp_path / "home", state)


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
@pytest.mark.parametrize("app", ["lme-judge"], indirect=True)
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
@pytest.mark.parametrize("app", ["lme-judge"], indirect=True)
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
    case = json.loads((root / "home/datasets/training-set/cases/000000.json").read_text())
    assert case["queries"][0]["metadata"]["evidence_session_ids"] == ["s1"]
    debug = json.loads((root / "home/datasets/debug-cases/cases/000000.json").read_text())
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
