"""Candidate version Jobs exercised against real isolated Git repositories."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name,unused-argument

import asyncio
from pathlib import Path
import subprocess
from threading import Event

import pytest
import pytest_asyncio
from reme_home.code import snapshot
from reme_home.code.models import ARGUMENTS
from reme_home.workspace import _initialization_lock

from test_preparation import (  # noqa: F401  # pylint: disable=unused-import
    home_application,
    preparation_inputs,
    command,
)


from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper


@pytest_asyncio.fixture
async def prepared(app, inputs):
    spec, _ = inputs
    response = await app.run_job("home_prepare", spec=spec)
    assert response.success, response.answer
    repo = Path(app.config.workspace_dir) / "home/code/repo"
    return app, repo, spec


async def run(app, job, **kwargs):
    response = await app.run_job(job, **kwargs)
    assert response.success, response.answer
    return response.answer


async def create(app, name="candidate"):
    return await run(app, "home_create_candidate_branch", branch_name=name)


async def commit(app, repo, text="value = 2\n"):
    (repo / "plugins/beam/src/reme_beam/__init__.py").write_text(text)
    return await run(app, "home_commit_candidate", message="Improve candidate")


@pytest.mark.asyncio
async def test_branch_commit_diff_and_immutable_read(prepared):
    app, repo, _ = prepared
    initial = await run(app, "home_list_candidate_branches")
    assert initial["current_branch"] == "init"
    assert initial["candidate_branches"] == []
    created = await create(app)
    assert created["candidate_branch_count"] == 1
    result = await commit(app, repo)
    assert result["commit_sha"] != initial["commit_sha"]
    assert (await run(app, "home_code_status"))["clean"]
    diff = await run(app, "home_code_diff", base_revision="init", revision=result["commit_sha"])
    assert "+value = 2" in diff["content"]
    (repo / "plugins/beam/src/reme_beam/__init__.py").write_text("dirty = 3\n")
    status = await run(app, "home_code_status")
    assert status["changes"] == [{"status": " M", "path": "plugins/beam/src/reme_beam/__init__.py"}]
    assert not status["clean"]
    assert "+dirty = 3" in (await run(app, "home_code_diff"))["content"]
    read = await run(
        app,
        "home_read_code_file",
        path="plugins/beam/src/reme_beam/__init__.py",
        revision=initial["commit_sha"],
    )
    assert read["content"] == "value = 1\n"
    with pytest.raises(ValueError, match="clean"):
        await app.context.components["home.code"]["default"].run("switch_branch", branch_name="init")


@pytest.mark.asyncio
async def test_host_pinned_job_tool_and_schema(prepared):
    app, repo, _ = prepared
    initial = command(repo, "rev-parse", "HEAD")
    await create(app)
    await commit(app, repo)
    job = app.context.jobs["home_read_code_file"]
    tool = AsAgentWrapper._make_tool(job, injected_job_kwargs={"revision": initial})
    assert "revision" not in tool.input_schema["properties"]
    # Test the actual wrapper's merge contract through its executable FunctionTool.
    response = await tool(path="plugins/beam/src/reme_beam/__init__.py")
    assert "value = 1" in response.content[0].text
    assert "value = 2" not in response.content[0].text
    with pytest.raises(ValueError, match="injected"):
        await tool(path="plugins/beam/src/reme_beam/__init__.py", revision="HEAD")
    assert all(model.model_json_schema()["additionalProperties"] is False for model in ARGUMENTS.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["init", "a/b", "../escape", "-bad", "a\\b", "@{-1}"])
async def test_invalid_candidate_branch(prepared, name):
    app, _, _ = prepared
    response = await app.run_job("home_create_candidate_branch", branch_name=name)
    assert not response.success
    assert (await run(app, "home_code_status"))["current_branch"] == "init"


@pytest.mark.asyncio
async def test_protected_init_and_scope(prepared):
    app, repo, _ = prepared
    response = await app.run_job("home_commit_candidate", message="not allowed")
    assert not response.success
    await create(app)
    baseline = command(repo, "rev-parse", "HEAD")
    readme = repo / "plugins/beam/README.md"
    readme.write_text("outside declared editable roots")
    response = await app.run_job("home_commit_candidate", message="outside scope")
    assert not response.success
    assert command(repo, "rev-parse", "HEAD") == baseline
    assert not command(repo, "diff", "--cached", "--name-only")
    readme.write_text("Plugin installation instructions\n")
    target = repo / "plugins/beam/src/link"
    target.symlink_to("/etc/passwd")
    response = await app.run_job("home_commit_candidate", message="link")
    assert not response.success
    target.unlink()
    result = await commit(app, repo)
    command(repo, "update-ref", "refs/heads/init", result["commit_sha"])
    response = await app.run_job("home_code_status")
    assert not response.success


@pytest.mark.asyncio
async def test_merge_then_explicit_commit(prepared):
    app, repo, _ = prepared
    await create(app, "left")
    left = await commit(app, repo)
    await run(app, "home_switch_candidate_branch", branch_name="init")
    await create(app, "right")
    (repo / "plugins/beam/src/new.py").write_text("another mechanism\n")
    right = await run(app, "home_commit_candidate", message="right")
    merged = await run(app, "home_merge_candidate_branch", branch_name="left")
    assert merged["status"] == "merged_not_committed"
    assert merged["commit_sha"] == right["commit_sha"]
    assert (await run(app, "home_code_status"))["merge_pending"]
    response = await app.run_job("home_switch_candidate_branch", branch_name="init")
    assert not response.success
    merged = await run(app, "home_commit_candidate", message="Combine compatible improvements")
    assert command(repo, "rev-list", "--parents", "-n", "1", "HEAD").split() == [
        merged["commit_sha"],
        right["commit_sha"],
        left["commit_sha"],
    ]


@pytest.mark.asyncio
async def test_conflicting_merge_aborts_cleanly(prepared):
    app, repo, _ = prepared
    await create(app, "left")
    await commit(app, repo, "left = 2\n")
    await run(app, "home_switch_candidate_branch", branch_name="init")
    await create(app, "right")
    right = await commit(app, repo, "right = 2\n")
    response = await app.run_job("home_merge_candidate_branch", branch_name="left")
    assert not response.success
    assert not (repo / ".git/MERGE_HEAD").exists()
    state = await run(app, "home_code_status")
    assert state["clean"] and state["commit_sha"] == right["commit_sha"]


@pytest.mark.asyncio
async def test_raw_snapshot_regex_paths_and_truncation(prepared):
    app, repo, _ = prepared
    await create(app)
    folder = repo / "plugins/beam/src"
    (folder / "space name.py").write_text("prefix\nhello\nworld\nsuffix\n" + "x\n" * 400)
    (folder / ".gitattributes").write_text('"space name.py" export-ignore\n')
    (folder / "binary").write_bytes(b"a\0b")
    (folder / "carriage.py").write_bytes(b"first\rsecond\rthird")
    (folder / "colon:name.py").write_text("hello")
    # Non-portable paths fail closed instead of becoming Git pathspecs.
    rejected = await app.run_job("home_commit_candidate", message="bad path")
    assert not rejected.success
    (folder / "colon:name.py").unlink()
    result = await run(app, "home_commit_candidate", message="Navigation fixtures")
    tree = await run(app, "home_list_code_tree", path="plugins/beam/src", max_results=1)
    assert tree["revision"] == result["commit_sha"] and tree["truncated"]
    found = await run(app, "home_search_code", pattern="hello\nworld", path="plugins/beam/src")
    assert found["occurrence_count"] == 1
    assert found["matches"][0]["start_line"] == 1
    assert "hello" in found["matches"][0]["content"]
    carriage = await run(app, "home_search_code", pattern="second", path="plugins/beam/src/carriage.py")
    assert "> 2 | second" in carriage["matches"][0]["content"]
    read = await run(app, "home_read_code_file", path="plugins/beam/src/space name.py")
    assert read["end_line"] == 300 and read["truncated"]
    for kwargs in [
        {"path": "../escape"},
        {"path": ".git/HEAD"},
        {"path": "/etc/passwd"},
        {"path": "plugins/beam/src/binary"},
        {"path": "plugins/beam/src", "revision": "--help"},
    ]:
        assert not (await app.run_job("home_read_code_file", **kwargs)).success
    for pattern in ("[", "^", ""):
        assert not (await app.run_job("home_search_code", pattern=pattern)).success
    assert not (await app.run_job("home_list_code_tree", path=":(glob)**")).success


@pytest.mark.asyncio
async def test_lock_restrictions_and_unprepared(app):
    assert not (await app.run_job("home_code_status")).success
    assert not (await app.run_job("home_code_status", _allowed_paths=[])).success
    component = app.context.components["home.code"]["default"]
    with pytest.raises(ValueError, match="Unknown"):
        await component.run("reset")
    with pytest.raises(ValueError):
        await component.run("create_branch", branch_name=1)
    with _initialization_lock(component.workspace.state_path / "workspace.lock"):
        assert not (await app.run_job("home_list_code_tree")).success


@pytest.mark.asyncio
async def test_cancel_waits_for_owned_git_thread(prepared, monkeypatch):
    app, _, _ = prepared
    component = app.context.components["home.code"]["default"]
    entered, release, finished = Event(), Event(), Event()
    original = component._run

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        try:
            return original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(component, "_run", blocked)
    task = asyncio.create_task(component.run("status"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set() and not component._pending


@pytest.mark.asyncio
async def test_snapshot_size_and_regex_timeout(prepared, monkeypatch):
    app, repo, _ = prepared
    await create(app)
    await commit(app, repo, 'x = "' + "a" * 10000 + '!"\n')
    original = snapshot.subprocess.run

    def short_timeout(*args, **kwargs):
        if args[0][0] == snapshot.sys.executable:
            kwargs["timeout"] = 0.05
        return original(*args, **kwargs)  # pylint: disable=subprocess-run-check

    monkeypatch.setattr(snapshot.subprocess, "run", short_timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        snapshot.search_code(repo, {}, pattern="(a+)+$", path="plugins/beam/src")
    monkeypatch.setattr(snapshot.subprocess, "run", original)
    monkeypatch.setattr(snapshot, "MAX_FILE_BYTES", 3)
    response = await app.run_job("home_read_code_file", path="plugins/beam/src/reme_beam/__init__.py")
    assert not response.success


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["staged", "untracked"])
async def test_branch_operations_refuse_dirty_inputs(prepared, mutation):
    app, repo, _ = prepared
    if mutation == "staged":
        target = repo / "plugins/beam/src/reme_beam/__init__.py"
        target.write_text("staged = True\n")
        command(repo, "add", ".")
    else:
        target = repo / "plugins/beam/src/untracked file.py"
        target.write_text("untracked = True\n")
    before = target.read_bytes()
    assert not (await app.run_job("home_create_candidate_branch", branch_name="dirty")).success
    assert target.read_bytes() == before
    assert command(repo, "branch", "--show-current") == "init"


@pytest.mark.asyncio
async def test_exact_paths_rename_deletion_and_revision_pinning(prepared):
    app, repo, _ = prepared
    await create(app)
    original = repo / "plugins/beam/src/reme_beam/__init__.py"
    name = "plugins/beam/src/space name\nand newline.py"
    target = repo / name
    original.rename(target)
    first = await run(app, "home_commit_candidate", message="Rename a file")
    assert (await run(app, "home_read_code_file", path=name))["content"] == "value = 1\n"
    target.write_text("value = 9\n")
    pinned = await run(app, "home_commit_candidate", message="Improve renamed code")
    target.unlink()
    await run(app, "home_commit_candidate", message="Remove obsolete file")
    await run(app, "home_switch_candidate_branch", branch_name="init")
    read = await run(app, "home_read_code_file", path=name, revision=pinned["commit_sha"])
    assert read["content"] == "value = 9\n"
    assert name in (await run(app, "home_list_code_tree", revision=first["commit_sha"]))["paths"]


@pytest.mark.asyncio
async def test_reject_symlink_repository_and_detached_candidate(prepared):
    app, repo, _ = prepared
    await create(app)
    commit_sha = (await commit(app, repo))["commit_sha"]
    command(repo, "checkout", "--detach", commit_sha)
    assert not (await app.run_job("home_commit_candidate", message="detached")).success
    assert (await run(app, "home_read_code_file", path="plugins/beam/src/reme_beam/__init__.py"))["content"]
    moved = repo.with_name("moved")
    repo.rename(moved)
    repo.symlink_to(moved, target_is_directory=True)
    assert not (await app.run_job("home_code_status")).success
