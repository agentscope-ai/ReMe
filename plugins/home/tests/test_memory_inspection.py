"""Memory evidence Jobs against isolated real Git repositories."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import shutil

import pytest
from pydantic import ValidationError
from reme_home.evaluation import memory_inspection_step as memory
from reme_home.preparation.code import git

from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    home_application,
    preparation_inputs,
    prepared_home,
)
from test_evaluation_inspection import call, snapshot, write_run


@pytest.fixture
def evidence(tmp_path):
    home = tmp_path / "home"
    run = write_run(home, case_ids=["case1"])
    repo = run / "cases/case1/memory_construction/reme_workspace"
    repo.mkdir(parents=True)
    git(repo, "init")
    git(repo, "config", "user.name", "test")
    git(repo, "config", "user.email", "test@localhost")
    (repo / "memory").mkdir()
    (repo / "memory/note.md").write_text("old fact\n")
    (repo / "removed.txt").write_text("removed\n")
    first = commit(repo, "session: session1")
    (repo / "memory/note.md").write_text("new fact\n")
    (repo / "removed.txt").unlink()
    (repo / "added.bin").write_bytes(b"\x00\xff")
    second = commit(repo, "session: session2")
    return home, run, repo, first, second


def commit(repo, subject):
    git(repo, "add", "--all")
    git(repo, "commit", "-m", subject)
    return git(repo, "rev-parse", "HEAD")


def history(run, **kwargs):
    return memory.MemoryHistoryInput(validation_path=str(run), case_id="case1", **kwargs)


def comparison(run, **kwargs):
    return memory.CompareMemoryInput(validation_path=str(run), case_id="case1", **kwargs)


def test_history_paging_filter_and_no_writes(evidence):
    home, run, repo, first, second = evidence
    (repo / "memory/note.md").write_text("dirty checkout ignored\n")
    before = snapshot(home)
    result = memory.inspect_history(home, history(run, limit=1))
    assert result["head"] == second and result["row_count"] == 2
    assert result["rows"][0]["session_id"] == "session2" and result["next_offset"] == 1
    result = memory.inspect_history(home, history(run, offset=1, limit=1))
    assert result["rows"][0]["commit_sha"] == first and result["next_offset"] is None
    assert memory.inspect_history(home, history(run, path="added.bin"))["row_count"] == 1
    assert memory.inspect_history(home, history(run, path="memory/*"))["rows"] == []
    assert snapshot(home) == before


def test_same_repository_diff_and_unique_prefix(evidence):
    home, run, repo, first, second = evidence
    (repo / "memory/note.md").write_text("dirty checkout ignored\n")
    before = snapshot(home)
    result = memory.compare_memory(home, comparison(run, base_ref=first[:7].upper()))
    assert result["base_commit"] == first and result["target_commit"] == second
    assert result["added"] == ["added.bin"] and result["removed"] == ["removed.txt"]
    assert result["modified"] == ["memory/note.md"]
    assert "-old fact" in result["diff"] and "+new fact" in result["diff"]
    assert "Binary files differ" in result["diff"] and not result["truncated"]
    assert "dirty checkout" not in result["diff"]
    assert memory.compare_memory(home, comparison(run, base_ref="HEAD~1", path="memory"))["added"] == []
    assert snapshot(home) == before


def test_cross_evaluation_final_snapshots(evidence):
    home, run, _repo, _first, second = evidence
    other = write_run(home, "2", case_ids=["case1"], commit="b" * 40)
    target = other / "cases/case1/memory_construction/reme_workspace"
    target.mkdir(parents=True)
    git(target, "init")
    git(target, "config", "user.name", "test")
    git(target, "config", "user.email", "test@localhost")
    (target / "target.txt").write_text("target\n")
    target_head = commit(target, "new history")
    before = snapshot(home)
    result = memory.compare_memory(home, comparison(run, target_validation_path=str(other)))
    assert result["base_commit"] == second and result["target_commit"] == target_head
    assert result["added"] == ["target.txt"]
    assert result["removed"] == ["added.bin", "memory/note.md"]
    assert result["base_tree_sha256"] != result["target_tree_sha256"]
    assert snapshot(home) == before


@pytest.mark.parametrize("state", ["workspace_missing", "history_missing", "history_empty"])
def test_missing_and_empty_history(tmp_path, state):
    home = tmp_path / "home"
    run = write_run(home, case_ids=["case1"])
    repo = run / "cases/case1/memory_construction/reme_workspace"
    if state != "workspace_missing":
        repo.mkdir(parents=True)
    if state == "history_empty":
        git(repo, "init")
    result = memory.inspect_history(home, history(run))
    assert result["status"] == state and result["rows"] == [] and not result["has_git_history"]
    with pytest.raises(ValueError, match="requires memory commits"):
        memory.compare_memory(home, comparison(run, base_ref="HEAD"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset": -1},
        {"limit": 101},
        {"limit": True},
        {"case_id": "../bad"},
        {"path": "../escape"},
        {"path": ".git/config"},
    ],
)
def test_input_validation(kwargs):
    with pytest.raises(ValidationError):
        memory.MemoryHistoryInput.model_validate({"validation_path": "a", "case_id": "case1", **kwargs})


def test_comparison_requires_base():
    with pytest.raises(ValidationError, match="base_ref is required"):
        comparison("a")


@pytest.mark.parametrize("kind", ["gitfile", "symlink", "alternates", "corrupt"])
def test_invalid_or_external_git_storage(evidence, tmp_path, kind):
    home, run, repo, _first, _second = evidence
    if kind == "gitfile":
        (repo / ".git").rename(repo / "saved-git")
        (repo / ".git").write_text("gitdir: saved-git\n")
    elif kind == "symlink":
        (repo / ".git/objects/external").symlink_to(tmp_path)
    elif kind == "alternates":
        (repo / ".git/objects/info/alternates").write_text(str(tmp_path))
    else:
        (repo / ".git/HEAD").write_text("broken\n")
    with pytest.raises((ValueError, memory.subprocess.CalledProcessError)):
        memory.inspect_history(home, history(run))


def test_scope_and_limits(evidence, monkeypatch):
    home, run, _repo, first, _second = evidence
    with pytest.raises(ValueError, match="case_id"):
        memory.inspect_history(home, memory.MemoryHistoryInput(validation_path=str(run), case_id="other"))
    with pytest.raises(ValueError):
        memory.inspect_history(home, history(home))
    monkeypatch.setattr(memory, "MAX_OUTPUT", 20)
    result = memory.compare_memory(home, comparison(run, base_ref=first))
    assert result["truncated"] and len(result["diff"]) <= 20
    monkeypatch.setattr(memory, "MAX_TREE_FILES", 1)
    with pytest.raises(ValueError, match="2000 files"):
        memory.compare_memory(home, comparison(run, base_ref=first))


def test_ambiguous_and_unreachable_revisions(evidence):
    home, run, repo, first, second = evidence
    # Publish real commits until two reachable SHAs share a one-character prefix.
    previous = second
    seen = {first[0]: first, second[0]: second}
    for index in range(100):
        sha = git(repo, "commit-tree", "HEAD^{tree}", "-p", previous, "-m", str(index))
        previous = sha
        if sha[0] in seen:
            prefix = sha[0]
            break
        seen[sha[0]] = sha
    git(repo, "update-ref", "HEAD", previous)
    with pytest.raises(ValueError, match="matches reachable"):
        memory.compare_memory(home, comparison(run, base_ref=prefix))
    with pytest.raises(ValueError, match="0 matches"):
        memory.compare_memory(home, comparison(run, base_ref="f" * 40))
    detached = git(repo, "commit-tree", "HEAD^{tree}", "-m", "unreachable")
    with pytest.raises(memory.subprocess.CalledProcessError):
        memory.compare_memory(home, comparison(run, base_ref=detached + "~0"))


@pytest.mark.asyncio
async def test_registered_jobs_and_restricted_scope(app, prepared, evidence):
    run = write_run(prepared, case_ids=["case1"])
    result = await call(app, "home_inspect_memory_history", validation_path=str(run), case_id="case1")
    assert result["status"] == "workspace_missing"
    response = await app.run_job("home_compare_memory", validation_path=str(run), case_id="case1")
    assert not response.success and "base_ref" in str(response.answer)
    response = await app.run_job(
        "home_inspect_memory_history",
        validation_path=str(run),
        case_id="case1",
        _allowed_paths=[str(run)],
    )
    assert not response.success and "unrestricted" in str(response.answer)

    _home, _run, repository, first, second = evidence
    destination = run / "cases/case1/memory_construction/reme_workspace"
    shutil.copytree(repository, destination)
    result = await call(app, "home_compare_memory", validation_path=str(run), case_id="case1", base_ref=first)
    assert result["target_commit"] == second and result["modified"] == ["memory/note.md"]


def test_unterminated_lines_and_snapshot_byte_limit(evidence, monkeypatch):
    home, run, repo, _first, second = evidence
    (repo / "memory/note.md").write_text("unterminated")
    commit(repo, "ordinary checkpoint")
    result = memory.compare_memory(home, comparison(run, base_ref=second, path="memory"))
    assert "+unterminated\n\\ No newline at end of file\n" in result["diff"]
    assert memory.inspect_history(home, history(run))["rows"][0]["session_id"] is None
    monkeypatch.setattr(memory, "MAX_TREE_BYTES", 1)
    with pytest.raises(ValueError, match="20 MB"):
        memory.compare_memory(home, comparison(run, base_ref=second))


def test_tracked_symlinks_are_not_followed(evidence):
    home, run, repo, _first, second = evidence
    (repo / "memory/link").symlink_to("/etc/passwd")
    commit(repo, "link")
    with pytest.raises(ValueError, match="regular files"):
        memory.compare_memory(home, comparison(run, base_ref=second))
