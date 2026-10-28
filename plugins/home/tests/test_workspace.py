"""Home layout initialization through the application and component lifecycle."""

import json
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from reme_home.workspace import DIRECTORIES, HomeWorkspace, _initialization_lock

from reme.application import Application
from reme.components.application_context import ApplicationContext


def component(root, **config):
    """Build an isolated component without starting model or service clients."""
    return HomeWorkspace(app_context=ApplicationContext(workspace_dir=str(root), **config))


@pytest.mark.asyncio
async def test_application_start_initializes_enabled_home(tmp_path, monkeypatch):
    """The installed plugin contributes its component without changing core."""
    entry = EntryPoint(name="home", value="reme_home", group="reme.plugins")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "beam-judge/src"))
    judge = EntryPoint(name="beam-judge", value="judge_beam", group="reme.plugins")
    monkeypatch.setattr("reme.plugin.find_entry_points", lambda group, name: [entry if name == "home" else judge])
    monkeypatch.setenv("LLM_API_KEY", "test")
    app = Application(
        workspace_dir=str(tmp_path),
        plugins=["home", "beam-judge"],
        service={"backend": "cli"},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )
    assert not (tmp_path / "home/code").exists()
    try:
        await app.start()
        assert all((tmp_path / "home" / path).is_dir() for path in DIRECTORIES)
        assert (tmp_path / "session").is_dir()
        assert (tmp_path / "home/metadata/workspace.json").is_file()
        assert not (tmp_path / "code").exists()
        assert not (tmp_path / "metadata/home").exists()
        workspace = app.context.components["home.workspace"]["default"]
        assert workspace.home_path == tmp_path / "home"
        assert workspace.state_path == tmp_path / "home/metadata"
        assert app.context.components["home.workspace"]["default"].manifest is not None
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_application_without_home_keeps_standard_layout(tmp_path):
    """Ordinary ReMe startup does not acquire Home directories."""
    app = Application(
        workspace_dir=str(tmp_path),
        service={"backend": "cli"},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )
    try:
        await app.start()
        assert not (tmp_path / "home/code").exists()
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_empty_roots_and_existing_memory_are_preserved(tmp_path):
    """First initialization accepts empty Home roots alongside user memory."""
    (tmp_path / "daily").mkdir()
    note = tmp_path / "daily/note.md"
    note.write_text("owned memory", encoding="utf-8")
    (tmp_path / "code").mkdir()
    unrelated = tmp_path / "code/user.py"
    unrelated.write_text("user-owned code", encoding="utf-8")
    (tmp_path / "home/code").mkdir(parents=True)
    workspace = component(tmp_path, metadata_dir="state")
    await workspace.start()
    await workspace.close()
    marker = tmp_path / "home/metadata/workspace.json"
    before = marker.read_bytes()
    (tmp_path / "home/code/repo/candidate.py").write_text("candidate", encoding="utf-8")
    (tmp_path / "home/sampling/plans").rmdir()
    reopened = component(tmp_path, metadata_dir="state")
    await reopened.start()
    await reopened.close()
    assert marker.read_bytes() == before
    assert note.read_text(encoding="utf-8") == "owned memory"
    assert unrelated.read_text(encoding="utf-8") == "user-owned code"
    assert (tmp_path / "home/code/repo/candidate.py").read_text(encoding="utf-8") == "candidate"
    assert (tmp_path / "home/sampling/plans").is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", ["code/note.txt", "code/repo", "docs/guide.md"])
async def test_first_initialization_rejects_nonempty_roots(tmp_path, relative):
    """Even an empty child directory makes a top-level directory nonempty."""
    target = tmp_path / "home" / relative
    target.parent.mkdir(parents=True)
    if relative.endswith("repo"):
        target.mkdir()
    else:
        target.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty Home directory"):
        await component(tmp_path).start()
    assert target.exists()
    assert not (tmp_path / "home/metadata/workspace.json").exists()
    assert not (tmp_path / "home/datasets").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", ["code", "metadata", "metadata/workspace.json", "."])
async def test_rejects_file_conflicts(tmp_path, relative):
    """Conflicting files are retained and reported."""
    target = tmp_path / "home" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("keep", encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        await component(tmp_path).start()
    assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", ["code", "metadata", "metadata/workspace.json", "."])
async def test_rejects_symlinks(tmp_path, relative):
    """Managed paths must not follow links, even to another internal path."""
    target = tmp_path / "home" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / "destination"
    destination.mkdir()
    target.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError):
        await component(tmp_path).start()
    assert not list(destination.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["broken json", '{"schema_version": 2}', '{"schema_version": 1}'])
async def test_rejects_invalid_manifest(tmp_path, content):
    """A broken or unsupported manifest never resets workspace identity."""
    marker = tmp_path / "home/metadata/workspace.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        await component(tmp_path).start()
    assert marker.read_text(encoding="utf-8") == content
    assert not (tmp_path / "home/code").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [{"daily_dir": "home/code"}, {"session_dir": "home/docs/dialog"}, {"metadata_dir": "home"}, {"metadata_dir": ".."}],
)
async def test_rejects_configuration_collisions(tmp_path, config):
    """ReMe data directories cannot overlap Home's reserved layout."""
    with pytest.raises(ValueError):
        await component(tmp_path, **config).start()
    assert not (tmp_path / "home/code").exists()


@pytest.mark.asyncio
async def test_publication_failure_rolls_back_only_created_directories(tmp_path, monkeypatch):
    """Failed initialization can be retried without touching pre-existing paths."""
    (tmp_path / "home/code").mkdir(parents=True)
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")
    workspace = component(tmp_path)

    def fail_replace(*_args):
        raise OSError("disk error")

    with monkeypatch.context() as patch:
        patch.setattr("reme_home.workspace.os.replace", fail_replace)
        with pytest.raises(OSError, match="disk error"):
            await workspace.start()
    assert (tmp_path / "home/code").is_dir()
    assert not list((tmp_path / "home/code").iterdir())
    assert not (tmp_path / "home/datasets").exists()
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not list((tmp_path / "home/metadata").glob("*.tmp"))
    await workspace.start()
    await workspace.close()
    assert json.loads((tmp_path / "home/metadata/workspace.json").read_text())["schema_version"] == 1


@pytest.mark.asyncio
async def test_initialization_lock_prevents_competing_writer(tmp_path):
    """A competing initializer fails explicitly and can retry after release."""
    state = tmp_path / "home/metadata"
    state.mkdir(parents=True)
    with _initialization_lock(state / "workspace.lock"):
        with pytest.raises(RuntimeError, match="already in progress"):
            await component(tmp_path).start()
    workspace = component(tmp_path)
    await workspace.start()
    await workspace.close()


@pytest.mark.asyncio
async def test_legacy_layout_requires_new_workspace(tmp_path):
    """Never silently adopt or move a previously initialized flat workspace."""
    marker = tmp_path / "metadata/home/workspace.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"legacy": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Legacy Home workspace layout"):
        await component(tmp_path).start()
    assert marker.read_text(encoding="utf-8") == '{"legacy": true}'
    assert not (tmp_path / "home").exists()
