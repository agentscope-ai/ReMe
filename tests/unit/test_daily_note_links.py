"""Daily-note scans preserve internal links without reading outside the workspace."""

import errno
from pathlib import Path

import pytest

from reme.steps.file_io import _daily_index

DAY = "2026-01-01"


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    """Skip platforms where creating a test symlink is unavailable."""
    try:
        link.symlink_to(target, target_is_directory=directory)
    except NotImplementedError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    except OSError as exc:
        if exc.errno in {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP}:
            pytest.skip(f"Symlinks unavailable: {exc}")
        raise


@pytest.fixture(name="daily_workspace")
def _daily_workspace(tmp_path):
    """Keep note targets and an outside sibling under pytest's temporary directory."""
    workspace = (tmp_path / "workspace").resolve()
    day_dir = workspace / "daily" / DAY
    day_dir.mkdir(parents=True)
    (day_dir / "regular.md").write_text("---\nname: regular\n---\nBody\n", encoding="utf-8")
    return workspace, day_dir


@pytest.mark.parametrize("kind", ["internal", "escaping", "dangling", "self-loop", "mutual-loop"])
def test_scan_note_links_keep_logical_paths_and_skip_unreadable_targets(kind, daily_workspace, monkeypatch):
    """Unsafe or unreadable links must neither hide a valid note nor reach read_text."""
    workspace, day_dir = daily_workspace
    link = day_dir / "linked.md"
    target = workspace / "target.md"
    if kind == "escaping":
        target = workspace.parent / "outside.md"
    if kind in {"internal", "escaping"}:
        target.write_text("---\nname: linked\nsource_resource: '[[resource/photo.png]]'\n---\nBody\n", encoding="utf-8")
    elif kind == "self-loop":
        target = link
    elif kind == "mutual-loop":
        target = day_dir / "other.md"
        _symlink(link=target, target=link)
    _symlink(link, target)

    read_paths = []
    original_read = Path.read_text

    def record_read(path, *args, **kwargs):
        read_paths.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)
    notes = _daily_index.scan_notes(workspace, DAY, "daily")

    expected_paths = [f"daily/{DAY}/regular.md"]
    if kind == "internal":
        expected_paths.insert(0, f"daily/{DAY}/linked.md")
        assert notes[0]["metadata"]["source_resource"] == "[[resource/photo.png]]"
        assert target in read_paths
    else:
        assert read_paths == [day_dir / "regular.md"]
    assert [note["path"] for note in notes] == expected_paths
    assert link.is_symlink()


@pytest.mark.parametrize("kind", ["internal", "escaping", "dangling", "self-loop", "mutual-loop"])
def test_scan_date_directory_links(kind, daily_workspace, monkeypatch):
    """A valid date alias retains its logical date; invalid aliases are not read."""
    workspace, day_dir = daily_workspace
    alias = day_dir.parent / "2026-01-02"
    target = day_dir
    if kind == "escaping":
        target = workspace.parent / "outside-day"
        target.mkdir()
        (target / "outside.md").write_text("---\nname: outside\n---\nBody\n", encoding="utf-8")
    elif kind == "dangling":
        target = day_dir.parent / "missing"
    elif kind == "self-loop":
        target = alias
    elif kind == "mutual-loop":
        target = day_dir.parent / "2026-01-03"
        _symlink(target, alias, directory=True)
    _symlink(alias, target, directory=True)

    read_paths = []
    original_read = Path.read_text

    def record_read(path, *args, **kwargs):
        read_paths.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)
    notes = _daily_index.scan_notes(workspace, alias.name, "daily")

    if kind == "internal":
        assert notes == [{"path": "daily/2026-01-02/regular.md", "metadata": {"name": "regular"}}]
        assert read_paths == [day_dir / "regular.md"]
    else:
        assert not notes
        assert not read_paths
    assert alias.is_symlink()


@pytest.mark.parametrize(
    "method,error_number",
    [("is_file", errno.ELOOP), ("is_file", errno.EIO), ("iterdir", errno.EACCES)],
)
def test_scan_only_suppresses_loop_errors(method, error_number, daily_workspace, monkeypatch):
    """Do not turn unrelated enumeration or stat failures into an empty history."""
    workspace, day_dir = daily_workspace
    note = day_dir / "regular.md"
    error = OSError(error_number, "injected filesystem error")
    original = getattr(Path, method)
    target = note if method == "is_file" else day_dir

    def fail_operation(path, *args, **kwargs):
        if path == target:
            raise error
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, fail_operation)

    if error_number == errno.ELOOP:
        assert not _daily_index.scan_notes(workspace, DAY, "daily")
    else:
        with pytest.raises(OSError) as captured:
            _daily_index.scan_notes(workspace, DAY, "daily")
        assert captured.value.errno == error_number
