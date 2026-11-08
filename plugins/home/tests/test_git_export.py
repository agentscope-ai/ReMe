"""Git exports preserve raw committed bytes, literal names, and executable modes."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from reme_home.preparation import code
from reme_home.code import repository as candidate_repository, snapshot


def test_batch_export_bypasses_filters_and_checkout(tmp_path, monkeypatch):
    """One batch reads all selected blobs, including binary and large files."""
    repository = tmp_path / "repo"
    repository.mkdir()
    code.git(repository, "init", "--template=")
    contents = {
        "space name": b"$Format:%H$\n",
        'quotes"\tand\nnewline': b"\x00binary\xff\n",
        "脚本": b"#!/bin/sh\nexit 0\n",
        "large": b"x" * 5_000_001,
        "empty": b"",
        ".gitattributes": b"* export-ignore export-subst filter=blocked\n",
    }
    for name, data in contents.items():
        (repository / name).write_bytes(data)
    (repository / "脚本").chmod(0o755)
    code.git(repository, "add", "--all")
    code.git(repository, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-m", "files")
    code.git(repository, "config", "filter.blocked.process", "/bin/false")
    code.git(repository, "config", "filter.blocked.required", "true")
    for name in contents:
        (repository / name).write_bytes(b"dirty checkout")
    run = Mock(wraps=code.git_bytes)
    monkeypatch.setattr(code, "git_bytes", run)
    destination = tmp_path / "export"

    code.export_git_tree(repository, code.git_tree(repository, "HEAD"), lambda path: destination / path)

    assert run.call_count == 2
    assert run.call_args.args == (repository, "cat-file", "--batch")
    for name, data in contents.items():
        target = destination / name
        assert target.read_bytes() == data
        assert target.stat().st_mode & 0o777 == (0o755 if name == "脚本" else 0o644)
    # Browsing and single-file export use the same committed objects while
    # retaining their own size limits and streaming behavior.
    objects = dict(snapshot.tree(repository, "HEAD"))
    assert set(objects) == set(contents)
    assert snapshot.blob(repository, objects['quotes"\tand\nnewline']) == contents['quotes"\tand\nnewline']
    with pytest.raises(ValueError, match="exceeds 5 MB"):
        snapshot.blob(repository, objects["large"])
    streamed = tmp_path / "streamed"
    snapshot.export_blob(repository, objects["large"], streamed)
    assert streamed.read_bytes() == contents["large"]


@pytest.mark.parametrize("reader", ["preparation", "snapshot", "candidate"])
@pytest.mark.parametrize(
    "mode,kind,name",
    [
        ("120000", "blob", "link"),
        ("160000", "commit", "submodule"),
        ("100644", "blob", ".git/config"),
        ("100644", "blob", "../escape"),
        ("100644", "blob", "/escape"),
        ("100644", "blob", "~escape"),
        ("100644", "blob", "folder\\escape"),
        ("100644", "blob", "other:bad.py"),
    ],
)
def test_tree_rejects_unsafe_paths_and_nonregular_files(monkeypatch, mode, kind, name, reader):
    """Reject unsafe object metadata before any file is exported."""
    monkeypatch.setattr(code, "git_bytes", Mock(return_value=f"{mode} {kind} {'a' * 40}\t{name}\0".encode()))
    monkeypatch.setattr(candidate_repository, "git", Mock(return_value=""))
    with pytest.raises(ValueError):
        if reader == "preparation":
            code.git_tree(Path("unused"), "HEAD")
        elif reader == "snapshot":
            snapshot.tree(Path("unused"), "HEAD")
        else:
            candidate_repository.validate_tree(
                Path("unused"),
                {"code": {"candidate_paths": ["plugin"], "initial_commit": "init"}},
                "HEAD",
            )


def test_tree_selection_uses_literal_roots(monkeypatch):
    """Ignore out-of-scope links without mistaking similar prefixes for selected roots."""
    raw = (
        f"100644 blob {'a' * 40}\tplugin/file\0"
        f"120000 blob {'b' * 40}\tplugin-other/link\0"
        f"160000 commit {'c' * 40}\tother\0"
    ).encode()
    monkeypatch.setattr(code, "git_bytes", Mock(return_value=raw))
    assert code.git_tree(Path("unused"), "HEAD", [Path("plugin")]) == [(Path("plugin/file"), "100644", "a" * 40)]
    assert snapshot.tree(Path("unused"), "HEAD", "plugin") == [("plugin/file", "a" * 40)]


def test_snapshot_selection_treats_metacharacters_literally(monkeypatch):
    """A browsing prefix selects exact names and descendants, never glob matches."""
    names = ['literal[ab]/quotes"\tand\nnewline', "literala/unselected", "literal[ab]-other/unselected"]
    raw = "".join(f"100755 blob {'a' * 40}\t{name}\0" for name in names).encode()
    monkeypatch.setattr(code, "git_bytes", Mock(return_value=raw))
    assert snapshot.tree(Path("unused"), "HEAD", "literal[ab]") == [(names[0], "a" * 40)]
    assert snapshot.tree(Path("unused"), "HEAD", names[0]) == [(names[0], "a" * 40)]


def test_snapshot_browsing_ignores_invalid_names_outside_selected_path(tmp_path):
    """Unrelated Git names cannot prevent browsing a valid literal directory."""
    repository = tmp_path / "repo"
    repository.mkdir()
    code.git(repository, "init", "--template=")
    (repository / "good").mkdir()
    (repository / "good/file.py").write_text("value = 1\n")
    (repository / "other:bad.py").write_text("outside browsing prefix\n")
    code.git(repository, "add", "--all")
    code.git(repository, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-m", "files")

    assert snapshot.list_code_tree(repository, {}, path="good")["paths"] == ["good/file.py"]
    assert snapshot.read_code_file(repository, {}, path="good/file.py")["content"] == "value = 1\n"
    assert snapshot.search_code(repository, {}, path="good", pattern="value")["occurrence_count"] == 1
    with pytest.raises(ValueError, match="Invalid relative path"):
        snapshot.tree(repository, "HEAD")
    with pytest.raises(ValueError, match="Invalid relative path"):
        code.git_tree(repository, "HEAD", [Path("good")])
