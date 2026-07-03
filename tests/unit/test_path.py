"""Minimal unit tests for file_io path resolution."""

import tempfile
from pathlib import Path

from reme.steps.file_io._path import resolve_path


def test_resolve_path_accepts_absolute_inside_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        target, err = resolve_path(ws, str((ws / "Notes.md").resolve()))
        assert err is None
        assert target == (ws / "Notes.md").resolve()


def test_resolve_path_rejects_absolute_outside_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        target, err = resolve_path(ws, "/etc/passwd")
        assert target is None
        assert err is not None
        assert "workspace" in err.lower()


def test_resolve_path_rejects_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        target, err = resolve_path(ws, "../escape.md")
        assert target is None
        assert err is not None
        assert ".." in err


def test_resolve_path_accepts_relative():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        target, err = resolve_path(ws, "Notes/A.md")
        assert err is None
        assert target == (ws / "Notes/A.md").resolve()


def test_resolve_path_allow_empty_returns_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        target, err = resolve_path(ws, "", allow_empty=True)
        assert err is None
        assert target == ws.resolve()
