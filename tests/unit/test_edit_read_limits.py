"""Edits must not write back a prefix produced by a bounded file read."""

import asyncio
from functools import partial
from types import SimpleNamespace

import pytest

from reme.components import ApplicationContext
from reme.steps.file_io import _file_io, edit as edit_module


def _small_edit_limit(monkeypatch):
    """Exercise the real bounded reader with small files, without allocating 200 MiB."""
    monkeypatch.setattr(edit_module, "MAX_FILE_READ_BYTES", 64, raising=False)
    monkeypatch.setattr(edit_module, "read_file_safe", partial(_file_io.read_file_safe, max_bytes=64))


@pytest.mark.parametrize("suffix", [".md", ".txt"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_edit_rejects_truncated_read_without_writing(tmp_path, monkeypatch, suffix, encoding):
    """An occurrence in the prefix must not cause the unread tail to be discarded."""
    _small_edit_limit(monkeypatch)
    path = tmp_path / ("note" + suffix)
    source = ("target" + "x" * 80 + "TAIL").encode(encoding)
    path.write_bytes(source)
    app = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        step = edit_module.EditStep(app_context=app)
        await step(path=path.name, old="target", new="edited")
        assert path.read_bytes() == source
        assert step.context.response.success is False
        assert "limit" in step.context.response.answer

    asyncio.run(run())


@pytest.mark.parametrize("suffix", [".md", ".txt"])
@pytest.mark.parametrize("encoding,filler", [("utf-8", 58), ("utf-16", 25)])
def test_edit_at_read_limit_preserves_encoding(tmp_path, monkeypatch, suffix, encoding, filler):
    """The exact byte limit remains editable, with the existing write rules."""
    _small_edit_limit(monkeypatch)
    text = "target" + "x" * filler
    source = text.encode(encoding)
    assert len(source) == 64
    path = tmp_path / ("note" + suffix)
    path.write_bytes(source)
    app = ApplicationContext(workspace_dir=str(tmp_path))

    async def run():
        step = edit_module.EditStep(app_context=app)
        await step(path=path.name, old="target", new="edited")
        assert step.context.response.success is True
        expected = text.replace("target", "edited") + ("\n" if suffix == ".md" else "")
        assert path.read_bytes() == expected.encode(encoding)

    asyncio.run(run())


def test_edit_rejects_growth_after_stat(tmp_path, monkeypatch):
    """A stale small stat cannot turn a larger actual read into a destructive edit."""
    _small_edit_limit(monkeypatch)
    path = tmp_path / "note.txt"
    source = b"target" + b"x" * 80 + b"TAIL"
    path.write_bytes(source)
    app = ApplicationContext(workspace_dir=str(tmp_path))

    async def stale_stat(_path):
        return SimpleNamespace(st_size=8)

    monkeypatch.setattr(_file_io.aiofiles.os, "stat", stale_stat)

    async def run():
        step = edit_module.EditStep(app_context=app)
        await step(path=path.name, old="target", new="edited")
        assert step.context.response.success is False
        assert "incomplete contents" in step.context.response.answer
        assert path.read_bytes() == source

    asyncio.run(run())


@pytest.mark.parametrize("size", [0, 63, 64, 65])
def test_read_helper_complete_mode_and_default_truncation(tmp_path, size):
    """Ordinary limited reads keep their behavior; complete reads reject only overflow."""
    path = tmp_path / "note.txt"
    source = b"x" * size
    path.write_bytes(source)

    async def run():
        text, _ = await _file_io.read_file_safe(path, max_bytes=64)
        assert text == "x" * min(size, 64)
        if size > 64:
            with pytest.raises(ValueError, match="limit"):
                await _file_io.read_file_safe(path, max_bytes=64, require_complete=True)
        else:
            text, _ = await _file_io.read_file_safe(path, max_bytes=64, require_complete=True)
            assert text == "x" * size
        assert path.read_bytes() == source

    asyncio.run(run())
