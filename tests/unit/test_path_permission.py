"""Tests for PrefixCheck white/black path-prefix permission filtering.

Covers ReadStep and WriteStep:
- Path-component boundary correctness (``daily`` vs ``daily-report``).
- Path form variants: ``./`` prefix, absolute paths, ``~/`` paths.
- White/black stage semantics: None (inactive), [] (deny-all / no-op), combined.
"""

# pylint: disable=protected-access

import os
import tempfile
from pathlib import Path

import pytest

from reme.components.file_store import LocalFileStore
from reme.steps.file_io import read as crud_read
from reme.steps.file_io import write as crud_write


class temp_chdir:
    """Context manager to temporarily chdir into a path and restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


def _seed(workspace: Path, rel: str, body: str = "body\n") -> Path:
    target = workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


async def _make_store() -> LocalFileStore:
    store = LocalFileStore(name="t_perm", embedding_store="")
    await store.start()
    return store


async def _read(store: LocalFileStore, *, step_kwargs: dict | None = None, **kwargs):
    """Run a ReadStep; ``step_kwargs`` go to step init (white/black_path_prefix)."""
    step = crud_read.ReadStep(file_store=store, **(step_kwargs or {}))
    await step(**kwargs)
    return step.context.response


async def _write(store: LocalFileStore, *, step_kwargs: dict | None = None, **kwargs):
    """Run a WriteStep; ``step_kwargs`` go to step init (white/black_path_prefix)."""
    step = crud_write.WriteStep(file_store=store, **(step_kwargs or {}))
    await step(**kwargs)
    return step.context.response


# -- white stage ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_white_prefix_boundary_and_nesting():
    """white=["daily"] allows daily/... (any depth) but denies daily-report/..."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/notes.md", "daily notes\n")
        _seed(Path(tmp), "daily/sub/deep.md", "deep\n")
        _seed(Path(tmp), "daily-report/secret.md", "leaked\n")
        store = await _make_store()
        kw = {"white_path_prefix": ["daily"]}

        resp = await _read(store, step_kwargs=kw, path="daily/notes.md")
        assert resp.success is True
        assert "daily notes" in str(resp.answer)

        resp = await _read(store, step_kwargs=kw, path="daily/sub/deep.md")
        assert resp.success is True

        resp = await _read(store, step_kwargs=kw, path="daily-report/secret.md")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_white_exact_file_and_empty_list():
    """white=["foo/bar.md"] allows only that file; white=[] denies everything."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "foo/bar.md", "bar\n")
        _seed(Path(tmp), "foo/baz.md", "baz\n")
        _seed(Path(tmp), "a.md", "a\n")
        store = await _make_store()

        resp = await _read(store, step_kwargs={"white_path_prefix": ["foo/bar.md"]}, path="foo/bar.md")
        assert resp.success is True
        resp = await _read(store, step_kwargs={"white_path_prefix": ["foo/bar.md"]}, path="foo/baz.md")
        assert resp.success is False

        resp = await _read(store, step_kwargs={"white_path_prefix": []}, path="a.md")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


# -- black stage ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_black_prefix_boundary_and_empty():
    """black=["secret"] denies secret/... but allows secret-notes/...; black=[] is inactive."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "secret/x.md", "hidden\n")
        _seed(Path(tmp), "secret-notes/public.md", "public\n")
        _seed(Path(tmp), "a.md", "a\n")
        store = await _make_store()
        kw = {"black_path_prefix": ["secret"]}

        resp = await _read(store, step_kwargs=kw, path="secret/x.md")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()

        resp = await _read(store, step_kwargs=kw, path="secret-notes/public.md")
        assert resp.success is True
        assert "public" in str(resp.answer)

        resp = await _read(store, step_kwargs={"black_path_prefix": []}, path="a.md")
        assert resp.success is True
        await store.close()


# -- combined & default ----------------------------------------------------------


@pytest.mark.asyncio
async def test_white_then_black_combined():
    """white=["daily"] + black=["daily/secret"] — allows daily/notes, denies daily/secret/."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/notes.md", "ok\n")
        _seed(Path(tmp), "daily/secret/x.md", "hidden\n")
        store = await _make_store()
        kw = {"white_path_prefix": ["daily"], "black_path_prefix": ["daily/secret"]}

        resp = await _read(store, step_kwargs=kw, path="daily/notes.md")
        assert resp.success is True

        resp = await _read(store, step_kwargs=kw, path="daily/secret/x.md")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_no_filters_allows_all():
    """Default (no white/black configured) imposes no restriction."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "anywhere/x.md", "free\n")
        store = await _make_store()
        resp = await _read(store, path="anywhere/x.md")
        assert resp.success is True
        await store.close()


# -- path form variants ----------------------------------------------------------


@pytest.mark.asyncio
async def test_dot_slash_prefix_path():
    """path="./daily/notes.md" resolves identically to "daily/notes.md"."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/notes.md", "dot slash\n")
        _seed(Path(tmp), "daily-report/x.md", "leak\n")
        store = await _make_store()
        kw = {"white_path_prefix": ["daily"]}

        resp = await _read(store, step_kwargs=kw, path="./daily/notes.md")
        assert resp.success is True
        assert "dot slash" in str(resp.answer)

        resp = await _read(store, step_kwargs=kw, path="./daily-report/x.md")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_absolute_path_inside_workspace():
    """Absolute path inside workspace passes permission check normally."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/notes.md", "abs ok\n")
        _seed(Path(tmp), "secret/x.md", "abs deny\n")
        store = await _make_store()

        abs_ok = str(Path(tmp).resolve() / "daily" / "notes.md")
        resp = await _read(store, step_kwargs={"white_path_prefix": ["daily"]}, path=abs_ok)
        assert resp.success is True
        assert "abs ok" in str(resp.answer)

        abs_deny = str(Path(tmp).resolve() / "secret" / "x.md")
        resp = await _read(store, step_kwargs={"white_path_prefix": ["daily"]}, path=abs_deny)
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_absolute_path_outside_workspace():
    """Absolute path outside workspace is rejected by resolve_path before permission check."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _read(store, step_kwargs={"white_path_prefix": ["daily"]}, path="/etc/passwd")
        assert resp.success is False
        assert "no permission" not in str(resp.answer).lower()  # rejected by path resolution
        await store.close()


@pytest.mark.asyncio
async def test_tilde_path_expanded_to_home():
    """~/... in path arg is expanded via expanduser(); outside workspace → rejected."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()

        # ~/notes.md expands to $HOME/notes.md which is outside workspace → rejected.
        resp = await _read(store, path="~/notes.md")
        assert resp.success is False
        assert "inside the workspace" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_tilde_prefix_in_config_expanded_and_dropped():
    """~/... in prefix config IS expanded by _normalize_prefixes; outside workspace → dropped."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/notes.md", "ok\n")
        store = await _make_store()

        # white=["~/daily"] expands to $HOME/daily which is outside workspace → dropped → [].
        # An empty white list denies everything.
        resp = await _read(store, step_kwargs={"white_path_prefix": ["~/daily"]}, path="daily/notes.md")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_dot_slash_in_prefix_config():
    """Prefix "./daily" normalizes to "daily" and works identically."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/notes.md", "prefix dot\n")
        _seed(Path(tmp), "daily-report/x.md", "leak\n")
        store = await _make_store()
        kw = {"white_path_prefix": ["./daily"]}

        resp = await _read(store, step_kwargs=kw, path="daily/notes.md")
        assert resp.success is True

        resp = await _read(store, step_kwargs=kw, path="daily-report/x.md")
        assert resp.success is False
        await store.close()


@pytest.mark.asyncio
async def test_absolute_prefix_in_config():
    """Absolute prefix inside workspace is normalized to relative and works."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/notes.md", "abs prefix\n")
        _seed(Path(tmp), "other/x.md", "no\n")
        store = await _make_store()
        abs_prefix = str(Path(tmp).resolve() / "daily")
        kw = {"white_path_prefix": [abs_prefix]}

        resp = await _read(store, step_kwargs=kw, path="daily/notes.md")
        assert resp.success is True

        resp = await _read(store, step_kwargs=kw, path="other/x.md")
        assert resp.success is False
        await store.close()


# -- WriteStep permission tests -------------------------------------------------


@pytest.mark.asyncio
async def test_write_white_prefix_allows_and_denies():
    """white=["daily"] allows write to daily/... but denies daily-report/..."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        kw = {"white_path_prefix": ["daily"]}

        resp = await _write(store, step_kwargs=kw, path="daily/notes.md", content="ok")
        assert resp.success is True
        assert (Path(tmp) / "daily" / "notes.md").exists()

        resp = await _write(store, step_kwargs=kw, path="daily-report/x.md", content="leak")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        assert not (Path(tmp) / "daily-report" / "x.md").exists()
        await store.close()


@pytest.mark.asyncio
async def test_write_black_prefix_denies():
    """black=["secret"] denies write to secret/... but allows other paths."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        kw = {"black_path_prefix": ["secret"]}

        resp = await _write(store, step_kwargs=kw, path="secret/x.md", content="hidden")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()

        resp = await _write(store, step_kwargs=kw, path="notes.md", content="ok")
        assert resp.success is True
        assert (Path(tmp) / "notes.md").exists()
        await store.close()


@pytest.mark.asyncio
async def test_write_white_then_black_combined():
    """white=["daily"] + black=["daily/secret"] — allows daily/notes, denies daily/secret/."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        kw = {"white_path_prefix": ["daily"], "black_path_prefix": ["daily/secret"]}

        resp = await _write(store, step_kwargs=kw, path="daily/notes.md", content="ok")
        assert resp.success is True

        resp = await _write(store, step_kwargs=kw, path="daily/secret/x.md", content="hidden")
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()

        resp = await _write(store, step_kwargs=kw, path="other/x.md", content="no")
        assert resp.success is False  # blocked by white stage
        await store.close()


@pytest.mark.asyncio
async def test_write_no_filters_allows_all():
    """Default (no white/black configured) imposes no restriction on writes."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _write(store, path="anywhere/x.md", content="free")
        assert resp.success is True
        assert (Path(tmp) / "anywhere" / "x.md").exists()
        await store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
