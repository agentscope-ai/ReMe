"""Tests for ReadStep white/black path-prefix permission filtering.

Focus: path-component boundary correctness. A prefix ``"daily"`` must match
``daily/notes.md`` but NOT ``daily-report/notes.md``. This is the regression
suite for the ``str.startswith`` -> path-aware ``_under_prefix`` fix; the old
implementation let a whitelist of ``["daily"]`` leak ``daily-report/...``
through (and a blacklist of ``["secret"]`` wrongly denied ``secret-notes/...``).
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme.components.file_store import LocalFileStore
from reme.steps.file_io import read as crud_read

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


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


def _run(coro):
    asyncio.run(coro)


def _seed(workspace_dir: Path, rel: str, body: str = "body\n") -> Path:
    target = workspace_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


async def _make_store(files: dict[str, str] | None = None) -> LocalFileStore:
    """LocalFileStore with files seeded on disk (graph not needed for read perms)."""
    store = LocalFileStore(name="t_perm", embedding_store="")
    await store.start()
    for rel, content in (files or {}).items():
        abs_path = Path.cwd() / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
    return store


async def _read(store: LocalFileStore, *, step_kwargs: dict | None = None, **kwargs):
    """Run a ReadStep; ``step_kwargs`` go to step init (white/black_path_prefix)."""
    step = crud_read.ReadStep(file_store=store, **(step_kwargs or {}))
    await step(**kwargs)
    return step.context.response


# -- white stage: path boundary ---------------------------------------------


def test_white_prefix_respects_path_boundary():
    """white=["daily"] allows daily/notes.md but denies daily-report/secret.md.

    This is the core regression: the old ``startswith`` let daily-report/...
    through because ``"daily-report/secret.md".startswith("daily")`` is True.
    """

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "daily/notes.md", "daily notes\n")
            _seed(Path(tmp), "daily-report/secret.md", "leaked\n")
            store = await _make_store()
            resp_ok = await _read(store, step_kwargs={"white_path_prefix": ["daily"]}, path="daily/notes.md")
            assert resp_ok.success is True
            assert "daily notes" in str(resp_ok.answer)

            resp_leak = await _read(store, step_kwargs={"white_path_prefix": ["daily"]}, path="daily-report/secret.md")
            assert resp_leak.success is False
            assert "no permission" in str(resp_leak.answer).lower()
            await store.close()
        print("✓ test_white_prefix_respects_path_boundary passed")

    _run(run())


def test_white_prefix_allows_deeply_nested():
    """white=["daily"] allows files nested arbitrarily deep under daily/."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "daily/sub/deep.md", "deep\n")
            store = await _make_store()
            resp = await _read(store, step_kwargs={"white_path_prefix": ["daily"]}, path="daily/sub/deep.md")
            assert resp.success is True
            assert "deep" in str(resp.answer)
            await store.close()
        print("✓ test_white_prefix_allows_deeply_nested passed")

    _run(run())


def test_white_prefix_exact_file_match():
    """white=["foo/bar.md"] allows that exact file but denies a sibling."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "foo/bar.md", "bar\n")
            _seed(Path(tmp), "foo/baz.md", "baz\n")
            store = await _make_store()
            resp_ok = await _read(store, step_kwargs={"white_path_prefix": ["foo/bar.md"]}, path="foo/bar.md")
            assert resp_ok.success is True

            resp_no = await _read(store, step_kwargs={"white_path_prefix": ["foo/bar.md"]}, path="foo/baz.md")
            assert resp_no.success is False
            await store.close()
        print("✓ test_white_prefix_exact_file_match passed")

    _run(run())


def test_white_empty_list_denies_all():
    """white=[] (empty list) denies every file — distinct from None (inactive)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "a.md", "a\n")
            store = await _make_store()
            resp = await _read(store, step_kwargs={"white_path_prefix": []}, path="a.md")
            assert resp.success is False
            assert "no permission" in str(resp.answer).lower()
            await store.close()
        print("✓ test_white_empty_list_denies_all passed")

    _run(run())


# -- black stage: path boundary ---------------------------------------------


def test_black_prefix_respects_path_boundary():
    """black=["secret"] denies secret/x.md but allows secret-notes/public.md.

    Regression: the old ``startswith`` wrongly denied secret-notes/... because
    ``"secret-notes/public.md".startswith("secret")`` is True.
    """

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "secret/x.md", "hidden\n")
            _seed(Path(tmp), "secret-notes/public.md", "public\n")
            store = await _make_store()
            resp_deny = await _read(store, step_kwargs={"black_path_prefix": ["secret"]}, path="secret/x.md")
            assert resp_deny.success is False
            assert "no permission" in str(resp_deny.answer).lower()

            resp_ok = await _read(store, step_kwargs={"black_path_prefix": ["secret"]}, path="secret-notes/public.md")
            assert resp_ok.success is True
            assert "public" in str(resp_ok.answer)
            await store.close()
        print("✓ test_black_prefix_respects_path_boundary passed")

    _run(run())


def test_black_empty_list_is_inactive():
    """black=[] is inactive (same as None) — files still readable."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "a.md", "a\n")
            store = await _make_store()
            resp = await _read(store, step_kwargs={"black_path_prefix": []}, path="a.md")
            assert resp.success is True
            await store.close()
        print("✓ test_black_empty_list_is_inactive passed")

    _run(run())


# -- two-stage combination --------------------------------------------------


def test_white_then_black_combined():
    """white=["daily"] then black=["daily/secret"] — secret subdir still denied."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "daily/notes.md", "ok\n")
            _seed(Path(tmp), "daily/secret/x.md", "hidden\n")
            store = await _make_store()
            step_kwargs = {"white_path_prefix": ["daily"], "black_path_prefix": ["daily/secret"]}

            resp_ok = await _read(store, step_kwargs=step_kwargs, path="daily/notes.md")
            assert resp_ok.success is True

            resp_deny = await _read(store, step_kwargs=step_kwargs, path="daily/secret/x.md")
            assert resp_deny.success is False
            assert "no permission" in str(resp_deny.answer).lower()
            await store.close()
        print("✓ test_white_then_black_combined passed")

    _run(run())


def test_no_filters_allows_all():
    """Default (no white/black configured) imposes no restriction."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed(Path(tmp), "anywhere/x.md", "free\n")
            store = await _make_store()
            resp = await _read(store, path="anywhere/x.md")
            assert resp.success is True
            await store.close()
        print("✓ test_no_filters_allows_all passed")

    _run(run())


if __name__ == "__main__":
    test_white_prefix_respects_path_boundary()
    test_white_prefix_allows_deeply_nested()
    test_white_prefix_exact_file_match()
    test_white_empty_list_denies_all()
    test_black_prefix_respects_path_boundary()
    test_black_empty_list_is_inactive()
    test_white_then_black_combined()
    test_no_filters_allows_all()
