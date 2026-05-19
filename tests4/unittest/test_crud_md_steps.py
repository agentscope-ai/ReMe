"""End-to-end tests for reme4 crud_md steps: spawn `reme4 start`, drive via HTTP,
verify responses, then shut down. Each test uses an isolated cwd so the working_dir
(.reme by default) does not collide.

CLI rule: `path=` is relative-only, rooted at the reme working_dir. A bare path with
no suffix auto-appends `.md`; non-`.md` suffix is rejected. Absolute paths are
rejected.
"""

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme4.utils import call_action, call_and_check, mock_reme_server

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class _temp_chdir:
    """chdir to path for the duration of the block; restore on exit."""

    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


def _run(coro):
    """Run an async coroutine on a fresh isolated event loop."""
    asyncio.run(coro)


def _seed_md(working_dir: Path, rel: str, body: str) -> Path:
    target = working_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Individual job tests
# ---------------------------------------------------------------------------


def test_read_relative_path():
    """`reme4 read path=Templates/Recipe.md` returns the file body from .reme/."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            body = "# Recipe\n\nMix flour and water.\n"
            _seed_md(working, "Templates/Recipe.md", body)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Templates/Recipe.md",
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "# Recipe" in str(r.get("answer", ""))
                        and "flour and water" in str(r.get("answer", ""))
                    ),
                )
        print("✓ test_read_relative_path passed")

    _run(run())


def test_read_no_suffix_autoappends_md():
    """A bare path with no suffix auto-appends `.md`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Templates/Recipe.md", "auto-md\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Templates/Recipe",
                    validator=lambda r: (
                        isinstance(r, dict) and r.get("success") is True and "auto-md" in str(r.get("answer", ""))
                    ),
                )
        print("✓ test_read_no_suffix_autoappends_md passed")

    _run(run())


def test_read_line_range():
    """start_line / end_line slice the file 1-based, inclusive."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Notes.md", "L1\nL2\nL3\nL4\nL5\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Notes.md",
                    start_line=2,
                    end_line=4,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "L2" in str(r["answer"])
                        and "L3" in str(r["answer"])
                        and "L4" in str(r["answer"])
                        and "L1" not in str(r["answer"])
                        and "L5" not in str(r["answer"])
                    ),
                )
        print("✓ test_read_line_range passed")

    _run(run())


def test_read_absolute_path_rejected():
    """Absolute paths are rejected (relative-only after refactor)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            target = _seed_md(working, "Abs.md", "x\n")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path=str(target.resolve()),
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "absolute" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected absolute-path rejection, got {result!r}")
        print("✓ test_read_absolute_path_rejected passed")

    _run(run())


def test_read_non_md_rejected():
    """Paths whose suffix is not `.md` are rejected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="data/foo.txt",
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "markdown" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected markdown-only rejection, got {result!r}")
        print("✓ test_read_non_md_rejected passed")

    _run(run())


def test_read_missing_file():
    """Reading a non-existent file should fail with a clear error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="NotThere.md",
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "does not exist" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected missing-file rejection, got {result!r}")
        print("✓ test_read_missing_file passed")

    _run(run())


def test_read_start_after_end():
    """start_line > end_line is invalid."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Range.md", "a\nb\nc\n")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="Range.md",
                    start_line=3,
                    end_line=1,
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "start_line" in str(result.get("answer", ""))
                ):
                    raise AssertionError(f"expected start>end rejection, got {result!r}")
        print("✓ test_read_start_after_end passed")

    _run(run())


def test_read_start_line_exceeds_total():
    """start_line beyond total line count is invalid."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Short.md", "only-one-line\n")
            async with mock_reme_server() as (host, port):
                result = await call_action(
                    "read",
                    host=host,
                    port=port,
                    path="Short.md",
                    start_line=99,
                )
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "exceeds" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected exceeds-length rejection, got {result!r}")
        print("✓ test_read_start_line_exceeds_total passed")

    _run(run())


def test_read_truncation():
    """A small max_bytes triggers truncation with a continuation notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            body = "\n".join(f"line {i}" for i in range(200)) + "\n"
            _seed_md(working, "Big.md", body)
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Big.md",
                    max_bytes=64,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "truncated" in str(r["answer"])
                        and "start_line=" in str(r["answer"])
                    ),
                )
        print("✓ test_read_truncation passed")

    _run(run())


def test_read_empty_path_rejected():
    """An empty `path` should be rejected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            async with mock_reme_server() as (host, port):
                result = await call_action("read", host=host, port=port, path="")
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                    and "required" in str(result.get("answer", "")).lower()
                ):
                    raise AssertionError(f"expected `path` required rejection, got {result!r}")
        print("✓ test_read_empty_path_rejected passed")

    _run(run())


# ---------------------------------------------------------------------------
# Aggregate test: reuse one server instance for all read cases (faster).
# ---------------------------------------------------------------------------


def test_all_read_cases_one_server():
    """Run multiple read scenarios against a single shared server for efficiency."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            working.mkdir(parents=True, exist_ok=True)
            _seed_md(working, "Templates/Recipe.md", "# Recipe\nbody\n")
            _seed_md(working, "Notes.md", "L1\nL2\nL3\n")
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Templates/Recipe.md",
                    validator=lambda r: r.get("success") is True and "# Recipe" in r["answer"],
                )
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Notes",
                    validator=lambda r: r.get("success") is True and "L1" in r["answer"],
                )
                await call_and_check(
                    "read",
                    host=host,
                    port=port,
                    path="Notes.md",
                    start_line=2,
                    end_line=2,
                    validator=lambda r: r.get("success") is True and r["answer"].strip() == "L2",
                )
        print("✓ test_all_read_cases_one_server passed")

    _run(run())


if __name__ == "__main__":
    print("\n=== reme4 crud_md (read) E2E tests ===")
    test_read_relative_path()
    test_read_no_suffix_autoappends_md()
    test_read_line_range()
    test_read_absolute_path_rejected()
    test_read_non_md_rejected()
    test_read_missing_file()
    test_read_start_after_end()
    test_read_start_line_exceeds_total()
    test_read_truncation()
    test_read_empty_path_rejected()
    test_all_read_cases_one_server()
    print("\n所有测试通过!")
