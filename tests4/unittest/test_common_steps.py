"""End-to-end tests for reme4 common steps: spawn `reme4 start`, drive via HTTP,
verify responses, then shut down. Each test uses an isolated cwd so the working_dir
(.reme by default) does not collide.
"""

import asyncio
import os
import tempfile
import warnings

from reme4 import __version__ as REME_VERSION
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


# ---------------------------------------------------------------------------
# Individual job tests
# ---------------------------------------------------------------------------


def test_version_job():
    """version job should return the package version string."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "version",
                    host=host,
                    port=port,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and r.get("answer") == REME_VERSION
                        and r.get("metadata", {}).get("version") == REME_VERSION
                    ),
                )
        print("✓ test_version_job passed")

    _run(run())


def test_help_job():
    """help job should list jobs except itself."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                result = await call_and_check(
                    "help",
                    host=host,
                    port=port,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and isinstance(r.get("answer"), str)
                        and r.get("metadata", {}).get("job_count", 0) > 0
                        and "help" not in r["answer"]
                    ),
                )
                # Spot-check that a couple of known jobs appear in the listing.
                answer = result["answer"]
                for expected_job in ("version", "health_check", "search"):
                    if expected_job not in answer:
                        raise AssertionError(f"help output missing job {expected_job!r}: {answer!r}")
        print("✓ test_help_job passed")

    _run(run())


def test_search_job_empty_store():
    """search on an empty store should return successfully with zero results."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "search",
                    host=host,
                    port=port,
                    query="hello world",
                    limit=5,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and isinstance(r.get("metadata"), dict)
                        and isinstance(r["metadata"].get("counts"), dict)
                        and r["metadata"]["counts"].get("returned", -1) == 0
                    ),
                )
        print("✓ test_search_job_empty_store passed")

    _run(run())


def test_search_job_missing_query():
    """search without a query should surface the assertion error in `answer`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                result = await call_action("search", host=host, port=port, query="")
                if not isinstance(result, dict):
                    raise AssertionError(f"expected dict response, got {result!r}")
                if "query" not in str(result.get("answer", "")).lower():
                    raise AssertionError(f"expected query-related error in answer, got {result!r}")
        print("✓ test_search_job_missing_query passed")

    _run(run())


def test_demo_job():
    """demo job should echo back the normalized query and adjusted min_score."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "demo",
                    host=host,
                    port=port,
                    query="  Hello World  ",
                    min_score=0.8,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and "hello world" in str(r.get("answer", ""))
                        and abs(r.get("metadata", {}).get("adjusted_min_score", 0) - 0.72) < 1e-6
                    ),
                )
        print("✓ test_demo_job passed")

    _run(run())


if __name__ == "__main__":
    print("\n=== reme4 common steps E2E tests ===")
    test_version_job()
    test_help_job()
    test_search_job_empty_store()
    test_search_job_missing_query()
    test_demo_job()
    print("\n所有测试通过!")
